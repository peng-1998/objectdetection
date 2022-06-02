from typing import Callable, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import Module
from torchvision.models import resnet, vgg
from torchvision.ops import clip_boxes_to_image, nms, remove_small_boxes

from utils import applydxywh, applyIndex, resizeBox


class RPNLayer(Module):

    def __init__(self, in_channels, num_anchors) -> None:
        super().__init__()
        self.num_anchors = num_anchors
        self.conv3x3 = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.bnlayer = nn.BatchNorm2d(in_channels)
        self.conv1x1 = nn.Conv2d(in_channels, 6 * num_anchors, 1)

    def forward(self, input: Tensor) -> Tuple[Tensor, Tensor]:
        output: Tensor = self.conv3x3(input)
        output: Tensor = self.bnlayer(output)
        output: Tensor = self.conv1x1(output)
        return output.split([self.num_anchors * 4, self.num_anchors * 2], dim=1)


class Proposal(Module):

    def __init__(self, anchors: Tensor, max_num_box: int, image_size: Tuple[int, int], min_box_size: Union[int, float], nms_threshold: float) -> None:
        super().__init__()
        self.anchors = anchors  # [H,W,k,4] 'xyxy
        self.max_num_box = max_num_box
        self.image_size = image_size
        self.min_box_size = min_box_size
        self.nms_threshold = nms_threshold

    def forward(self, score: Tensor, box_regression: Tensor) -> Tensor:
        # score:[H,W,k,2] box_regression:[H,W,k,4]
        indices: Tensor = score[..., 1] > 0.5  # [H,W,k]
        boxes, box_regression, score = applyIndex(indices, self.anchors, box_regression, score[..., 1])
        if score.shape[0] > self.max_num_box:
            boxes, box_regression, score = applyIndex(score.topk(self.max_num_box).indices, boxes, box_regression, score)
        boxes = applydxywh(boxes, box_regression)
        boxes = clip_boxes_to_image(boxes, self.image_size)
        boxes, score = applyIndex(remove_small_boxes(boxes, self.min_box_size), boxes, score)
        return boxes[nms(boxes, score, self.nms_threshold)].clone()


class ROIPoolingLayer(Module):

    def __init__(self, image_size: Tuple[int, int], output_size: Tuple[int, int]) -> None:
        super().__init__()
        self.image_size = image_size
        self.output_size = output_size

    def forward(self, feature_map: Tensor, boxes: Tensor):
        # feature_map:[1,C,H,W] boxes:[N,4]

        # 实现1 使用自适应均匀/最大值池化
        h, w = feature_map.shape[-2:]
        boxes = resizeBox(self.image_size, (w, h), boxes).floor().long()
        boxes[:, 2:] = boxes[:, 2:] + 1
        roi_area = torch.cat([F.adaptive_avg_pool2d(feature_map[..., box[1]:box[3], box[0]:box[2]], self.output_size) for box in boxes])
        # roi_area = torch.cat([F.adaptive_max_pool2d(feature_map[..., box[1]:box[3], box[0]:box[2]], self.output_size) for box in boxes])
        return roi_area

        # 实现2 使用双线性插值采样
        boxes = resizeBox(self.image_size, (w, h), boxes)
        roi_grid = torch.stack([torch.stack([torch.meshgrid(torch.linspace(box[0], box[2], self.output_size[0]), torch.linspace(box[1], box[3], self.output_size[1]), indexing='xy')], dim=-1) for box in boxes])
        roi_area = F.grid_sample(feature_map.expand(boxes.shape[0], *feature_map.shape[1:]), roi_grid, mode='bilinear')
        return roi_area


class FasterRCNN(Module):

    def __init__(self, backbone: Module, stage: str, **kargs) -> None:
        super().__init__()
        self.backbone = backbone
        self.stage = stage
        if isinstance(self.backbone, vgg.VGG):
            del self.backbone.features[-2:]
            self.backbone = self.backbone.features
            self.rpn_in_channels = 512
        if isinstance(self.backbone, resnet.ResNet):
            self.backbone = nn.Sequential(self.backbone.conv1, self.backbone.bn1, nn.ReLU(), self.backbone.layer1, self.backbone.layer2, self.backbone.layer3, self.backbone.layer4)
            self.rpn_in_channels = 2048
        self.rpn_layer = RPNLayer(self.rpn_in_channels, kargs['anchors'].shape[-2])
        self.proposal_layer = Proposal(kargs['anchors'], kargs['max_num_box'], kargs['image_size'], kargs['min_box_size'], kargs['nms_threshold'])
        self.roi_pooling_layer = ROIPoolingLayer(kargs['image_size'], kargs['roi_size'])
        layers = []
        for _ in range(4):
            layers += [nn.Conv2d(self.rpn_in_channels, self.rpn_in_channels, 3, 1, 1), nn.BatchNorm2d(self.rpn_in_channels), nn.ReLU()]
        self.pre_pred = nn.Sequential(*layers)
        self.classifier_bbox_pred = nn.Sequential(nn.Linear(self.rpn_in_channels * kargs['roi_size'][0] * kargs['roi_size'][1], 4096), nn.ReLU(True), nn.Dropout(p=kargs['dropout']), nn.Linear(4096, 4096), nn.ReLU(True), nn.Dropout(p=kargs['dropout']), nn.Linear(4096, kargs['num_classes'] + 4))

    def forward(self, input: Tensor) -> Union[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor]]:
        feature_map: Tensor = self.backbone(input)
        regression, anchors_cls = self.rpn_layer(feature_map)  # [N,4k,H,W] [N,2k,H,W]
        regression: Tensor = regression.permute(0, 2, 3, 1).reshape(feature_map.shape[0], *self.proposal_layer.anchors.shape)
        anchors_cls: Tensor = anchors_cls.permute(0, 2, 3, 1).reshape(feature_map.shape[0], *self.proposal_layer.anchors.shape[:-1], -1)
        if self.stage == 'train RPN layer':
            return regression, anchors_cls
        assert feature_map.shape[0] == 1
        regression: Tensor = regression[0]
        anchors_cls: Tensor = anchors_cls[0]
        bboxes = self.proposal_layer(anchors_cls.softmax(-1), regression)
        roi_areas = self.roi_pooling_layer(feature_map, bboxes)
        pred_cls_bbox: Tensor = self.classifier_bbox_pred(self.pre_pred(roi_areas).view(roi_areas.shape[0], -1))
        cls_pred, bbox_pred = pred_cls_bbox.split([pred_cls_bbox.shape[1] - 4, 4], dim=1)
        if self.stage == 'train classifier':
            return bboxes, cls_pred, bbox_pred
        cls_pred = cls_pred.argmax(-1)
        return applydxywh(bboxes, bbox_pred)[cls_pred != 0], cls_pred[cls_pred != 0]


def faster_rcnn_vgg16(anchors: Tensor, image_size: Tuple[int, int], num_classes: int, **kargs) -> FasterRCNN:
    args = {'anchors': anchors, 'max_num_box': 2000, 'image_size': image_size, 'roi_size': (7, 7), 'min_box_size': 10, 'nms_threshold': 0.5, 'dropout': 0.5, 'num_classes': num_classes}
    for key in args.keys():
        if key in kargs and kargs[key] != args[key]:
            args[key] = kargs[key]
    return FasterRCNN(vgg.vgg16(True, True), '', **args)
