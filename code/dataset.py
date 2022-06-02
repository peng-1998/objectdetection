from typing import Any, Callable, List, Optional, Tuple

import torch
import torchvision.transforms.functional as TF
from torch import Tensor, no_grad
from torchvision.datasets.coco import CocoDetection
from torchvision.ops import box_convert

from utils import resizeBox, tileAnchors, calculatedxywh, boxIou


class FasterRCNNCOCODataset(CocoDetection):

    def __init__(self, root: str, annFile: str, anchors: Tensor, gridsize: Tuple[int, int], imagesize: Tuple[int, int], imagetransformers: Optional[Callable], outputItems: List[str]) -> None:
        '''
        outputItems item in ['image','bboxes','category_ids','positive_negative','regression']
        '''
        super().__init__(root, annFile)

        self.gridsize = gridsize
        self.imagesize = imagesize
        self.imagetransformers = imagetransformers
        self.outputItems = outputItems
        self.anchorsWH = anchors
        self.anchors = tileAnchors(anchors, gridsize, imagesize)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        image, target = super().__getitem__(index)  # [PIL.Image,List]
        image = TF.to_tensor(image)
        h, w = image.shape[-2:]
        image = TF.resize(image, (self.imagesize[1], self.imagesize[0]))
        if self.imagetransformers:
            image = self.imagetransformers(image)
        bboxes = Tensor([_['bbox'] for _ in target])
        bboxes = resizeBox((w, h), self.imagesize, bboxes) if 0 not in bboxes.shape else bboxes
        bboxes = box_convert(bboxes, 'xywh', 'xyxy') if 0 not in bboxes.shape else bboxes
        category_ids = Tensor([_['category_id'] for _ in target]).long()  # COCO 数据集标签从1开始
        if 'positive_negative' in self.outputItems or 'regression' in self.outputItems:
            positive_negative, regression = self.getPositiveNegativeRegression(bboxes)
        result = []
        for _ in self.outputItems:
            result.append(eval(_))
        return result

    @no_grad()
    def getPositiveNegativeRegression(self, boxes: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        if 0 in boxes.shape:
            return torch.stack([torch.zeros(self.anchors.shape[:-1]), torch.ones(self.anchors.shape[:-1])]).long().clone(), torch.zeros_like(self.anchors).float()
        ious = boxIou(self.anchors, boxes)
        maxiou, indices = ious.max(dim=-1)
        current_gt = boxes[indices]
        return torch.stack([(maxiou > 0.7).long(), (maxiou < 0.3).long()]).clone(), calculatedxywh(box_convert(self.anchors,'xyxy','xywh'),box_convert(current_gt,'xyxy','xywh'))
