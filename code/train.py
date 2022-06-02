from distutils.log import debug
import logging
import os
from typing import Generator, Iterable, List, Tuple

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch import Tensor, optim
from torch.utils.data import DataLoader
from torchvision.ops import box_convert
from torchvision.utils import draw_bounding_boxes
from tqdm import tqdm

from dataset import FasterRCNNCOCODataset
from model import FasterRCNN, faster_rcnn_vgg16
from utils import applyIndex, boxIou, calculatedxywh

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TempOut():

    def __init__(self, dataset: FasterRCNNCOCODataset, outputItems: List[str]) -> None:
        self.dataset = dataset
        self.outputItems = outputItems

    def __enter__(self):
        self.dataset.outputItems, self.outputItems = self.outputItems, self.dataset.outputItems

    def __exit__(self,*args):
        self.dataset.outputItems, self.outputItems = self.outputItems, self.dataset.outputItems


class FasterTrainer:

    def __init__(self) -> None:
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.anchors = self.get_anchors()
        self.train_data = FasterRCNNCOCODataset(root='COCO/train2017', annFile='COCO/instances_train2017.json', anchors=self.anchors, gridsize=(32, 32), imagesize=(512, 512), imagetransformers=None, outputItems=[])
        self.train_loader = None
        self.model = faster_rcnn_vgg16(self.train_data.anchors.to(self.device), (512, 512), 91) # COCO类别为1-90,0作为没有检测到目标的标签
        self.model = self.model.to(self.device)
        self.epochs = [5, 5]  #[100, 100]
        self.save_frequency = 5
        self.model_save_path = './models'
        not os.path.exists(self.model_save_path) and os.makedirs(self.model_save_path)
        self.match_threshold = 0.5

    def train_rpn_layer(self) -> None:
        self.model.stage = 'train RPN layer'
        self.train_loader = DataLoader(self.train_data, batch_size=16, shuffle=True, num_workers=2)
        ...
        self.model.backbone.requires_grad_(False)
        self.model.classifier_bbox_pred.requires_grad_(False)
        self.model.pre_pred.requires_grad_(False)
        self.model.rpn_layer.requires_grad_(True)
        ...
        optimizer = optim.Adam(self.model.rpn_layer.parameters())
        lr_s = optim.lr_scheduler.StepLR(optimizer, self.epochs[0] // 3, 0.1)
        ...
        with TempOut(self.train_data, ['image', 'positive_negative', 'regression']):
            for epoch in range(self.epochs[0]):
                bar = tqdm(self.train_loader)
                acc_list, loss_list = [], []
                for image, positive_negative, regression in self.datas_to_device(bar):
                    pred_reg, pred_pn = self.model(image)  # [N,H,W,K,4] [N,H,W,K,2]
                    # 正负例平衡采样(1:2)
                    positive, negative = self.balance_positive_negative(positive_negative)  # [N,H,W,K]
                    # box变化回归损失
                    loss_reg = self.pos_weight_smooth_l1_loss(positive, pred_reg, regression)
                    logger.debug(f'{positive.shape},{negative.shape},{pred_pn.shape}')
                    # 正负例分类损失
                    loss_cls = self.pos_weight_crossentrpy(positive + negative, pred_pn, positive)
                    ...
                    loss = loss_reg + loss_cls
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    ...
                    # 显示当前损失,可能会出现损失无穷大.这是由于没有匹配的目标框导致没有用于计算的正负例样本,求平均时除以0导致的,但是不会影响训练,batch_size越大其出现的概率越小
                    bar.set_description(f'loss cls:{loss_cls.item():.4f} loss reg:{loss_reg.item():.4f}')
                    acc = (torch.argmax(pred_pn, dim=-1) == positive)[(positive + negative) != 0].float().mean().item()
                    acc_list.append(acc)
                    loss_list.append([loss_cls.item(), loss_reg.item(), loss.item()])
                ...
                if (epoch + 1) % self.save_frequency == 0:
                    mean_loss = torch.tensor(loss_list).mean(dim=0)
                    logger.info(f'epoch {epoch+1}: accuracy={torch.tensor(acc_list).mean().item():.4f} , loss cls={mean_loss[0].item():.4f} , loss reg={mean_loss[1].item():.4f} , total loss={mean_loss[2].item():.4f}')
                    torch.save(self.model.state_dict(), os.path.join(self.model_save_path, f'model_train_rpn_{epoch+1}.pth'))
                lr_s.step()

    def get_anchors(self) -> Tensor:
        anchors = torch.zeros((12, 2))
        areas = torch.tensor([64**2, 128**2, 256**2, 512**2])
        wh21 = torch.sqrt(areas / 2)
        anchors[:4, 0] = wh21
        anchors[:4, 1] = 2 * wh21
        anchors[4:8, 0] = 2 * wh21
        anchors[4:8, 1] = wh21
        anchors[8:, 0] = anchors[8:, 1] = torch.sqrt(areas)
        return anchors

    def to_device(self, *args) -> List[Tensor]:
        return [arg.to(self.device) for arg in args]

    def datas_to_device(self, datas: Iterable) -> Generator[List[Tensor], None, None]:
        for data in datas:
            yield self.to_device(*data)

    def pos_weight_crossentrpy(self, weight: Tensor, pred: Tensor, target: Tensor) -> Tensor:
        # logger.debug(F.cross_entropy(pred.permute(0, 4, 1, 2, 3), target, reduction='none').shape)
        # logger.debug(weight.shape)
        # return (weight * F.cross_entropy(pred.permute(0, 4, 1, 2, 3), target, reduction='none')).mean()
        return F.cross_entropy(pred.permute(0, 4, 1, 2, 3), target, reduction='none')[weight.bool()].mean()

    def pos_weight_smooth_l1_loss(self, weight: Tensor, pred: Tensor, target: Tensor) -> Tensor:
        return F.smooth_l1_loss(pred, target, reduction='none')[weight.bool()].mean()

    def balance_positive_negative(self, positive_negative: Tensor) -> Tuple[Tensor, Tensor]:
        positive, negative = positive_negative[:, 0], positive_negative[:, 1]
        p = positive.sum() / negative.sum()  # 求出采样概率,目标是1:1
        sampling = torch.rand(negative.shape).to(self.device) < 2 * p
        return positive, sampling.long() * negative

    def train_classifier(self) -> None:
        self.model.classifier_bbox_pred.requires_grad_(True)
        self.model.backbone.requires_grad_(False)
        self.model.pre_pred.requires_grad_(True)
        self.model.rpn_layer.requires_grad_(False)
        ...
        self.train_loader = DataLoader(self.train_data, batch_size=1, shuffle=True, num_workers=2)
        optimizer = optim.Adam(self.model.rpn_layer.parameters())
        lr_s = optim.lr_scheduler.StepLR(optimizer, self.epochs[1] // 3, 0.1)
        self.model.stage = 'train classifier'
        ...
        with TempOut(self.train_data, ['image', 'bboxes', 'category_ids']):
            for epoch in range(self.epochs[1]):
                bar = tqdm(self.train_loader)
                acc_list, loss_list = [], []
                for image, bboxes, category_ids in self.datas_to_device(bar):
                    pred_bboxes, cls_pred, bbox_pred = self.model(image)
                    # 求预测框与标签的匹配, pred_indices是布尔索引表示是否有与之匹配的标签,bbox_pred是数值索引表示与预测框匹配的标签的下标
                    bboxes, category_ids = applyIndex(0, bboxes, category_ids)
                    try:
                        pred_indices, label_indices = self.match_pred_label(pred_bboxes, bboxes, self.match_threshold)
                    except:
                        continue
                    if not pred_indices.any():
                        continue
                    ...
                    logger.debug(pred_indices, label_indices)
                    bbox_pred = applyIndex(pred_indices, bbox_pred)
                    bboxes = applyIndex(label_indices, bboxes)
                    logger.debug(bbox_pred,bboxes)
                    label_cls = torch.zeros(cls_pred.shape[0]).to(self.device).long()
                    label_cls[pred_indices] = category_ids[label_indices]
                    label_reg = calculatedxywh(box_convert(bbox_pred,'xyxy','xywh'), box_convert(bboxes,'xyxy','xywh'))  # 计算第二次回归的标签
                    logger.debug(f'loabel reg {label_reg}')
                    ...
                    loss_cls = F.cross_entropy(cls_pred, label_cls)
                    loss_reg = F.smooth_l1_loss(bbox_pred, label_reg)
                    loss = loss_reg + loss_cls
                    ...
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    ...
                    bar.set_description(f'loss cls:{loss_cls.item():.4f} loss reg:{loss_reg.item():.4f}')
                    acc = (torch.argmax(cls_pred, dim=-1) == label_cls).float().mean().item()
                    acc_list.append(acc)
                    loss_list.append([loss_cls.item(), loss_reg.item(), loss.item()])
                ...
                if (epoch + 1) % self.save_frequency == 0:
                    mean_loss = torch.tensor(loss_list).mean(dim=0)
                    logger.info(f'epoch {epoch+1}: accuracy={torch.tensor(acc_list).mean().item():.4f} , loss cls={mean_loss[0].item():.4f} , loss reg={mean_loss[1].item():.4f} , total loss={mean_loss[2].item():.4f}')
                    torch.save(self.model.state_dict(), os.path.join(self.model_save_path, f'model_train_classifier_{epoch+1}.pth'))
                lr_s.step()

    def match_pred_label(self, pred_box: Tensor, label_box: Tensor, threshold: float) -> Tuple[Tensor, Tensor]:
        if 0 in pred_box.shape or 0 in label_box.shape:
            raise Exception('zero boxes')
        logger.debug(pred_box, label_box)
        ious = boxIou(pred_box, label_box)
        max_iou, indices = ious.max(dim=-1)
        bool_indices = max_iou > threshold
        return bool_indices, indices[bool_indices]

    def chack_dataset(self, index: int, save_path: str):
        with TempOut(self.train_data, ['image', 'bboxes', 'category_ids']):
            img, boxes, cids = self.train_data[index]
            img = TF.to_pil_image(img)
            save_img = draw_bounding_boxes(image=TF.pil_to_tensor(img), boxes=boxes, labels=[str(d.item()) for d in cids])
            TF.to_pil_image(save_img).save(os.path.join(save_path, f'{index}.png'))
        with TempOut(self.train_data,['bboxes','positive_negative','regression']):
            bboxes,pn,reg = self.train_data[index]
            logger.debug(bboxes)
            logger.debug(self.train_data.anchors[pn[0].bool()])
    def chack_stage1_output(self,index:int):
        with TempOut(self.train_data, ['image', 'bboxes']):
            img, bboxes= self.train_data[index]
            self.model.stage = 'train classifier'
            pred_boxes,_,_ = self.model(img.unsqueeze(0).to(self.device))
            pred_boxes=pred_boxes[0].cpu()
            logger.debug(bboxes)
            logger.debug(pred_boxes)
        ...


if __name__ == '__main__':
    torch.manual_seed(0)
    trainer = FasterTrainer()
    trainer.chack_dataset(11,'.')
    trainer.model.load_state_dict(torch.load('models/model_train_rpn_5.pth'))
    trainer.train_rpn_layer()
    trainer.chack_stage1_output(11)
    trainer.train_classifier()
