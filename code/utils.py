from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, no_grad


@no_grad()
def boxIou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    '''
    Args:
        boxes1:[...,4] such as [10,12,4]
        boxes1:[...,4] such as [10,12,4]
    Returns:
        iou:[boxes1.shape[:-1],boxes2.shape[:-1]] such as [10,12,10,12]

    相比于torchvision的box_iou函数,这个实现不需要在运算前后reshape
    '''
    area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
    lenboxes2shape = len(boxes2.shape)
    lt = torch.max(eval('boxes1[..., ' + ','.join(['None' for _ in range(lenboxes2shape - 1)]) + ', :2]'), boxes2[..., :2])  # [N,M,2]
    rb = torch.min(eval('boxes1[..., ' + ','.join(['None' for _ in range(lenboxes2shape - 1)]) + ', 2:]'), boxes2[..., 2:])  # [N,M,2]
    wh = (rb - lt).clamp(min=0)  # 去除为负数的长宽
    inter = wh[..., 0] * wh[..., 1]  # [N,M]
    union = eval('area1[..., ' + ','.join(['None' for _ in range(lenboxes2shape - 1)]) + ']') + area2 - inter
    return inter / union


@no_grad()
def resizeBox(orgsize: Tuple[int, int], tarsize: Tuple[int, int], boxes: Tensor) -> Tensor:
    matrix = torch.diag(Tensor(tarsize) / Tensor(orgsize)).to(boxes.device)  # 长宽变化比例
    return torch.cat([boxes[..., :2] @ matrix, boxes[..., 2:] @ matrix], -1)


@no_grad()
def tileAnchors(anchors: Tensor, gridsize: Tuple[int, int], imagesize: Tuple[int, int]) -> Tensor:
    baselength = Tensor(imagesize) / Tensor(gridsize)
    # 平铺时的网格中心坐标 [H,W,k,2]
    baseposition = torch.stack(torch.meshgrid(torch.arange(baselength[0] / 2, imagesize[0], baselength[1]), torch.arange(baselength[1] / 2, imagesize[1], baselength[1]), indexing='xy'), dim=-1)  # (H,W,2)
    baseanchors = torch.zeros(*baseposition.shape[:2], anchors.shape[0], 4)
    baseanchors[..., :2] = baseposition.unsqueeze(-2) - anchors / 2
    baseanchors[..., 2:] = baseposition.unsqueeze(-2) + anchors / 2
    return baseanchors  # [H,W,k,4]


@no_grad()
def calculatedxywh(anchors: Tensor, boxes: Tensor) -> Tensor:
    '''
    anchors:[...,4] boxes:[...,4]
    '''
    dxywh = torch.zeros_like(anchors)
    dxywh[..., :2] = (boxes[..., :2] - anchors[..., :2]) / anchors[..., 2:]
    dxywh[..., 2:] = torch.log(boxes[..., 2:] / anchors[..., 2:])
    dxywh = dxywh.nan_to_num(0)  # 当分母为0时会计算得到无穷大,在后续计算当中会导致损失无穷大
    return dxywh


@no_grad()
def applydxywh(anchors: Tensor, dxywh: Tensor) -> Tensor:
    '''
    anchors:[...,4] dxywh:[...,4]
    box_xyxy=[x,y,x',y']^T box_xywh=[x,y,w,h]^T
    box_xywh=⌈ 1, 0, 0, 0⌉  ⌈x ⌉ box_xyxy=⌈ 1, 0, 0, 0⌉ ⌈x⌉
-------------| 0, 1, 0, 0| |y |         | 0, 1, 0, 0| |y|
-------------|-1, 0, 1, 0| |x'|         | 1, 0, 1, 0| |w|
-------------⌊ 0,-1, 0, 1⌋  ⌊y'⌋          ⌊ 0, 1, 0, 1⌋ ⌊h⌋ 
    box_xywh'=⌈ 1, 0,dx, 0⌉ ⌈x⌉
--------------| 0, 1, 0,dy||y|
--------------| 0, 0,dw, 0||w|
--------------⌊ 0, 0, 0,dh⌋ ⌊h⌋ 
    '''
    matrix = torch.zeros(*dxywh.shape, 4).to(anchors.device)
    matrix[..., :2, 2:] = dxywh[..., :2].diag_embed()
    matrix[..., :2, :2] = torch.tensor([1, 1]).float().diag_embed().to(dxywh.device) - matrix[..., :2, 2:]
    matrix[..., 2:, 2:] = torch.exp(dxywh[..., 2:]).diag_embed() + matrix[..., :2, 2:]
    matrix[..., 2:, :2] = torch.tensor([1, 1]).float().diag_embed().to(dxywh.device) - matrix[..., 2:, 2:]
    anchors = (matrix @ anchors.unsqueeze(-1)).squeeze(-1)
    return anchors


def applyIndex(indices: Tensor, *args) -> Tuple[Tensor, ...]:
    if len(args) == 1:
        return args[0][indices]
    return tuple([_[indices] for _ in args])


# 实现仅供参考,没有使用到
@no_grad()
def nms(boxes: Tensor, score: Tensor, threshold: float) -> Tensor:
    # boxes:(N,4) score:(N)
    _, indices = score.sort(descending=True)
    result = []
    while 0 not in indices.shape:
        result.append(indices[0])
        temp_boxes = boxes[indices]
        ious = boxIou(temp_boxes[:1], temp_boxes[1:])[0]
        indices = indices[1:][ious <= threshold]
    return torch.stack(result)