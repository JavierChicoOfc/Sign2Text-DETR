from typing import Any, List, Tuple

import torch
from torch import Tensor


def box_cxcywh_to_xyxy(x: Tensor) -> Tensor:
    """
    Convert bounding boxes from (center_x, center_y, width, height) format to (x_min, y_min, x_max, y_max) format.

    Args:
        x (Tensor): Tensor of shape (..., 4) containing boxes in (cx, cy, w, h) format.

    Returns:
        Tensor: Converted boxes in (x_min, y_min, x_max, y_max) format.
    """
    x_c, y_c, w, h = x.unbind(-1)
    b = [x_c - 0.5 * w, y_c - 0.5 * h, x_c + 0.5 * w, y_c + 0.5 * h]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x: Tensor) -> Tensor:
    """
    Convert bounding boxes from (x_min, y_min, x_max, y_max) format to (center_x, center_y, width, height) format.

    Args:
        x (Tensor): Tensor of shape (..., 4) containing boxes in (x_min, y_min, x_max, y_max) format.

    Returns:
        Tensor: Converted boxes in (center_x, center_y, width, height) format.
    """
    x1, y1, x2, y2 = x.unbind(-1)
    b = [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]
    return torch.stack(b, dim=-1)


def rescale_bboxes(out_bbox: Tensor, size: Tuple[int, int]) -> Tensor:
    """
    Rescale bounding boxes to the given image size.

    Args:
        out_bbox (Tensor): Bounding boxes in (cx, cy, w, h) format, normalized [0, 1].
        size (Tuple[int, int]): Image size as (width, height).

    Returns:
        Tensor: Rescaled bounding boxes in (x_min, y_min, x_max, y_max) format.
    """
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    scale = torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32, device=out_bbox.device)
    return b * scale


def box_area(boxes: Tensor) -> Tensor:
    """
    Compute the area of bounding boxes in (x_min, y_min, x_max, y_max) format.

    Args:
        boxes (Tensor): Tensor of shape (N, 4) containing boxes in (x_min, y_min, x_max, y_max) format.

    Returns:
        Tensor: Area of each box, shape (N,).
    """
    return (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)


def box_iou(boxes1: Tensor, boxes2: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Compute the Intersection over Union (IoU) between two sets of bounding boxes.

    Args:
        boxes1 (Tensor): Tensor of shape (N, 4) in (x_min, y_min, x_max, y_max) format.
        boxes2 (Tensor): Tensor of shape (M, 4) in (x_min, y_min, x_max, y_max) format.

    Returns:
        Tuple[Tensor, Tensor]:
            - IoU matrix of shape (N, M)
            - Union area matrix of shape (N, M)
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2 - inter
    eps = 1e-7
    iou = inter / (union + eps)
    return iou, union


def generalized_box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """
    Compute the Generalized Intersection over Union (GIoU) between two sets of bounding boxes.

    Both sets of boxes are expected to be in (x_min, y_min, x_max, y_max) format.

    Args:
        boxes1 (Tensor): Tensor of shape (N, 4).
        boxes2 (Tensor): Tensor of shape (M, 4).

    Returns:
        Tensor: GIoU matrix of shape (N, M).
    """
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)
    c_area = wh[:, :, 0] * wh[:, :, 1]
    eps = 1e-7
    return iou - (c_area - union) / (c_area + eps)


def stacker(batch: List[Tuple[Tensor, Any]]) -> Tuple[Tensor, List[Any]]:
    """
    Custom collate function for DETR dataloaders.

    Handles batches with varying numbers of objects per image. Images are stacked into a single tensor,
    while targets remain as a list (since each image can have a different number of objects).

    Args:
        batch (List[Tuple[Tensor, Any]]): List of (image, target) tuples.

    Returns:
        Tuple[Tensor, List[Any]]: Stacked images tensor and list of targets.
    """
    images = [image for image, _ in batch]
    targets = [target for _, target in batch]
    images = torch.stack(images, dim=0)
    return images, targets
