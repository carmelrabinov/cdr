import torch
import numpy as np
from typing import List


def calc_IOU(seg_omg1: torch.BoolTensor, seg_omg2: torch.BoolTensor, eps: float = 1.e-6) -> float:
    """
    calculate intersection over union between 2 boolean segmentation masks
    :param seg_omg1: first segmentation mask
    :param seg_omg2: second segmentation mask
    :param eps: eps for numerical stability
    :return: IOU
    """
    dim = [1, 2, 3] if len(seg_omg1.shape) == 4 else [1, 2]
    intersection = (seg_omg1 & seg_omg2).sum(dim=dim)
    union = (seg_omg1 | seg_omg2).sum(dim=dim)
    return (intersection.float() / (union.float() + eps)).mean().item()


def calc_mean_iou(ref_seg_mask: torch.BoolTensor, seg_masks: List[torch.BoolTensor]) -> [float, List[float]]:
    """calculate mean IOU for k-nearest neighbours"""
    scores = [calc_IOU(ref_seg_mask, seg_mask) for seg_mask in seg_masks]
    return np.mean(scores), scores
