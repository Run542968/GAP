# ------------------------------------------------------------------------
# TadTR: End-to-end Temporal Action Detection with Transformer
# Copyright (c) 2021. Xiaolong Liu.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Utilities for segment manipulation and IoU.
"""
import torch
import numpy as np
# from torchvision.ops.boxes import box_area


def segment_cw_to_t1t2(x):
    '''corresponds to box_cxcywh_to_xyxy in detr
    Params:
        x: segments in (center, width) format, shape=(*, 2)
    Returns:
        segments in (t_start, t_end) format, shape=(*, 2)
    '''
    if not isinstance(x, np.ndarray):
        x_c, w = x.unbind(-1)
        b = [(x_c - 0.5 * w), (x_c + 0.5 * w)]
        return torch.stack(b, dim=-1)
    else:
        x_c, w = x[..., 0], x[..., 1]
        b = [(x_c - 0.5 * w)[..., None], (x_c + 0.5 * w)[..., None]]
        return np.concatenate(b, axis=-1)


def segment_t1t2_to_cw(x):
    '''corresponds to box_xyxy_to_cxcywh in detr
    Params:
        x: segments in (t_start, t_end) format, shape=(*, 2)
    Returns:
        segments in (center, width) format, shape=(*, 2)
    '''
    if not isinstance(x, np.ndarray):
        x1, x2 = x.unbind(-1)
        b = [(x1 + x2) / 2, (x2 - x1)]
        return torch.stack(b, dim=-1)
    else:
        x1, x2 = x[..., 0], x[..., 1]
        b = [((x1 + x2) / 2)[..., None], (x2 - x1)[..., None]]
        return np.concatenate(b, axis=-1)


def segment_length(segments):
    return (segments[:, 1]-segments[:, 0]).clamp(min=0)


# modified from torchvision to also return the union
def segment_iou(segments1, segments2):
    """
    Temporal IoU between 

    The segmemts should be in [start, end] format

    Returns a [N, M] pairwise matrix, where N = len(segments1)
    and M = len(segments2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (segments1[:, 1] >= segments1[:, 0]).all(), f"segments1:{segments1}"
    assert (segments2[:, 1] >= segments2[:, 0]).all(), f"segments2:{segments2}"

    area1 = segment_length(segments1) # N
    area2 = segment_length(segments2) # M

    l = torch.max(segments1[:, None, 0], segments2[:, 0])  # N,M
    r = torch.min(segments1[:, None, 1], segments2[:, 1])  # N,M
    inter = (r - l).clamp(min=0)  # [N,M]

    union = area1[:, None] + area2 - inter # N,M

    iou = inter / union

    return iou

def seg_iou(segments1, segments2):
    """
    Temporal IoU between 

    The segmemts should be in [start, end] format

    Returns a [N, M] pairwise matrix, where N = len(segments1)
    and M = len(segments2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (segments1[:, 1] >= segments1[:, 0]).all()
    assert (segments2[:, 1] >= segments2[:, 0]).all()

    area1 = segment_length(segments1) # N
    area2 = segment_length(segments2) # M

    l = torch.max(segments1[:, None, 0], segments2[:, 0])  # N,M
    r = torch.min(segments1[:, None, 1], segments2[:, 1])  # N,M
    inter = (r - l).clamp(min=0)  # [N,M]

    union = area1[:, None] + area2 - inter # N,M

    iou = inter / union

    return iou, union

def generalized_seg_iou(segments1, segments2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The segmemts should be in [start, end] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)

    !!!! NOTE that: for temporal data, then generalized_seg_iou is equal to seg_iou, since the area==union !!!!!
    !!!! NOTE that: so use this may be issue !!!!!!!!!!!!
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (segments1[:, 1] >= segments1[:, 0]).all()
    assert (segments2[:, 1] >= segments2[:, 0]).all()

    iou, union = seg_iou(segments1, segments2)

    l = torch.min(segments1[:, None, 0], segments2[:, 0])  # N,M
    r = torch.max(segments1[:, None, 1], segments2[:, 1])  # N,M

    area = (r - l).clamp(min=0)  # [N,M]

    return iou - (area - union) / area

def temporal_iou_numpy(proposal_min, proposal_max, gt_min, gt_max):
    """Compute IoU score between a groundtruth instance and the proposals.

    Args:
        proposal_min (list[float]): List of temporal anchor min.
        proposal_max (list[float]): List of temporal anchor max.
        gt_min (float): Groundtruth temporal box min.
        gt_max (float): Groundtruth temporal box max.

    Returns:
        list[float]: List of iou scores.
    """
    len_anchors = proposal_max - proposal_min
    int_tmin = np.maximum(proposal_min, gt_min)
    int_tmax = np.minimum(proposal_max, gt_max)
    inter_len = np.maximum(int_tmax - int_tmin, 0.)
    union_len = len_anchors - inter_len + gt_max - gt_min
    jaccard = np.divide(inter_len, union_len)
    return jaccard


def temporal_iou_numpy(proposal_min, proposal_max, gt_min, gt_max):
    """Compute IoP score between a groundtruth bbox and the proposals.

    Compute the IoP which is defined as the overlap ratio with
    groundtruth proportional to the duration of this proposal.

    Args:
        proposal_min (list[float]): List of temporal anchor min.
        proposal_max (list[float]): List of temporal anchor max.
        gt_min (float): Groundtruth temporal box min.
        gt_max (float): Groundtruth temporal box max.

    Returns:
        list[float]: List of intersection over anchor scores.
    """
    len_anchors = np.array(proposal_max - proposal_min)
    int_tmin = np.maximum(proposal_min, gt_min)
    int_tmax = np.minimum(proposal_max, gt_max)
    inter_len = np.maximum(int_tmax - int_tmin, 0.)
    scores = np.divide(inter_len, len_anchors)
    return scores


def soft_nms(proposals, alpha, low_threshold, high_threshold, top_k):
    """Soft NMS for temporal proposals.

    Args:
        proposals (np.ndarray): Proposals generated by network.
        alpha (float): Alpha value of Gaussian decaying function.
        low_threshold (float): Low threshold for soft nms.
        high_threshold (float): High threshold for soft nms.
        top_k (int): Top k values to be considered.

    Returns:
        np.ndarray: The updated proposals.
    """
    proposals = proposals[proposals[:, -1].argsort()[::-1]]
    tstart = list(proposals[:, 0])
    tend = list(proposals[:, 1])
    tscore = list(proposals[:, 2])
    rstart = []
    rend = []
    rscore = []

    while len(tscore) > 0 and len(rscore) <= top_k:
        max_index = np.argmax(tscore)
        max_width = tend[max_index] - tstart[max_index]
        iou_list = temporal_iou_numpy(tstart[max_index], tend[max_index],
                                      np.array(tstart), np.array(tend))
        iou_exp_list = np.exp(-np.square(iou_list) / alpha)

        for idx, _ in enumerate(tscore):
            if idx != max_index:
                current_iou = iou_list[idx]
                if current_iou > low_threshold + (high_threshold -
                                                  low_threshold) * max_width:
                    tscore[idx] = tscore[idx] * iou_exp_list[idx]

        rstart.append(tstart[max_index])
        rend.append(tend[max_index])
        rscore.append(tscore[max_index])
        tstart.pop(max_index)
        tend.pop(max_index)
        tscore.pop(max_index)

    rstart = np.array(rstart).reshape(-1, 1)
    rend = np.array(rend).reshape(-1, 1)
    rscore = np.array(rscore).reshape(-1, 1)
    new_proposals = np.concatenate((rstart, rend, rscore), axis=1)
    return new_proposals


def temporal_nms(segments, thresh):
    """
    One-dimensional non-maximal suppression
    :param segments: [[st, ed, score, ...], ...]
    :param thresh:
    :return:
    """
    t1 = segments[:, 0]
    t2 = segments[:, 1]
    scores = segments[:, 2]

    durations = t2 - t1
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        tt1 = np.maximum(t1[i], t1[order[1:]])
        tt2 = np.minimum(t2[i], t2[order[1:]])
        intersection = tt2 - tt1
        IoU = intersection / \
            (durations[i] + durations[order[1:]] - intersection).astype(float)

        inds = np.where(IoU <= thresh)[0]
        order = order[inds + 1]

    return segments[keep, :]
