# Mostly copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

'''Focal loss implementation'''


import torch
import torch.nn.functional as F


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes


def softmax_ce_loss(inputs, targets, num_boxes):
    """
    Naive CrossEntropy loss
    Args:
        inputs: A float tensor of [b,nq,c]
        targets: A float tensor with the same shape as inputs
    Returns:
        Loss tensor
    """
    targets = targets / (
            torch.sum(targets, dim=-1, keepdim=True) + 1e-4) # [b,nq,c]
    
    loss_pos = -(targets * F.log_softmax(inputs, dim=-1)).sum(dim=-1) # [b,nq]
    
    pro = targets.softmax(dim=-1)
    loss_neg = -((1-targets)*torch.log(1 - pro+1e-8)).sum(dim=-1) # [b,nq]

    loss = loss_pos+loss_neg
    return loss.mean(1).sum() / num_boxes


if __name__ == "__main__":
    import numpy as np
    pred = torch.from_numpy(np.random.random([8, 2]))
    target = torch.from_numpy(np.random.random(8) > 0.5).long()
    loss = sigmoid_focal_loss(pred, target)
    

