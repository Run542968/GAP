# ------------------------------------------------------------------------
# Modules to compute the matching cost and solve the corresponding LSAP.
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------

import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from utils.segment_ops import segment_cw_to_t1t2,segment_iou


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1, args=None):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        # self.actionness_loss = args.actionness_loss
        # self.eval_proposal = args.eval_proposal
        # self.enable_classAgnostic = args.enable_classAgnostic
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, logits, pred_boxes, tgt_ids, tgt_bbox, sizes):
        """ Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 2] with the predicted box coordinates
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the binary class labels
                 "semantic_lables": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "segments": Tensor of dim [num_target_boxes, 2] containing the target box coordinates
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """

        bs, num_queries = logits.shape[:2] 
        out_prob = logits.flatten(0, 1).sigmoid()  # [batch_size * num_queries, 1]
        out_bbox = pred_boxes.flatten(0, 1)  # [batch_size * num_queries, 2]
        tgt_ids = tgt_ids
        tgt_bbox = tgt_bbox

 
        # Compute the classification cost.
        alpha = 0.25
        gamma = 2.0
        neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())

        cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids] # [b*num_queries, gt_instance_num]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1) # [b*num_queries, gt_instance_num]

        # Compute the giou cost betwen boxes
        # NOTE that: for temporal data, then generalized_seg_iou is equal to seg_iou, since the area==union
        cost_giou = -segment_iou(
                    segment_cw_to_t1t2(out_bbox).clamp(min=0,max=1), 
                    segment_cw_to_t1t2(tgt_bbox).clamp(min=0,max=1)
                    ) # [b*num_queries, gt_instance_num], the clamp is to deal with the case "center"-"width/2" < 0 and "center"+"width/2" < 1

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu() # [b, num_queries, gt_instance_num]


        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(args):
    return HungarianMatcher(cost_class=args.set_cost_class, 
                            cost_bbox=args.set_cost_bbox, 
                            cost_giou=args.set_cost_giou,
                            args = args
                            )