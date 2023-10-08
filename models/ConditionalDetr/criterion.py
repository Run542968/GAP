from utils.misc import (accuracy)
from utils.segment_ops import segment_cw_to_t1t2,segment_iou
import torch.nn as nn
import torch
import torch.nn.functional as F

def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example. [bs,num_queries,num_classes]
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs. [bs,num_queries,num_classes]
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid() # [bs,num_queries,num_classes]
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets) # [bs,num_queries,num_classes]
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes 

class SetCriterion(nn.Module):
    """ This class computes the loss for Conditional DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict,focal_alpha, args, base_losses=['labels', 'boxes']):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            base_losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
            gamma: gamma in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.base_losses = ['labels', 'boxes']
        self.focal_alpha = focal_alpha
        self.gamma = args.gamma
        self.instance_loss = args.instance_loss
        self.instance_loss_type = args.instance_loss_type
        self.matching_loss = args.matching_loss
        self.mask_loss = args.mask_loss
        self.segmentation_loss = args.segmentation_loss
        self.instance_loss_v2 = args.instance_loss_v2
        self.instance_loss_v3 = args.instance_loss_v3
        self.distillation_loss = args.distillation_loss
        

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits'] # [bs,num_queries,num_classes]

        idx = self._get_src_permutation_idx(indices) # (batch_idx,src_idx)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)]) # [batch_target_class_id]
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device) # [bs,num_queries]
        target_classes[idx] = target_classes_o # [bs,num_queries]

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2]+1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device) # [bs,num_queries,num_classes+1]
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:,:,:-1] # [bs,num_queries,num_classes]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=self.gamma) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0] # [batch_matched_queries,num_classes]
        return losses

 
    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "segments" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices) # (batch_idx,src_idx)
        src_boxes = outputs['pred_boxes'][idx] # [batch_matched_queries,2]
        target_boxes = torch.cat([t['segments'][i] for t, (_, i) in zip(targets, indices)], dim=0) # [batch_target,2]

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(segment_iou(
            segment_cw_to_t1t2(src_boxes).clamp(min=0,max=1), 
            segment_cw_to_t1t2(target_boxes))) # the clamp is to deal with the case "center"-"width/2" < 0 and "center"+"width/2" < 1
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_segmentations(self,outputs, targets, indices, num_boxes):
        '''
        for dense prediction, which is like to maskformer, the loss compute on the memory of the output of encoder
        segmentation_logits: the output of segmentation_logits => [B,T,num_classes]
        feature_list: the feature list after backbone for temporal modeling => list[NestedTensor]
        '''
        # assert 'segmentation_logits' in outputs
        # assert 'segmentation_onehot_labels' in targets[0]

        # # obtain logits
        # segmentation_logits = outputs['segmentation_logits'] # [B,T,num_classes]
        # B,T,C = segmentation_logits.shape

        # # prepare labels
        # segmentation_onehot_labels_pad = torch.zeros_like(segmentation_logits) # [B,T,num_classes], zero in padding region
        # for i, tgt in enumerate(targets):
        #     feat_length = tgt['segmentation_onehot_labels'].shape[0]
        #     segmentation_onehot_labels_pad[i,:feat_length,:] = tgt['segmentation_onehot_labels'] # [feat_length, num_classes]

        # ce_loss = -(segmentation_onehot_labels_pad * F.log_softmax(segmentation_logits, dim=-1)).sum(dim=-1)

        # losses = {}
        # losses['loss_segmentation'] = ce_loss.sum(-1).sum() / B

        assert 'segmentation_logits' in outputs
        assert 'segmentation_labels' in targets[0]
        assert 'logits_mask' in outputs

        # obtain logits
        segmentation_logits = outputs['segmentation_logits'] # [B,T,num_classes+1]
        B,T,C = segmentation_logits.shape

        # prepare labels
        segmentation_labels_pad = torch.full(segmentation_logits.shape[:2], segmentation_logits.shape[2]-1, dtype=torch.int64, device=segmentation_logits.device) # [B,T]
        for i, tgt in enumerate(targets):
            feat_length = tgt['segmentation_labels'].shape[0]
            segmentation_labels_pad[i,:feat_length] = tgt['segmentation_labels'] # [feat_length]

        target_classes_onehot = torch.zeros_like(segmentation_logits) # [bs,T,num_classes+1]
        target_classes_onehot.scatter_(2, segmentation_labels_pad.unsqueeze(-1), 1)
        

        prob = segmentation_logits.sigmoid() # [batch_instance_num,num_classes]
        ce_loss = F.binary_cross_entropy_with_logits(segmentation_logits, target_classes_onehot, reduction="none")
        p_t = prob * target_classes_onehot + (1 - prob) * (1 - target_classes_onehot) # [bs,T,num_classes+1]
        loss = ce_loss * ((1 - p_t) ** 2)
        alpha = 0.25
        if alpha >= 0:
            alpha_t = alpha * target_classes_onehot + (1 - alpha) * (1 - target_classes_onehot)
            loss = alpha_t * loss

        logits_mask = ~outputs['logits_mask'] # covert False to 1
        loss = torch.einsum("btc,bt->btc",loss,logits_mask) # [b,t,c+1]

        losses = {}
        losses['loss_segmentation'] = loss.sum(-1).mean()
        return losses

    def loss_instances(self,outputs, targets, indices, num_boxes):
        '''
        for instance prediction, modeling the relation of instance region and text 
        instance_logits: the output of instance_logits => [batch_instance_num,num_classes]
        '''
        assert 'instance_logits' in outputs
 
        # obtain logits
        instance_logits = outputs['instance_logits'] #[batch_instance_num,num_classes]
        B,C = instance_logits.shape

        instance_gt = [] 
        for t in targets:
            gt_labels = t['labels'] # [num_instance]
            instance_gt.append(gt_labels)
        instance_gt = torch.cat(instance_gt,dim=0) # [batch_instance_num]->"class id"
        
        # prepare labels
        target_classes_onehot = torch.zeros_like(instance_logits) # [batch_instance_num,num_classes]
        target_classes_onehot.scatter_(1, instance_gt.reshape(-1,1), 1) # [batch_instance_num,num_classes]

        if self.instance_loss_type == "CE":
            loss = -(target_classes_onehot * F.log_softmax(instance_logits, dim=-1)).sum(dim=-1)
        elif self.instance_loss_type == "BCE":
            prob = instance_logits.sigmoid() # [batch_instance_num,num_classes]
            ce_loss = F.binary_cross_entropy_with_logits(instance_logits, target_classes_onehot, reduction="none")
            p_t = prob * target_classes_onehot + (1 - prob) * (1 - target_classes_onehot) # [bs,num_queries,num_classes]
            loss = ce_loss * ((1 - p_t) ** 2)
            alpha = 0.25
            if alpha >= 0:
                alpha_t = alpha * target_classes_onehot + (1 - alpha) * (1 - target_classes_onehot)
                loss = alpha_t * loss
        else:
            raise ValueError
        losses = {}
        losses['loss_instance'] = loss.mean()
        return losses

    def loss_matching(self,outputs, targets, indices, num_boxes):
        '''
        for instance prediction, modeling the relation of instance region and text 
        matching_logits: the output of matching_logits => [batch_instance_num,num_classes]
        '''
        assert 'matching_logits' in outputs
 
        # obtain logits
        matching_logits = outputs['matching_logits'] #[batch_instance_num,num_classes]
        B,C = matching_logits.shape

        instance_gt = [] 
        for t in targets:
            gt_labels = t['labels'] # [num_instance]
            instance_gt.append(gt_labels)
        instance_gt = torch.cat(instance_gt,dim=0) # [batch_instance_num]->"class id"
        
        # prepare labels
        target_classes_onehot = torch.zeros_like(matching_logits) # [batch_instance_num,num_classes]
        target_classes_onehot.scatter_(1, instance_gt.reshape(-1,1), 1) # [batch_instance_num,num_classes]

        prob = matching_logits.sigmoid() # [batch_instance_num,num_classes]
        ce_loss = F.binary_cross_entropy_with_logits(matching_logits, target_classes_onehot, reduction="none")
        p_t = prob * target_classes_onehot + (1 - prob) * (1 - target_classes_onehot) # [bs,num_queries,num_classes]
        loss = ce_loss * ((1 - p_t) ** 2)
        alpha = 0.25
        if alpha >= 0:
            alpha_t = alpha * target_classes_onehot + (1 - alpha) * (1 - target_classes_onehot)
            loss = alpha_t * loss

        losses = {}
        losses['loss_matching'] = loss.mean()
        return losses

    def loss_mask(self,outputs, targets, indices, num_boxes):
        '''
        for dense prediction, which is like to maskformer, the loss compute on the memory of the output of encoder
        mask_logits: the output of mask_logits => [B,T,num_classes]
        feature_list: the feature list after backbone for temporal modeling => list[NestedTensor]
        '''
        assert 'mask_logits' in outputs
        assert 'mask_labels' in targets[0]

        # obtain logits
        mask_logits = outputs['mask_logits'].squeeze(2) # [B,T,1]->[B,T]
        B,T= mask_logits.shape

        # prepare labels
        mask_labels_pad = torch.zeros_like(mask_logits) # [B,T], zero in padding region
        for i, tgt in enumerate(targets):
            feat_length = tgt['mask_labels'].shape[0]
            mask_labels_pad[i,:feat_length] = tgt['mask_labels'] # [feat_length]

        ce_loss = F.binary_cross_entropy_with_logits(mask_logits, mask_labels_pad, reduction="none").sum(-1) # [B,T]

        losses = {}
        losses['loss_mask'] = ce_loss.sum() / B
        return losses
    
    def loss_distillation(self,outputs, targets, indices, num_boxes):
        '''
        for distillation the CLIP feat to DETR detector
        '''
        assert 'student_logits' in outputs
        assert 'teacher_logits' in outputs

        # obtain logits
        student_logits = outputs['student_logits']
        teacher_logits = outputs['teacher_logits'] # [B,Q,dim]

        loss = torch.abs(teacher_logits-student_logits).sum(-1)

        losses = {}
        losses['loss_distillation'] = loss.mean()
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.base_losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.base_losses:
                    if loss == 'masks':
                        # masks loss don't have intermediate feature of decoder, ignore it
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if self.segmentation_loss: # adopt segmentation_loss
            segmentation_loss = self.loss_segmentations(outputs, targets, indices, num_boxes)
            losses.update(segmentation_loss)

        if self.instance_loss or self.instance_loss_v2 or self.instance_loss_v3: # adopt instance_loss
            instance_loss = self.loss_instances(outputs, targets, indices, num_boxes)
            losses.update(instance_loss)
        
        if self.matching_loss: # adopt matching_loss
            matching_loss = self.loss_matching(outputs, targets, indices, num_boxes)
            losses.update(matching_loss)
        
        if self.mask_loss:
            mask_loss = self.loss_mask(outputs, targets, indices, num_boxes)
            losses.update(mask_loss)

        if self.distillation_loss:
            distillation_loss = self.loss_distillation(outputs, targets, indices, num_boxes)
            losses.update(distillation_loss)
        return losses

def build_criterion(args,num_classes,matcher,weight_dict):
    criterion = SetCriterion(num_classes, 
                             matcher=matcher, 
                             weight_dict=weight_dict,
                             focal_alpha=args.focal_alpha,
                             args=args)
    return criterion