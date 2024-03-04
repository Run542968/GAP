# ------------------------------------------------------------------------
# TadTR: End-to-end Temporal Action Detection with Transformer
# Copyright (c) 2021. Xiaolong Liu.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0
# ------------------------------------------------------------------------
# and DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
TadTR model and criterion classes.
"""
import math
import copy

import torch
import torch.nn.functional as F
from torch import nn

from util import segment_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size,
                       is_dist_avail_and_initialized, inverse_sigmoid)
from models.matcher import build_matcher
from models.position_encoding import build_position_encoding
from .custom_loss import sigmoid_focal_loss,softmax_ce_loss
from .transformer import build_deformable_transformer
from .transformer_vanilla import build_alignment_decoder
from opts import cfg
from transformers import CLIPTokenizer, CLIPModel, CLIPTextModel
from zero_shot import split_75_train, split_75_test, split_50_train, split_50_test , train75_dict , test75_dict , train50_dict , test50_dict
import json



def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def get_norm(norm_type, dim, num_groups=None):
    if norm_type == 'gn':
        assert num_groups is not None, 'num_groups must be specified'
        return nn.GroupNorm(num_groups, dim)
    elif norm_type == 'bn':
        return nn.BatchNorm1d(dim)
    else:
        raise NotImplementedError


class TadTR(nn.Module):
    """ This is the TadTR module that performs temporal action detection """

    def __init__(self, position_embedding, transformer, recon_decoder, cls_dim, num_classes, num_queries, feat_dim,
                 aux_loss=True, 
                 with_segment_refine=True, 
                 recon_loss = False,
                 random_mask_raito = 0.3,
                 create_mask_method = 'create_mask_v1', 
                 text_target=False,
                 text_mode='prompt',
                 split=75,
                 description_anno = None,
                 binary = False
                 ):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture. See deformable_transformer.py
            recon_decoder: reconstruct decoder of one layer
            random_mask_raito: the random_mask_ratio of text token dropout
            recon_loss: whether adopting recon_loss
            num_classes: number of action classes
            num_queries: number of action queries, ie detection slot. This is the maximal number of actions
                         TadTR can detect in a single video.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_segment_refine: iterative segment refinement
            text_target: whether choose text as the final target 
            text_mode: what format is choose for text target ['prompt','description']
            split: the split of categories in zero-shot
            description_anno: the description for each class name
            binary: whether to generate class-agnostic proposal
            learnable_emb: whether to add learnable embedding
            feat_dim: the dimension of original feature
            cls_dim: the dimension of feature for alignment with text feature (CLIP: 512)
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        self.recon_decoder = recon_decoder
        hidden_dim = transformer.d_model
        self.feat_dim = feat_dim
        self.cls_dim = cls_dim

        # for zero-shot setting
        self.text_target = text_target
        self.text_mode = text_mode
        self.split = split
        self.description_anno = description_anno
        self.binary = binary

        self.recon_loss = recon_loss
        self.create_mask_method = create_mask_method
        self.random_mask_raito = random_mask_raito


        if not self.text_target or self.binary:
            self.class_embed = nn.Linear(hidden_dim, num_classes) # classfication head
        else:
            self.class_embed = nn.Linear(hidden_dim,cls_dim) # classfication matrix
        self.segment_embed = MLP(hidden_dim, hidden_dim, 2, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim*2)

        self.input_proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(feat_dim, hidden_dim, kernel_size=1),
                nn.GroupNorm(32, hidden_dim),
            )])

        self.position_embedding = position_embedding
        self.aux_loss = aux_loss
        self.with_segment_refine = with_segment_refine
        self.recon_loss = recon_loss

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        if not self.text_target or self.binary:
            self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        else:
            self.class_embed.bias.data = torch.ones(cls_dim) * bias_value
        nn.init.constant_(self.segment_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.segment_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        num_pred = transformer.decoder.num_layers
        if with_segment_refine: # specific parameters for each laryer
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.segment_embed = _get_clones(self.segment_embed, num_pred)
            nn.init.constant_(
                self.segment_embed[0].layers[-1].bias.data[1:], -2.0)
            # hack implementation for segment refinement
            self.transformer.decoder.segment_embed = self.segment_embed
        else: # shared parameters for each laryer
            nn.init.constant_(
                self.segment_embed.layers[-1].bias.data[1:], -2.0)
            self.class_embed = nn.ModuleList(
                [self.class_embed for _ in range(num_pred)])
            self.segment_embed = nn.ModuleList(
                [self.segment_embed for _ in range(num_pred)])
            self.transformer.decoder.segment_embed = None


        if self.text_target and not self.binary:
            self.txt_model = CLIPTextModel.from_pretrained("./clip-vit-base-patch32").float()
            self.tokenizer = CLIPTokenizer.from_pretrained("./clip-vit-base-patch32")


    def get_description(self,cl_names):
        temp_prompt = []
        for c in cl_names:
            temp_prompt.append(self.description_anno[c]['Elaboration']['Description'])
        return temp_prompt


    def text_features(self, cl_names,device,mode):
        def get_prompt(cl_names):
            temp_prompt = []
            for c in cl_names:
                temp_prompt.append("a video of action"+" "+c)
            return temp_prompt
        
        def get_description(cl_names):
            temp_prompt = []
            for c in cl_names:
                temp_prompt.append(self.description_anno[c]['Elaboration']['Description'])
            return temp_prompt

        if self.text_mode == 'prompt':
            act_prompt = get_prompt(cl_names)
        elif self.text_mode == 'description':
            act_prompt = get_description(cl_names)
        else: 
            raise ValueError("Don't define this text_mode.")
        
        texts = self.tokenizer(act_prompt, padding=True, return_tensors="pt").to(device) #{input_ids,attention_mask}->input_ids:[150,length],attention_mak:[150,length]
        attention_mask = texts['attention_mask'] # NOTE: notice that this mask is opposite with Transformer, here 0 represents padding value, but nn.MultiHeadAttention inverse
        mask = attention_mask.bool()
        mask = ~mask

        output = self.txt_model(**texts) # output.last_hidden_state:[150,length,dim], output.pooler_output:[150,dim]
        sentence_emb, element_emb = output.pooler_output, output.last_hidden_state


        return sentence_emb,element_emb,mask

    def create_mask(sefl,instance_mask,memory_mask,text_element_mask,random_mask_ratio=None):
        '''
            **reconstruct action instance feature and reconstruct text token, without any dropout**
        
            instance_mask: [slice_len], bool, the mask to confirm instane area, True represents this snippet is background and need to be masked
            memory_mask: [slice_len], bool, the mask for padding (padding is used only without set slice_len for uniform sampling), True represents this snippet is padding value that need to be masked
            text_element_mask: [text_length], bool, the mask for token padding, True represents this snippet is padding value that need to be masked

            return:
                padding_mask: the padding mask in MultiHead Attention
                recon_mask: the mask when compute the MSE loss
        '''

        action_mask = instance_mask | memory_mask
        padding_mask = torch.cat([action_mask,text_element_mask],dim=0) # [slice_len+length] without drop
        
        # construct the recon_loss_mask
        recon_loss_mask = torch.cat([action_mask,text_element_mask],dim=0) 

        return padding_mask, recon_loss_mask
    
    def create_mask_v1(sefl,instance_mask,memory_mask,text_element_mask,random_mask_ratio=None):
        '''
            **don't need to reconstruct the action instance feature, just reconstruct text token, dropout [input text token] based on drop_ratio**
        
            instance_mask: [slice_len], bool, the mask to confirm instane area, True represents this snippet is background and need to be masked
            memory_mask: [slice_len], bool, the mask for padding (padding is used only without set slice_len for uniform sampling), True represents this snippet is padding value that need to be masked
            text_element_mask: [text_length], bool, the mask for token padding, True represents this snippet is padding value that need to be masked

            return:
                padding_mask: the padding mask in MultiHead Attention
                recon_mask: the mask when compute the MSE loss
        '''

        action_mask = instance_mask | memory_mask
        # the padding mask in self-attention
        if random_mask_ratio is not None:
            random_mask = torch.rand(text_element_mask.shape,device=text_element_mask.device)
            random_mask = torch.where(random_mask<=random_mask_ratio,True,False)
            random_text_mask = text_element_mask | random_mask
        padding_mask = torch.cat([action_mask,random_text_mask],dim=0) 

        
        # construct the recon_loss_mask
        recon_action_mask = torch.ones_like(action_mask,dtype=bool,device=action_mask.device) 
        recon_loss_mask = torch.cat([recon_action_mask,text_element_mask],dim=0) # don't need to reconstruct the action instance feature, just reconstruct text token

        return padding_mask, recon_loss_mask

    def forward(self, samples, instance_masks, classes_name, device,mode="train"):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensors: batched images, of shape [batch_size x C x T]
               - samples.mask: a binary mask of shape [batch_size x T], containing 1 on padded pixels
            or a tuple of tensors and mask

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-action) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_segments": The normalized segments coordinates for all queries, represented as
                               (center, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized segment.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
            instance_masks: a list of mask dict, {'label_name':{'mask': T, 'label_id': 1}}
        """
        if not isinstance(samples, NestedTensor):
            if isinstance(samples, (list, tuple)):
                samples = NestedTensor(*samples)
            else:
                samples = nested_tensor_from_tensor_list(samples)  # (n, c, t)
        # print(f"samples.tensors.shape:{samples.tensors.shape}") # [b,dim,t]

        if self.text_target and not self.binary:
            with torch.no_grad():
                sentence_emb,element_emb,attention_mask = self.text_features(classes_name,device,mode) # sentence_emb: [num_classes, 512]; element_emb: [num_classes, length, 512]; attention_mask: [num_classes, length]
        

        pos = [self.position_embedding(samples)]
        src, mask = samples.tensors, samples.mask 
        srcs = [self.input_proj[0](src)] # low-level Conv1d
        masks = [mask]

        query_embeds = self.query_embed.weight
        # print(f"query_embeds.shape:{query_embeds.shape}") # [b,nq,dim]
        hs, init_reference, inter_references, memory = self.transformer(
            srcs, masks, pos, query_embeds)
        # print(f"len(hs):{len(hs)}") # [4]
        # print(f"hs[-1].shape:{hs[-1].shape}") # [b,nq,dim]
        # print(f"init_reference.shape:{init_reference.shape}") # [b,nq,1]
        # print(f"inter_references.len:{len(inter_references)},inter_references[-1].shape:{inter_references[-1].shape}") # # [4] [b,nq,2]
        # print(f"memory.shape:{memory.shape}") # [b,hidden_dim,t]

        if self.training and self.recon_loss:
            # process mask, easily connect, then 1 layer decoder
            new_batches = []
            padding_masks = []
            recon_loss_masks = []
            recon_targets = []
            for im, mem, s, pm in zip(instance_masks, memory, src, mask):
                for k,v in im.items():
                    label_id = v['label_id'] # [1]
                    text_element = element_emb[label_id].T # [length,512]->[512,length]
                    # print(f"text_element.shape:{text_element.shape}")
                    text_element_mask = attention_mask[label_id] # [length]
                    # print(f"text_element_mask.shape:{text_element_mask.shape}")
                    # print(f"text_element_mask:{text_element_mask}")

                    instance_mask = v['mask'] # [slice_len]
                    # print(f"instance_mask.shape:{instance_mask.shape}")
                    # print(f"instance_mask:{instance_mask}")
                    padding_mask = pm # [slice_len]
                    # print(f"memory_mask.shape:{memory_mask.shape}")
                    # print(f"memory_mask:{memory_mask}")

                    new_feat = torch.cat([mem,text_element],dim=1) # [512,slice_len+length]
                    # print(f"new_feat.shape:{new_feat.shape}")
                    padding_mask, recon_loss_mask = getattr(self,self.create_mask_method)(instance_mask,padding_mask,text_element_mask,self.random_mask_raito)
                    # print(f"padding_mask.shape:{padding_mask.shape}")
                    # print(f"padding_mask:{padding_mask}")
                    # print(f"recon_loss_mask.shape:{recon_loss_mask.shape}")
                    # print(f"recon_loss_mask:{recon_loss_mask}")
                    recon_target = torch.cat([s,text_element],dim=1) # [512,slice_len+length]


                    new_batches.append(new_feat)
                    padding_masks.append(padding_mask)
                    recon_loss_masks.append(recon_loss_mask)
                    recon_targets.append(recon_target)
            new_batches = torch.stack(new_batches,dim=0) # [new_b,512,slice_len+length]
            padding_masks = torch.stack(padding_masks,dim=0) # [new_b, slice_len+length]
            recon_loss_masks = torch.stack(recon_loss_masks,dim=0) # [new_b, slice_len+length]
            recon_targets = torch.stack(recon_targets,dim=0) # [new_b,512,slice_len+length]
            # print(f"new_batches.shape:{new_batches.shape}") 
            # print(f"new_masks.shape:{new_masks.shape}") 

            recon_feature = self.recon_decoder(new_batches.permute(0,2,1), src_key_padding_mask=padding_masks, pos=None) # [new_b,slice_len+length,512]

            recon_out = {
                'recon_feature':recon_feature,
                'padding_masks':padding_masks,
                'recon_targets':recon_targets.permute(0,2,1), # [new_b,slice_len+length,512]
                'recon_loss_masks':recon_loss_masks
            }
        else:
            recon_out = None

        outputs_classes = []
        outputs_coords = []
        # gather outputs from each decoder layer
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]

            reference = inverse_sigmoid(reference)
            if not self.text_target or self.binary:
                outputs_class = self.class_embed[lvl](hs[lvl]) # [b,nq,c]
            else:
                outputs_class = torch.einsum("bqd,cd->bqc",self.class_embed[lvl](hs[lvl]),sentence_emb)
            tmp = self.segment_embed[lvl](hs[lvl]) # [b,nq,2]
            # the l-th layer (l >= 2)
            if reference.shape[-1] == 2:
                tmp += reference
            # the first layer
            else:
                assert reference.shape[-1] == 1
                tmp[..., 0] += reference[..., 0]
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)
        # print(f"outputs_class.shape{outputs_class.shape}, outputs_coord.shape:{outputs_coord.shape}") # [num_layers,b,nq,num_classes] [num_layers,b,nq,2]

        out = {'pred_logits': outputs_class[-1],
                'pred_segments': outputs_coord[-1]}

        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(
                outputs_class, outputs_coord)


        if self.training:
            return out, recon_out
        else:
            return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_segments': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class SetCriterion(nn.Module):
    """ This class computes the loss for TadTR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth segments and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and segment)
    """

    def __init__(self, num_classes, matcher, weight_dict, losses, focal_alpha=0.25, ce_loss=False):
        """ Create the criterion.
        Parameters:
            num_classes: number of action categories, omitting the special no-action category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
            ce_loss: Whether to adopt CrossEntropy loss instead of BCE, so sigmoid is substituted by softmax
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.ce_loss = ce_loss

    def loss_labels(self, outputs, targets, indices, num_segments, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_segments]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits'] # [b,nq,c]

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)]) # [tgt_num]
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o # [b,nq]

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device) # [b,nq,c+1]
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:,:,:-1] # [b,nq,c]

        if self.ce_loss:
            loss_ce = softmax_ce_loss(src_logits, target_classes_onehot, num_segments)
        else:
            loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_segments, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]  # nq
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]

        return losses

    def loss_segments(self, outputs, targets, indices, num_segments):
        """Compute the losses related to the segmentes, the L1 regression loss and the IoU loss
           targets dicts must contain the key "segments" containing a tensor of dim [nb_target_segments, 2]
           The target segments are expected in format (center, width), normalized by the video length.
        """
        assert 'pred_segments' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_segments = outputs['pred_segments'][idx] # [tgt_num,2]
        target_segments = torch.cat([t['segments'][i] for t, (_, i) in zip(targets, indices)], dim=0) # [tgt_num,2]

        loss_segment = F.l1_loss(src_segments, target_segments, reduction='none')

        losses = {}
        losses['loss_segments'] = loss_segment.sum() / num_segments


        loss_iou = 1 - torch.diag(segment_ops.segment_iou(
            segment_ops.segment_cw_to_t1t2(src_segments),
            segment_ops.segment_cw_to_t1t2(target_segments))) # torch.diag(), get the main diag elements

        losses['loss_iou'] = loss_iou.sum() / num_segments
        return losses

    # def loss_actionness(self, outputs, targets, indices, num_segments):
    #     """Compute the actionness regression loss
    #        targets dicts must contain the key "segments" containing a tensor of dim [nb_target_segments, 2]
    #        The target segments are expected in format (center, width), normalized by the video length.
    #     """
    #     assert 'pred_segments' in outputs
    #     assert 'pred_actionness' in outputs
    #     src_segments = outputs['pred_segments'].view((-1, 2)) # [b*nq,2]
    #     target_segments = torch.cat([t['segments'] for t in targets], dim=0) # [b*num_tgt,2]

    #     losses = {}
    #     iou_mat = segment_ops.segment_iou(
    #         segment_ops.segment_cw_to_t1t2(src_segments),
    #         segment_ops.segment_cw_to_t1t2(target_segments)) # [src,target]

    #     gt_iou = iou_mat.max(dim=1)[0]
    #     pred_actionness = outputs['pred_actionness']
    #     loss_actionness = F.l1_loss(pred_actionness.view(-1), gt_iou.view(-1).detach())   

    #     losses['loss_actionness'] = loss_actionness
    #     return losses

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

    def get_loss(self, loss, outputs, targets, indices, num_segments, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'segments': self.loss_segments,
            # 'actionness': self.loss_actionness,
        }

        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_segments, **kwargs)

    def forward(self, outputs, targets, recon_outputs=None):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target segments accross all nodes, for normalization purposes
        num_segments = sum(len(t["labels"]) for t in targets)
        num_segments = torch.as_tensor([num_segments], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_segments)
        num_segments = torch.clamp(num_segments / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            kwargs = {}
            losses.update(self.get_loss(loss, outputs, targets, indices, num_segments, **kwargs))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    # we do not compute actionness loss for aux outputs
                    if 'actionness' in loss:
                        continue
         
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs['log'] = False
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_segments, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        self.indices = indices

        # intermediate reconstruct loss
        if recon_outputs is not None:
            recon_feature, recon_loss_masks, recon_targets = recon_outputs['recon_feature'], recon_outputs['recon_loss_masks'], recon_outputs['recon_targets'] # [new_b,slice_len+length,512]
            rec_loss = (recon_feature-recon_targets).pow(2).sum(-1) # [new_b, slice_len+length]
            rec_loss = torch.mul(rec_loss,recon_loss_masks).sum() # [new_b,slice_len+length]->[1]
            N = recon_loss_masks.sum()
            rec_loss = rec_loss/N
            rec_loss_dict = {'loss_recon':rec_loss}
            losses.update(rec_loss_dict)


        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the TADEvaluator"""

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size] containing the duration of each video of the batch
        """
        out_logits, out_segments = outputs['pred_logits'], outputs['pred_segments']

        assert len(out_logits) == len(target_sizes)
        # assert target_sizes.shape[1] == 1

        prob = out_logits.sigmoid()   # [bs, nq, C]

        segments = segment_ops.segment_cw_to_t1t2(out_segments)   # bs, nq, 2

        if cfg.postproc_rank == 1:     # default
            # sort across different instances, pick top 100 at most
            topk_values, topk_indexes = torch.topk(prob.view(
                out_logits.shape[0], -1), min(cfg.postproc_ins_topk, prob.shape[1]*prob.shape[2]), dim=1)
            scores = topk_values
            topk_segments = topk_indexes // out_logits.shape[2] # [bs,nq*c]
            labels = topk_indexes % out_logits.shape[2]

            # bs, nq, 2; bs, num, 2
            segments = torch.gather(
                segments, 1, topk_segments.unsqueeze(-1).repeat(1, 1, 2)) # [bs,topk,2]
            query_ids = topk_segments # [bs,topk]
        else:
            # pick topk classes for each query
            # pdb.set_trace()
            scores, labels = torch.topk(prob, cfg.postproc_cls_topk, dim=-1)
            scores, labels = scores.flatten(1), labels.flatten(1)
            # (bs, nq, 1, 2)
            segments = segments[:, [
                i//cfg.postproc_cls_topk for i in range(cfg.postproc_cls_topk*segments.shape[1])], :]
            query_ids = (torch.arange(0, cfg.postproc_cls_topk*segments.shape[1], 1, dtype=labels.dtype,
                         device=labels.device) // cfg.postproc_cls_topk)[None, :].repeat(labels.shape[0], 1)

        # from normalized [0, 1] to absolute [0, length] (second) coordinates
        vid_length = target_sizes
        scale_fct = torch.stack([vid_length, vid_length], dim=1) # [bs,2]
        segments = segments * scale_fct[:, None, :] # [bs,topk,2] transform fraction to second

        results = [{'scores': s, 'labels': l, 'segments': b, 'query_ids': q}
                   for s, l, b, q in zip(scores, labels, segments, query_ids)] 

        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k)
                                    for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    if args.task_setting == 'close_set':
        if args.binary:
            num_classes = 1
        else:
            if args.dataset_name == 'thumos14':
                num_classes = 20
            elif args.dataset_name == 'activitynet1.3':
                num_classes = 200
            else:
                raise ValueError('unknown dataset {} in close_set setting'.format(args.dataset_name))
    elif args.task_setting == 'zero_shot':
        if args.binary:
            num_classes = 1
        else: # must set it for decide the idx of background class in SetCriterion
            if args.split == 75: 
                num_classes = len(split_75_train)
            elif args.split == 50:
                num_classes = len(split_50_train)
            else:
                raise ValueError("Don't define this split mode in zero-shot setting.")
    else:
        raise ValueError("Don't have this task setting.")

    pos_embed = build_position_encoding(args)
    transformer = build_deformable_transformer(args)
    recon_decoder = build_alignment_decoder(args)
    
    # description anno
    if args.dataset_name == 'thumos14':
        description_path = "./data/STALE_anno/Thumos14/Thumos14_des.json"
    elif args.dataset_name == 'activitynet1.3':
        description_path = "./data/STALE_anno/ActivityNet1.3/ActivityNet1.3_des.json"
    else:
        raise ValueError('unknown dataset {}'.format(args.dataset_name))
    with open(description_path) as json_file:
        description_anno = json.load(json_file)

    model = TadTR(
        pos_embed,
        transformer,
        recon_decoder,
        cls_dim=args.cls_dim,
        num_classes=num_classes,
        num_queries=args.num_queries,
        feat_dim=args.feature_dim,
        aux_loss=args.aux_loss,
        with_segment_refine=args.seg_refine,
        recon_loss=args.recon_loss,
        random_mask_raito=args.random_mask_raito,
        create_mask_method=args.create_mask_method,
        text_target=args.text_target,
        text_mode=args.text_mode,
        split=args.split,
        description_anno = description_anno,
        binary = args.binary
    )

    matcher = build_matcher(args)
    losses = ['labels', 'segments']

    weight_dict = {
        'loss_ce': args.cls_loss_coef, 
        'loss_segments': args.seg_loss_coef,
        'loss_iou': args.iou_loss_coef,
        'loss_recon':args.rec_loss_coef,
        # 'class_error': args.cls_error_coef
        }

    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items() if k != 'loss_recon'})
        aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items() if k != 'loss_recon'})
        weight_dict.update(aux_weight_dict)

    criterion = SetCriterion(num_classes, matcher,
        weight_dict, losses, focal_alpha=args.focal_alpha, ce_loss=args.ce_loss)

    postprocessor = PostProcess()

    return model, criterion, postprocessor
