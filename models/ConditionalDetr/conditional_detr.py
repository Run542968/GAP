# ------------------------------------------------------------------------
# Conditional DETR model and criterion classes.
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------

import math
import torch
import torch.nn.functional as F
from torch import nn
import logging

from utils.misc import (NestedTensor, nested_tensor_from_tensor_list, inverse_sigmoid)

from .backbone import build_backbone
from .matcher import build_matcher
from .transformer import build_transformer
from .criterion import build_criterion
from .postprocess import build_postprocess
from models.clip import build_text_encoder
from .semantic_head import build_semantic_head
from models.clip import clip as clip_pkg
import torchvision.ops.roi_align as ROIalign
import numpy as np

logger = logging.getLogger()


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class ConditionalDETR(nn.Module):
    """ This is the Conditional DETR module that performs object detection """
    def __init__(self, 
                 backbone, 
                 transformer, 
                 text_encoder, 
                 logit_scale, 
                 device, 
                 num_classes, 
                 visual_semantic_head,
                 text_semantic_head,
                 args):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            text_encoder: text_encoder from CLIP model. See clip.__init__.py
            logit_scale: the logit_scale of CLIP. See clip.__init__.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         Conditional DETR can detect in a single image. For COCO, we recommend 100 queries.
            target_type: use one-hot or text as target. [none,prompt,description]
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            norm_embed: Just for the similarity compute of CLIP visual and text embedding, following the CLIP setting
        """
        super().__init__()
        self.backbone = backbone
        self.transformer = transformer
        self.text_encoder = text_encoder
        self.logit_scale = logit_scale
        self.device = device
        self.num_classes = num_classes
        self.args = args

        self.num_queries = args.num_queries
        self.target_type = args.target_type
        self.aux_loss = args.aux_loss
        self.norm_embed = args.norm_embed
        self.ROIalign_strategy = args.ROIalign_strategy
        self.ROIalign_size = args.ROIalign_size
        self.instance_loss = args.instance_loss
        self.instance_head_type = args.instance_head_type
        self.exp_logit_scale = args.exp_logit_scale
        self.subaction_version = args.subaction_version
        self.instance_loss_ensemble = args.instance_loss_ensemble
        self.ensemble_rate = args.ensemble_rate
        self.ensemble_strategy = args.ensemble_strategy
        self.matching_loss = args.matching_loss
        self.matching_loss_type = args.matching_loss_type
        self.mask_loss = args.mask_loss

        hidden_dim = transformer.d_model


        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 2, 3)
        self.query_embed = nn.Embedding(self.num_queries, hidden_dim)
        self.input_proj = nn.Conv1d(backbone.feat_dim, hidden_dim, kernel_size=1)
        

        if self.instance_loss: # instance head
            self.instance_visual_head = visual_semantic_head
            self.instance_text_head = text_semantic_head

        if self.matching_loss and self.matching_loss_type == "learnable":
            # self.matching_head = MLP(2*hidden_dim,2*hidden_dim,1,3)
            self.instance_text_head = nn.MultiheadAttention(hidden_dim, 4, dropout=0.1, batch_first=True)

        if self.mask_loss:
            self.mask_head = nn.Linear(hidden_dim, 1)

        # init prior_prob setting for focal loss
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value

        # init bbox_mebed
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)


        if self.target_type != "none":
            logger.info(f"The target_type is {self.target_type}, using text embedding as target, on task: {args.task}!")
        else:
            logger.info(f"The target_type is {self.target_type}, using one-hot coding as target, must in close_set!")

    def get_text_feats(self, cl_names, description_dict, device, target_type):
        def get_prompt(cl_names):
            temp_prompt = []
            for c in cl_names:
                temp_prompt.append("a video of a person doing"+" "+c)
            return temp_prompt
        
        def get_description(cl_names):
            temp_prompt = []
            for c in cl_names:
                temp_prompt.append(description_dict[c]['Elaboration']['Description'][0]) # NOTE: default the idx of description is 0.
            return temp_prompt
        
        if target_type == 'prompt':
            act_prompt = get_prompt(cl_names)
        elif target_type == 'description':
            act_prompt = get_description(cl_names)
        elif target_type == 'name':
            act_prompt = cl_names
        else: 
            raise ValueError("Don't define this text_mode.")
        
        tokens = clip_pkg.tokenize(act_prompt).long().to(device) # input_ids->input_ids:[150,length]
        text_feats = self.text_encoder(tokens).float()

        return text_feats
    

    def get_subaction_feats(self,cl_names, description_dict, device, version):
        '''
        v1: only adopt the sub-action 
        v2: class_name + sub-action 
        v3: class_name prompt + sub-action 
        '''
        text_feats = []
        for c in cl_names:
            if version == "v1":
                subaction_list = []
            elif version == "v2":
                subaction_list = [c]
            elif version == "v3":
                subaction_list = ["a video of a person doing"+" "+c]
            else: 
                raise ValueError(f"Don't have this version: {version}")
            subaction_list = subaction_list + description_dict[c]['Elaboration']['Description'] # [sub-action num, length]
            subaction_tokens = clip_pkg.tokenize(subaction_list).long().to(device)
            subaction_feats = self.text_encoder(subaction_tokens).float() # [sub-action num, dim]
            text_feats.append(subaction_feats)
        nested_text_feats = nested_tensor_from_tensor_list(text_feats) # tensors: [num_classes,unify length,dim] mask: [num_classes,unify length]

        return nested_text_feats 


    def _to_roi_align_format(self, rois, truely_length, scale_factor=1):
        '''Convert RoIs to RoIAlign format.
        Params:
            RoIs: normalized segments coordinates, shape (batch_size, num_segments, 2)
            T: length of the video feature sequence
        '''
        # transform to absolute axis
        B, N = rois.shape[:2]
        rois_center = rois[:, :, 0:1] # [B,N,1]
        rois_size = rois[:, :, 1:2] * scale_factor # [B,N,1]
        truely_length = truely_length.reshape(-1,1,1) # [B,1,1]
        rois_abs = torch.cat(
            (rois_center - rois_size/2, rois_center + rois_size/2), dim=2) * truely_length # [B,N,2]->"start,end"
        # expand the RoIs
        _max = truely_length.repeat(1,N,2)
        _min = torch.zeros_like(_max)
        rois_abs = torch.clamp(rois_abs, min=_min, max=_max)  # (B, N, 2)
        # transfer to 4 dimension coordination
        rois_abs_4d = torch.zeros((B,N,4),dtype=rois_abs.dtype,device=rois_abs.device)
        rois_abs_4d[:,:,0], rois_abs_4d[:,:,2] = rois_abs[:,:,0], rois_abs[:,:,1] # x1,0,x2,0

        # add batch index
        batch_ind = torch.arange(0, B).view((B, 1, 1)).to(rois_abs.device) # [B,1,1]
        batch_ind = batch_ind.repeat(1, N, 1) # [B,N,1]
        rois_abs_4d = torch.cat((batch_ind, rois_abs_4d), dim=2) # [B,N,1+4]->"batch_id,x1,0,x2,0"
        # NOTE: stop gradient here to stablize training
        return rois_abs_4d.view((B*N, 5)).detach()

    def _roi_align(self, rois, origin_feat, mask, ROIalign_size, scale_factor=1):
        B,Q,_ = rois.shape
        B,T,C = origin_feat.shape
        truely_length = T-torch.sum(mask,dim=1) # [B]
        rois_abs_4d = self._to_roi_align_format(rois,truely_length,scale_factor)
        feat = origin_feat.permute(0,2,1) # [B,dim,T]
        feat = feat.reshape(B,C,1,T)
        roi_feat = ROIalign(feat, rois_abs_4d, output_size=(1,ROIalign_size))
        roi_feat = roi_feat.reshape(B,Q,C,-1) # [B,Q,dim,output_width]
        roi_feat = roi_feat.permute(0,1,3,2) # [B,Q,output_width,dim]
        return roi_feat

    def _get_roi_prediction_v1(self,samples,coord,text_feats,ROIalign_size):
        '''
        _get_roi_prediction_v1: ROIalign before compute similarity, this strategy is easy to avoid the padded zero value

        samples: NestedTensor
        coord: [B,Q,2]
        '''
        origin_feat, mask = samples.decompose() # [B,T,dim] [B,T]
        roi_feat = self._roi_align(rois=coord,origin_feat=origin_feat,mask=mask,ROIalign_size=ROIalign_size).mean(-2) # [B,Q,dim]

        roi_logits = self._compute_similarity(roi_feat,text_feats) # [b,Q,num_classes]
        return roi_logits
    
    def _get_roi_prediction_v2(self,snippet_logits,mask,coord,ROIalign_size):
        '''
        _get_roi_prediction_v2: ROIalign before compute similarity, this strategy is more reasonable to utilize the CLIP semantic

        snippet_logits: [B,T,num_classes]
        coord: [B,Q,2]
        '''

        roi_logits = self._roi_align(rois=coord,origin_feat=snippet_logits,ROIalign_size=ROIalign_size,mask=mask).mean(-2) # [B,Q,num_classes]

        return roi_logits
    
    def _compute_similarity(self, visual_feats, text_feats):
        '''
        text_feats: [num_classes,dim]
        '''
        if len(visual_feats.shape)==2: # batch_num_instance,dim
            if self.norm_embed:
                visual_feats = visual_feats / visual_feats.norm(dim=-1,keepdim=True)
                text_feats = text_feats / text_feats.norm(dim=-1,keepdim=True)
                if self.exp_logit_scale:
                    logit_scale = self.logit_scale.exp()
                else:
                    logit_scale = self.logit_scale
                logits = torch.einsum("bd,cd->bc",logit_scale*visual_feats,text_feats)
            else:
                logits = torch.einsum("bd,cd->bc",visual_feats,text_feats)
            return logits
        elif len(visual_feats.shape)==3:# batch,num_queries/snippet_length,dim
            if self.norm_embed:
                visual_feats = visual_feats / visual_feats.norm(dim=-1,keepdim=True)
                text_feats = text_feats / text_feats.norm(dim=-1,keepdim=True)
                if self.exp_logit_scale:
                    logit_scale = self.logit_scale.exp()
                else:
                    logit_scale = self.logit_scale
                logits = torch.einsum("bqd,cd->bqc",logit_scale*visual_feats,text_feats)
            else:
                logits = torch.einsum("bqd,cd->bqc",visual_feats,text_feats)
            return logits
        else:
            NotImplementedError

    def forward(self, samples: NestedTensor, classes_name, description_dict, targets):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched video, of shape [batch_size x T x C]
               - samples.mask: a binary mask of shape [batch_size x T], containing 1 (i.e., true) on padded snippet
            classes_name: the class name of involved category
            description_dict: the dict of description file
            targets: the targets that contain gt

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x num_classes]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center, width). These values are normalized in [0, 1],
                               relative to the size of each individual video (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)

        # origin CLIP features
        clip_feat, mask = samples.decompose()

        # backbone for temporal modeling
        feature_list, pos = self.backbone(samples) # list of [b,t,c], list of [b,t]

        # prepare text target
        if self.target_type != "none":
            with torch.no_grad():
                if self.instance_loss:
                    text_feats = self.get_subaction_feats(classes_name,description_dict,self.device,self.subaction_version) # nested_text_feats, tensors: [num_classes,unify length,dim] mask: [num_classes,unify length]
                elif self.matching_loss:
                    text_feats = self.get_text_feats(classes_name, description_dict, self.device, self.target_type) # [N classes,dim]
                else:
                    text_feats = self.get_text_feats(classes_name, description_dict, self.device, self.target_type) # [N classes,dim]

        # feed into model
        src, mask = feature_list[-1].decompose()
        assert mask is not None
        src = self.input_proj(src.permute(0,2,1)).permute(0,2,1)
        memory, hs, reference = self.transformer(src, mask, self.query_embed.weight, pos[-1]) # [enc_layers, b,t,c], [dec_layers,b,num_queries,c], [b,num_queries,1]
        
        # record result
        out = {}
        reference_before_sigmoid = inverse_sigmoid(reference) # [b,num_queries,1], Reference point is the predicted center point.
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            tmp = self.bbox_embed(hs[lvl]) # [b,num_queries,2], tmp is the predicted offset value.
            tmp[..., :1] += reference_before_sigmoid # [b,num_queries,2], only the center coordination add reference point
            outputs_coord = tmp.sigmoid() # [b,num_queries,2]
            outputs_coords.append(outputs_coord)
        outputs_coord = torch.stack(outputs_coords) # [dec_layers,b,num_queries,2]
        out['pred_boxes'] = outputs_coord[-1]

        if self.mask_loss:
            mask_logits = self.mask_head(memory[-1]) # [bs,t,1]
            out['mask_logits'] = mask_logits

        if self.target_type != "none":
            # compute the class-agnostic foreground score
            foreground_logits = self.class_embed(hs) # [dec_layers,b,num_queries,1]
            out['pred_logits'] = foreground_logits[-1]

            if self.training:
                if self.instance_loss:
                    # prepare instance coordination
                    instance_roi_feat = [] 
                    for i, t in enumerate(targets):
                        gt_coordinations = t['segments'].unsqueeze(0) # [1,num_instance,2]->"center,width"
                        clip_feat_i = clip_feat[i].unsqueeze(0) # [1,T,dim]
                        mask_i = mask[i].unsqueeze(0) # [1,T]
                        roi_feat = self._roi_align(gt_coordinations,clip_feat_i,mask_i,self.ROIalign_size).squeeze(dim=0) # [1,num_instance,ROIalign_size,dim]->[num_instance,ROIalign_size,dim]
                        instance_roi_feat.append(roi_feat)
                    instance_roi_feat = torch.cat(instance_roi_feat,dim=0) # [batch_instance_num,ROIalign_size,dim]

                    # visual_feats = self.instance_visual_head(instance_roi_feat)[-1] # [layers, batch_instance_num,ROIalign_size,dim]->[batch_instance_num,ROIalign_size,dim]
                    visual_feats = instance_roi_feat.mean(dim=1) # [batch_instance_num,dim]

                    text_feats_tensors, text_feats_mask = text_feats.decompose()
                    text_feats = self.instance_text_head(text_feats_tensors,src_key_padding_mask=text_feats_mask)[-1] # [layers, num_classes,padding length,dim]->[num_classes,padding length,dim]
                    # text_feats = text_feats.sum(dim=1)/torch.sum(~text_feats_mask,dim=1,keepdim=True) # [num_classes,dim]
                    text_feats = text_feats[:,0,:] # [num_classes,dim] get the firt position token, i.e., class_name token

                    instance_logits = self._compute_similarity(visual_feats,text_feats)
                    out['instance_logits'] = instance_logits # [batch_instance_num,num_classes]
                elif self.matching_loss:
                    # prepare instance coordination
                    instance_roi_feat = [] 
                    for i, t in enumerate(targets):
                        gt_coordinations = t['segments'].unsqueeze(0) # [1,num_instance,2]->"center,width"
                        clip_feat_i = clip_feat[i].unsqueeze(0) # [1,T,dim]
                        mask_i = mask[i].unsqueeze(0) # [1,T]
                        roi_feat = self._roi_align(gt_coordinations,clip_feat_i,mask_i,self.ROIalign_size).squeeze(dim=0) # [1,num_instance,ROIalign_size,dim]->[num_instance,ROIalign_size,dim]
                        instance_roi_feat.append(roi_feat)
                    instance_roi_feat = torch.cat(instance_roi_feat,dim=0) # [batch_instance_num,ROIalign_size,dim]

                    text_feats = text_feats.unsqueeze(0).repeat(instance_roi_feat.shape[0],1,1) # [batch_instance_num,num_classes,dim]
                    
                    if self.matching_loss_type == "fixed":
                        # non-parameters cross-attention
                        scale = torch.tensor(text_feats.shape[-1]).pow(0.5).to(self.device)
                        qk = torch.einsum("bcd,btd->bct",text_feats,instance_roi_feat)
                        qk = qk/scale
                        attn = qk.softmax(-1)
                        res = torch.einsum("bct,btd->bcd",attn,instance_roi_feat)
                    elif self.matching_loss_type == "learnable":
                        res,_ = self.instance_text_head(text_feats,instance_roi_feat,instance_roi_feat) # [batch_instance_num,num_classes,dim]
                    # tgt = torch.cat((text_feats,res),dim=-1) # [batch_query_num,num_classes,2*dim]
                    # matching_logits = self.matching_head(tgt).squeeze(2) # [batch_instance_num,num_classes,1]->[batch_instance_num,num_classes]
                    matching_logits = torch.einsum("bcd,bcd->bcd",text_feats,res).sum(-1) # [batch_instance_num,num_classes]
                    out['matching_logits'] = matching_logits # [batch_instance_num,num_classes]


            # obtain the ROIalign logits
            if not self.training: # only in inference stage
                if self.instance_loss:
                    roi_feat = self._roi_align(out['pred_boxes'],clip_feat,mask,self.ROIalign_size).squeeze() # [bs,num_queries,ROIalign_size,dim]
                    b,q,l,d = roi_feat.shape
                    roi_feat = roi_feat.reshape(b*q,l,d)

                    # visual_feats = self.instance_visual_head(roi_feat)[-1] # [layers, bs*num_queries,ROIalign_size,dim]->[bs*num_queries,ROIalign_size,dim]
                    visual_feats = roi_feat.mean(dim=1).reshape(b,q,d)
                    
                    text_feats_tensors, text_feats_mask = text_feats.decompose()
                    text_feats = self.instance_text_head(text_feats_tensors,src_key_padding_mask=text_feats_mask)[-1] # [layers, num_classes,padding length,dim]->[num_classes,padding length,dim]
                    # text_feats = text_feats.sum(dim=1)/torch.sum(~text_feats_mask,dim=1,keepdim=True) # [num_classes,dim]
                    text_feats = text_feats[:,0,:]
                    
                    instance_logits = self._compute_similarity(visual_feats,text_feats) # [b,num_queries,num_classes]
                    out['ROIalign_logits'] = instance_logits
                    if self.instance_loss_ensemble:
                        fixed_text_feats = self.get_text_feats(classes_name, description_dict, self.device, "prompt") # [N classes,dim]

                        if self.ROIalign_strategy == "before_pred":
                            ROIalign_logits = self._get_roi_prediction_v1(samples,out['pred_boxes'],fixed_text_feats,self.ROIalign_size)
                        else:
                            visual_feats = clip_feat + 1e-8 # avoid the NaN [B,T,dim]
                            snippet_logits = self._compute_similarity(visual_feats,fixed_text_feats) # [b,T,num_classes]
                            ROIalign_logits = self._get_roi_prediction_v2(snippet_logits,mask,out['pred_boxes'],self.ROIalign_size) # this operation must cooperate with segmenatation_loss, [b,num_queries,num_classes]
                        
                        if self.ensemble_strategy == "arithmetic":
                            prob = self.ensemble_rate*instance_logits + (1-self.ensemble_rate)*ROIalign_logits
                        elif self.ensemble_strategy == "geomethric":
                            prob = torch.mul(instance_logits.pow(self.ensemble_rate),ROIalign_logits.pow(1-self.ensemble_rate))
                        else:
                            NotImplementedError
                        out['ROIalign_logits'] = prob 
                elif self.matching_loss:
                    roi_feat = self._roi_align(out['pred_boxes'],clip_feat,mask,self.ROIalign_size).squeeze() # [bs,num_queries,ROIalign_size,dim]
                    b,q,l,d = roi_feat.shape
                    roi_feat = roi_feat.reshape(b*q,l,d)
                    
                    text_feats = text_feats.unsqueeze(0).repeat(roi_feat.shape[0],1,1) # [batch_query_num,num_classes,dim]
                    
                    if self.matching_loss_type == "fixed":
                        # non-parameters cross-attention
                        scale = torch.tensor(text_feats.shape[-1]).pow(0.5).to(self.device)
                        qk = torch.einsum("bcd,btd->bct",text_feats,roi_feat)
                        qk = qk/scale
                        attn = qk.softmax(-1)
                        res = torch.einsum("bct,btd->bcd",attn,roi_feat)
                    elif self.matching_loss_type == "learnable":
                        res,_ = self.instance_text_head(text_feats,roi_feat,roi_feat) # [batch_instance_num,num_classes,dim]
                    # tgt = torch.cat((text_feats,res),dim=-1) # [batch_query_num,num_classes,2*dim]
                    # matching_logits = self.matching_head(tgt).squeeze(2) # [batch_query_num,num_classes,1]->[batch_query_num,num_classes]
                    matching_logits = torch.einsum("bcd,bcd->bcd",text_feats,res).sum(-1) # [batch_query_num,num_classes]
                    out['ROIalign_logits'] = matching_logits.reshape(b,q,-1) # [batch_query_num,num_classes]->[batch,query_num,num_classes]
                    
                else: # if don't use the instance loss, directly adopt clip_visual_feat to classification
                    if self.ROIalign_strategy == "before_pred":
                        ROIalign_logits = self._get_roi_prediction_v1(samples,out['pred_boxes'],text_feats,self.ROIalign_size)
                    else:
                        visual_feats = clip_feat + 1e-8 # avoid the NaN [B,T,dim]
                        if self.mask_loss:
                            visual_feats = mask_logits*visual_feats
                        snippet_logits = self._compute_similarity(visual_feats,text_feats) # [b,T,num_classes]
                        ROIalign_logits = self._get_roi_prediction_v2(snippet_logits,mask,out['pred_boxes'],self.ROIalign_size) # this operation must cooperate with segmenatation_loss, [b,num_queries,num_classes]
                    out['ROIalign_logits'] = ROIalign_logits # update the term of out [b,num_queries,num_classes]

        else:
            detector_logits = self.class_embed(hs) # [dec_layers,b,num_queries,num_classes]
            out['pred_logits'] = detector_logits[-1]

        # aux_loss
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(detector_logits, outputs_coord)
        

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]





def build(args, device):
    if args.target_type != "none" or args.eval_proposal: # adopt one-hot as target, only used in close_set
        num_classes = 1
    else:
        num_classes = args.num_classes

    text_encoder, logit_scale = build_text_encoder(args,device)
    backbone = build_backbone(args)
    transformer = build_transformer(args)
    visual_semantic_head,text_semantic_head = build_semantic_head(args)

    model = ConditionalDETR(
        backbone,
        transformer,
        text_encoder,
        logit_scale,
        device=device,
        num_classes=num_classes,
        visual_semantic_head=visual_semantic_head,
        text_semantic_head=text_semantic_head,
        args=args
    )
    matcher = build_matcher(args)

    weight_dict = {'loss_ce': args.cls_loss_coef, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef
    
    if args.instance_loss: # adopt segmentation_loss
        weight_dict['loss_instance'] = args.instance_loss_coef
    if args.matching_loss:
        weight_dict['loss_matching'] = args.matching_loss_coef
    if args.mask_loss:
        weight_dict['loss_mask'] = args.mask_loss_coef
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)
    
    criterion = build_criterion(args, num_classes, matcher=matcher, weight_dict=weight_dict)
    criterion.to(device)

    postprocessor = build_postprocess(args)

    return model, criterion, postprocessor
