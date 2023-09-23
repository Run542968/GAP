#!/usr/bin/python3
# -*- encoding: utf-8 -*-
"""
@File :  conditional_detr_v2.py
@Time :  2023/09/21 13:34:45
@Author :  Jia-Run Du
@Version :  1.0
@Contact :  dujr6@mail2.sysu.edu.cn
@License :  Copyright (c) ISEE Lab
@Desc :  The v2 model for crop the segment for decouple the localization and classification, adopting ROIalign
"""


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
from models.clip import clip as clip_pkg
import torchvision.ops.roi_align as ROIalign

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
    def __init__(self, backbone, transformer, text_encoder, logit_scale, device, num_classes, args):
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
        self.disable_scale = args.disable_scale
        self.enable_wrapper = args.enable_wrapper

        hidden_dim = transformer.d_model

        if self.target_type != "none":
            self.class_embed = nn.Linear(hidden_dim,hidden_dim)
        else:
            self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 2, 3)
        self.query_embed = nn.Embedding(self.num_queries, hidden_dim)
        self.input_proj = nn.Conv1d(backbone.feat_dim, hidden_dim, kernel_size=1)


        # init prior_prob setting for focal loss
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        if self.target_type != "none":
            self.class_embed.bias.data = torch.ones(hidden_dim) * bias_value
        else:
            self.class_embed.bias.data = torch.ones(num_classes) * bias_value

        # init bbox_mebed
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)

        if self.target_type != "none":
            logger.info(f"The target_type is {self.target_type}, using text embedding as target!")
        else:
            logger.info(f"The target_type is {self.target_type}, using one-hot coding as target!")

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
        else: 
            raise ValueError("Don't define this text_mode.")
        
        tokens = clip_pkg.tokenize(act_prompt).long().to(device) #{input_ids,attention_mask}->input_ids:[150,length],attention_mak:[150,length]
        text_feats = self.text_encoder(tokens).float()

        return text_feats

    def _to_roi_align_format(self, rois, T, scale_factor=1):
        '''Convert RoIs to RoIAlign format.
        Params:
            RoIs: normalized segments coordinates, shape (batch_size, num_segments, 2)
            T: length of the video feature sequence
        '''
        # transform to absolute axis
        B, N = rois.shape[:2]
        rois_center = rois[:, :, 0:1] # [B,N,1]
        rois_size = rois[:, :, 1:2] * scale_factor # [B,N,1]
        rois_abs = torch.cat(
            (rois_center - rois_size/2, rois_center + rois_size/2), dim=2) * T # [B,N,2]->"start,end"
        # expand the RoIs
        rois_abs = torch.clamp(rois_abs, min=0, max=T)  # (B, N, 2)
        # transfer to 4 dimension coordination
        rois_abs_4d = torch.zeros((B,N,4),dtype=rois_abs.dtype,device=rois_abs.device)
        rois_abs_4d[:,:,0], rois_abs_4d[:,:,2] = rois_abs[:,:,0], rois_abs[:,:,1] # x1,0,x2,0

        # add batch index
        batch_ind = torch.arange(0, B).view((B, 1, 1)).to(rois_abs.device) # [B,1,1]
        batch_ind = batch_ind.repeat(1, N, 1) # [B,N,1]
        rois_abs_4d = torch.cat((batch_ind, rois_abs_4d), dim=2) # [B,N,1+4]->"batch_id,x1,0,x2,0"
        # NOTE: stop gradient here to stablize training
        return rois_abs_4d.view((B*N, 5)).detach()

    def _roi_align(self, rois, origin_feat, scale_factor=1):
        B,Q,_ = rois.shape
        B,T,C = origin_feat.shape
        rois_abs_4d = self._to_roi_align_format(rois,T,scale_factor)
        feat = origin_feat.permute(0,2,1) # [B,C,T]
        feat = feat.reshape(B,C,1,T)
        roi_feat = ROIalign(feat, rois_abs_4d, output_size=(1,3))
        roi_feat = roi_feat.reshape(B,Q,C,-1) # [B,Q,C,output_width]
        roi_feat = roi_feat.permute(0,1,3,2) # [B,Q,output_width,C]
        return roi_feat

    def forward(self, samples: NestedTensor, classes_name, description_dict):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched video, of shape [batch_size x T x C]
               - samples.mask: a binary mask of shape [batch_size x T], containing 1 (i.e., true) on padded snippet
            classes_name: the class name of involved category
            description_dict: the dict of description file

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
        features, pos = self.backbone(samples) # list of [b,t,c], list of [b,t]

        # prepare text target
        if self.target_type != "none":
            with torch.no_grad():
                text_feats = self.get_text_feats(classes_name, description_dict, self.device, self.target_type) # 


        # feed into model
        src, mask = features[-1].decompose()
        assert mask is not None
        src = self.input_proj(src.permute(0,2,1)).permute(0,2,1)
        memory, hs, reference = self.transformer(src, mask, self.query_embed.weight, pos[-1]) # [b,t,c], [dec_layers,b,num_queries,c], [b,num_queries,1]
        
        # record result
        reference_before_sigmoid = inverse_sigmoid(reference) # [b,num_queries,1], Reference point is the predicted center point.
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            tmp = self.bbox_embed(hs[lvl]) # [b,num_queries,2], tmp is the predicted offset value.
            tmp[..., :1] += reference_before_sigmoid # [b,num_queries,2], only the center coordination add reference point
            outputs_coord = tmp.sigmoid() # [b,num_queries,2]
            outputs_coords.append(outputs_coord)
        outputs_coord = torch.stack(outputs_coords) # [dec_layers,b,num_queries,2]

        if self.target_type != "none":
            visual_feats = self.class_embed(hs) # [dec_layers,b,num_queries,hidden_dim]

            if self.norm_embed:
                # normalize feature
                visual_feats = visual_feats / visual_feats.norm(dim=-1,keepdim=True)
                text_feats = text_feats / text_feats.norm(dim=-1,keepdim=True)
                if self.disable_scale:
                    outputs_class = torch.einsum("lbqd,cd->lbqc",visual_feats,text_feats)
                else: # default setting
                    outputs_class = torch.einsum("lbqd,cd->lbqc",self.logit_scale*visual_feats,text_feats)
            else:
                outputs_class = torch.einsum("lbqd,cd->lbqc",visual_feats,text_feats) # [dec_layers,b,num_queries,num_classes]
        else:
            outputs_class = self.class_embed(hs) # [dec_layers,b,num_queries,num_classes]

        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]





def build(args, device):
    if args.task == 'close_set':
        if args.binary:
            num_classes = 1
        else:
            num_classes = args.num_classes
    elif args.task == 'zero_shot':
        if args.binary:
            num_classes = 1
        else: # must set it for decide the idx of background class in SetCriterion
            num_classes = int(args.num_classes * args.split / 100)
    else:
        raise ValueError("Don't have this task setting.")


    text_encoder, logit_scale = build_text_encoder(args,device)
    backbone = build_backbone(args)
    transformer = build_transformer(args)
    model = ConditionalDETR(
        backbone,
        transformer,
        text_encoder,
        logit_scale,
        device=device,
        num_classes=num_classes,
        args=args
    )
    matcher = build_matcher(args)

    weight_dict = {'loss_ce': args.cls_loss_coef, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef
    
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)
    
    losses = ['labels', 'boxes']
    criterion = build_criterion(args, num_classes, matcher=matcher, weight_dict=weight_dict, losses=losses, gamma=args.gamma)
    criterion.to(device)

    postprocessor = build_postprocess(args)

    return model, criterion, postprocessor
