#!/usr/bin/python3
# -*- encoding: utf-8 -*-
"""
@File :  position_encoding.py
@Time :  2023/09/13 21:57:13
@Author :  Jia-Run Du
@Version :  1.0
@Contact :  dujr6@mail2.sysu.edu.cn
@License :  Copyright (c) ISEE Lab
@Desc :  modify the positional encoding into temporal format
"""


"""
Various positional encodings for the transformer.
"""
import math
import torch
from torch import nn

from utils.misc import NestedTensor


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors # [b,t,c]
        mask = tensor_list.mask # [b,t]
        assert mask is not None
        not_mask = ~mask
        x_embed = not_mask.cumsum(1, dtype=torch.float32) # b,t
        if self.normalize:
            eps = 1e-6
            x_embed = x_embed / (x_embed[:, -1:] + eps) * self.scale # b,t

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device) # [feat_dim]
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='trunc') / self.num_pos_feats) # [feat_dim]
        
        pos_x = x_embed[:, :, None] / dim_t # [b,t,c]
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2) # [b,t,feat_dim]
        pos = pos_x
        return pos


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.col_embed = nn.Embedding(1000, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors # [b,t,c]
        t, _ = x.shape[-2:]
        i = torch.arange(t, device=x.device) # [t]
        x_emb = self.col_embed(i) # [t,feat_dim]
        pos = x_emb.unsqueeze(0).repeat(x.shape[0], 1, 1) # [b,t,c]
        return pos


def build_position_encoding(args):
    N_steps = args.hidden_dim
    if args.position_embedding in ('v2', 'sine'):
        # TODO find a better way of exposing other arguments
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    elif args.position_embedding in ('v3', 'learned'):
        position_embedding = PositionEmbeddingLearned(N_steps)
    else:
        raise ValueError(f"not supported {args.position_embedding}")

    return position_embedding
