from .position_encoding import build_position_encoding
import math
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor


class Semantic_Head(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_layers=1,
                 dim_feedforward=2048, dropout=0.1,
                 activation="relu", 
                 normalize_before=False,
                 return_intermediate=False):
        super().__init__()
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        norm = nn.LayerNorm(d_model) if normalize_before else None
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        intermediate = []
        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)
            if self.return_intermediate:
                intermediate.append(output)

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)
        
        if self.return_intermediate:
            return torch.stack(intermediate,dim=0) # [layers,t,b,c]

        return output


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


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

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")



class Naive_Semantic_Head(nn.Module):
    '''
    most naive structure that only adopt a network and a residual connection
    '''
    def __init__(self, d_model, nhead, type):
        super().__init__()
        if type == "MLP":
            self.instance_head = MLP(d_model,d_model,d_model,3)
        elif type == "Conv":
            self.instance_head = nn.Conv1d(d_model,d_model, kernel_size=3, padding=1)
        elif type == "MHA":
            self.instance_head = nn.MultiheadAttention(d_model,nhead,batch_first=True)
        else:
            raise ValueError(f"Don't have this instance_head_type:{type}")
        self.type = type

    def forward(self,input,src_key_padding_mask=None):
        if self.type == "MLP":
            feats = self.instance_head(input) # [batch_instance_num,ROIalign_size,dim]
        elif self.type == "Conv":
            feats = self.instance_head(input.permute(0,2,1)).permute(0,2,1) # [batch_instance_num,ROIalign_size,dim]
        elif self.type == "MHA":
            feats, _ = self.instance_head(input,input,input,key_padding_mask=src_key_padding_mask) # [batch_instance_num,ROIalign_size,dim]
        else:
            raise ValueError(f"Don't have this instance_head_type:{self.instance_head_type}")
        feats = feats + input # residual connection
        return [feats]
    
def build_semantic_head(args):
    '''
    v1: most naive structure that only adopt a self-attention and a residual connection
    v2: transformer encoder
    '''
    if args.instance_loss or args.segmentation_loss:
        if args.semantic_head_version == "v1":
            position_embedding = None
            visual_semantic_head = Naive_Semantic_Head(d_model=args.hidden_dim,
                                                    nhead=args.semantic_visual_nheads,
                                                    type=args.instance_head_type)
            text_semantic_head = Naive_Semantic_Head(d_model=args.hidden_dim,
                                                    nhead=args.semantic_text_nheads,
                                                    type="MHA")
        elif args.semantic_head_version == "v2":
            assert args.instance_head_type == "MHA", f"only MHA is adopt in this semantic_head_version, not current {args.instance_head_type}."
            position_embedding = build_position_encoding(args)
            visual_semantic_head = Semantic_Head(d_model=args.hidden_dim,
                                                nhead=args.semantic_visual_nheads,
                                                num_layers=args.semantic_visual_layers,
                                                dim_feedforward=args.dim_feedforward,
                                                dropout=args.semantic_visual_dropout,
                                                normalize_before = False,
                                                return_intermediate=True)
            text_semantic_head = Semantic_Head(d_model=args.hidden_dim,
                                                nhead=args.semantic_text_nheads,
                                                num_layers=args.semantic_text_layers,
                                                dim_feedforward=args.dim_feedforward,
                                                dropout=args.semantic_text_dropout,
                                                normalize_before = False,
                                                return_intermediate=True)
        else:
            raise ValueError(f"Don't define this semantic_head_version:{args.semantic_head_version}") 
        return visual_semantic_head,text_semantic_head
    else:
        return None, None