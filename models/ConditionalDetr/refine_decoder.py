

import math
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from .attention import MultiheadAttention

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

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

def gen_sineembed_for_position(pos_tensor,hidden_dim):
    '''
    pos_tensor: [num_queries, b, 1]
    '''
    # n_query, bs, _ = pos_tensor.size()
    # sineembed_tensor = torch.zeros(n_query, bs, 256)
    scale = 2 * math.pi
    dim_t = torch.arange(hidden_dim, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (torch.div(dim_t, 2, rounding_mode='trunc')) / hidden_dim)
    x_embed = pos_tensor[:, :, 0] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos = pos_x # [num_queries, b, hidden_dim]
    return pos



def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")



class RefineDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False, d_model=256, args=None):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.query_scale = MLP(d_model, d_model, d_model, 2)

        self.ref_point_head = MLP(d_model, d_model, d_model, 2)
        for layer_id in range(num_layers - 1):
            self.layers[layer_id + 1].ca_qpos_proj = None
        

    def forward(self, tgt, memory, segment_feat, roi_feat, 
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                roi_pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        '''
            tgt: equal the query_feat, [b,n,c]
            roi_feat: [b,n,l,c]
            memory: [b,t,c]
            memory_key_padding_mask: equal the mask [b,t]
            pos: [t,b,c]
            roi_pos: [b,n,l,c]
            query_pos: euqal the reference point [b,num_queries,1]
        '''
        query_feat = tgt.permute(1,0,2) # [num_queries,b,dim]
        query_pos = query_pos.permute(1,0,2) # [num_queries,b,1]
        memory = memory.permute(1,0,2) # [t,b,dim]
        pos = pos.permute(1,0,2) # [t,b,dim]
        segment_feat = segment_feat.permute(1,0,2) # [num_queries,b,dim]
        roi_feat = roi_feat.permute(1,0,2,3) # [num_queries,b,l,dim]
        roi_pos = roi_pos.permute(1,0,2,3) # [num_queries,b,l,dim]

        t,b,hidden_dim = tgt.shape
        output = segment_feat # the first layer using segment_feat as tgt

        intermediate = []
        reference_points = query_pos.sigmoid().transpose(0,1) # [batch_size, num_queries, 1]

        # reference_points_before_sigmoid = self.ref_point_head(query_pos)    # [num_queries, batch_size, 1]->[num_queries, batch_size,dim]
        # reference_points = reference_points_before_sigmoid.sigmoid().transpose(0, 1) # [batch_size, num_queries, 1]

        for layer_id, layer in enumerate(self.layers):
            obj_center = reference_points[..., :1].transpose(0, 1)      # [num_queries, batch_size, 1]

            # For the first decoder layer, we do not apply transformation over p_s
            if layer_id == 0:
                pos_transformation = 1
            else:
                pos_transformation = self.query_scale(output)

            # get sine embedding for the query vector
            query_sine_embed = gen_sineembed_for_position(obj_center,hidden_dim) # [num_queries, b, c] 
            query_pos = self.ref_point_head(query_sine_embed) # [num_queries, b, c] 

            # apply transformation
            query_sine_embed = query_sine_embed * pos_transformation # [num_queries,b,c]*[1] or [num_queries,b,c]*[num_queries,b,c]
            
            output = layer(output, memory, query_feat, roi_feat, tgt_mask=tgt_mask,
                        memory_mask=memory_mask,
                        tgt_key_padding_mask=tgt_key_padding_mask,
                        memory_key_padding_mask=memory_key_padding_mask,
                        pos=pos, roi_pos=roi_pos, query_pos=query_pos, query_sine_embed=query_sine_embed,
                        is_first=(layer_id == 0)
                        )
            
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return [torch.stack(intermediate).transpose(1, 2), reference_points]

        return output.unsqueeze(0)


class RefineDecoderLayer(nn.Module):

    def __init__(self, d_model, nheads, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        # Decoder Self-Attention
        self.sa_qcontent_proj = nn.Linear(d_model, d_model)
        self.sa_qpos_proj = nn.Linear(d_model, d_model)
        self.sa_kcontent_proj = nn.Linear(d_model, d_model)
        self.sa_kpos_proj = nn.Linear(d_model, d_model)
        self.sa_v_proj = nn.Linear(d_model, d_model)
        self.self_attn = MultiheadAttention(d_model, nheads, dropout=dropout, vdim=d_model)

        # Decoder Cross-Attention Local
        self.ca_qcontent_proj_l = nn.Linear(d_model, d_model)
        self.ca_qpos_proj_l = nn.Linear(d_model, d_model)
        self.ca_kcontent_proj_l = nn.Linear(d_model, d_model)
        self.ca_kpos_proj_l = nn.Linear(d_model, d_model)
        self.ca_v_proj_l = nn.Linear(d_model, d_model)
        self.ca_qpos_sine_proj_l = nn.Linear(d_model, d_model)
        self.cross_attn_l = MultiheadAttention(d_model*2, nheads, dropout=dropout, vdim=d_model)

        # Decoder Cross-Attention Global
        self.ca_qcontent_proj_g = nn.Linear(d_model, d_model)
        self.ca_qpos_proj_g = nn.Linear(d_model, d_model)
        self.ca_kcontent_proj_g = nn.Linear(d_model, d_model)
        self.ca_kpos_proj_g = nn.Linear(d_model, d_model)
        self.ca_v_proj_g = nn.Linear(d_model, d_model)
        self.ca_qpos_sine_proj_g = nn.Linear(d_model, d_model)
        self.cross_attn_g = MultiheadAttention(d_model*2, nheads, dropout=dropout, vdim=d_model)

        self.nheads = nheads

        # Implementation of Feedforward model
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

        self.linear1_l = nn.Linear(d_model, dim_feedforward)
        self.linear2_l = nn.Linear(dim_feedforward, d_model)
        self.norm1_l = nn.LayerNorm(d_model)
        self.norm2_l = nn.LayerNorm(d_model)
        self.dropout1_l = nn.Dropout(dropout)
        self.dropout2_l = nn.Dropout(dropout)
        self.dropout3_l = nn.Dropout(dropout)

        self.linear1_g = nn.Linear(d_model, dim_feedforward)
        self.linear2_g = nn.Linear(dim_feedforward, d_model)
        self.norm1_g = nn.LayerNorm(d_model)
        self.norm2_g = nn.LayerNorm(d_model)
        self.dropout1_g = nn.Dropout(dropout)
        self.dropout2_g = nn.Dropout(dropout)
        self.dropout3_g = nn.Dropout(dropout)

        # Fusion Layer
        self.fusion = MLP(d_model*3,d_model*3,d_model,3)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos


    def forward(self, tgt, memory, query_feat, roi_feat,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     roi_pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None,
                     query_sine_embed = None,
                     is_first = False):
        '''
            tgt: equal the segment_feat, [n,b,c]
            roi_feat: [n,b,l,c]
            memory: [t,b,c]
            memory_key_padding_mask: equal the mask [b,t]
            pos: [t,b,c]
            roi_pos: [n,b,l,c]
            query_pos: euqal the reference point [num_queries,b,c]
        '''

        # ================ Start of segment_feat Cross-Attention Local ==================
        k_content_l = self.ca_kcontent_proj_l(roi_feat) # [n,b,l,c]
        v_l = self.ca_v_proj_l(roi_feat) # [n,b,l,c]

        k_pos_l = self.ca_kpos_proj_l(roi_pos) # [n,b,l,c]

        if is_first: # adopt the Text Feat as query
            q_content_l = self.ca_qcontent_proj_l(tgt) # [n,b,c]
            q_pos_l = self.ca_qpos_proj_l(query_pos) # [n,b,c]
            q_l = q_content_l + q_pos_l  # [n,b,c]
            k_l = k_content_l + k_pos_l  # [n,b,l,c]
        else:
            q_content_l = self.ca_qcontent_proj_l(tgt)
            q_l = q_content_l
            k_l = k_content_l

        num_queries, bs, n_model = q_content_l.shape
        _, _, l, _ = k_content_l.shape

        q_l = q_l.view(num_queries,bs,self.nheads,n_model // self.nheads) # [n,b,nheads,dim/nheads]
        query_sine_embed_l = self.ca_qpos_sine_proj_l(query_sine_embed) # [n,b,dim]
        query_sine_embed_l = query_sine_embed_l.view(num_queries, bs, self.nheads, n_model // self.nheads) # [n,b,nheads,dim/nheads]
        q_l = torch.cat([q_l,query_sine_embed_l],dim=3).view(1,num_queries*bs, n_model*2) # [1,n*b,dim*2]

        k_l = k_l.view(num_queries,bs,l,self.nheads,n_model // self.nheads) # [n,b,l,nheads,dim/nheads]
        k_pos_l = k_pos_l.view(num_queries,bs,l,self.nheads,n_model // self.nheads) # [n,b,l,nheads,dim/nheads]
        k_l = torch.cat([k_l,k_pos_l],dim=4).view(num_queries*bs,l,n_model*2).permute(1,0,2) # [l,n*b,dim*2]

        v_l = v_l.view(num_queries*bs,l,n_model).permute(1,0,2)# [l,n*b,dim*2]
        tgt_l = self.cross_attn_l(query=q_l,
                                  key=k_l,
                                  value=v_l)[0]
        tgt_l = tgt_l.reshape(num_queries,bs,n_model)
        
        tgt_l = tgt + self.dropout2_l(tgt_l)
        tgt_l = self.norm1_l(tgt_l)
        tgt2_l = self.linear2_l(self.dropout1_l(self.activation(self.linear1_l(tgt_l))))
        tgt_l = tgt_l + self.dropout3_l(tgt2_l)
        tgt_l = self.norm2_l(tgt_l)
        # ================ End of segment_feat Cross-Attention Local ==================


        # ================ Start of segment_feat Cross-Attention Global ==================
        k_content_g = self.ca_kcontent_proj_g(memory) # [t,b,c]
        v_g = self.ca_v_proj_g(memory) # [t,b,c]

        k_pos_g = self.ca_kpos_proj_g(pos) # [t,b,c]

        if is_first: # adopt the Text Feat as query
            q_content_g = self.ca_qcontent_proj_g(tgt) # [n,b,c]
            q_pos_g = self.ca_qpos_proj_g(query_pos) # [n,b,c]
            q_g = q_content_g + q_pos_g  # [n,b,c]
            k_g = k_content_g + k_pos_g  # [t,b,c]
        else:
            q_content_g = self.ca_qcontent_proj_g(tgt)
            q_g = q_content_g
            k_g = k_content_g

        num_queries, bs, n_model = q_content_g.shape
        t, _, _ = k_content_g.shape

        q_g = q_g.view(num_queries,bs,self.nheads,n_model // self.nheads) # [n,b,nheads,dim/nheads]
        query_sine_embed_g = self.ca_qpos_sine_proj_g(query_sine_embed) # [n,b,dim]
        query_sine_embed_g = query_sine_embed_g.view(num_queries, bs, self.nheads, n_model // self.nheads) # [n,b,nheads,dim/nheads]
        q_g = torch.cat([q_g,query_sine_embed_g],dim=3).view(num_queries, bs, n_model*2) # [n,b,dim*2]

        k_g = k_g.view(t,bs,self.nheads,n_model // self.nheads) # [t,b,nheads,dim/nheads]
        k_pos_g = k_pos_g.view(t,bs,self.nheads,n_model // self.nheads) # [t,b,nheads,dim/nheads]
        k_g = torch.cat([k_g,k_pos_g],dim=3).view(t,bs,n_model*2) # [t,b,dim*2]
        
        tgt_g = self.cross_attn_g(query=q_g,
                                  key=k_g,
                                  value=v_g,
                                  key_padding_mask=memory_key_padding_mask)[0] # [n,b,dim]
        
        tgt_g = tgt + self.dropout2_g(tgt_g)
        tgt_g = self.norm1_g(tgt_g)
        tgt2_g = self.linear2_g(self.dropout1_g(self.activation(self.linear1_g(tgt_g))))
        tgt_g = tgt_g + self.dropout3_g(tgt2_g)
        tgt_g = self.norm2_g(tgt_g) # [n,b,dim]
        # ================ End of segment_feat Cross-Attention Global ==================

        fusion_feat = torch.cat([query_feat,tgt_l,tgt_g],dim=2) # [n,b,3*dim]
        fusion_query_feat = self.fusion(fusion_feat) # [n,b,dim]

        # ================ Start of fusion_query_feat Self-attention ================
        q_content = self.sa_qcontent_proj(fusion_query_feat)      # target is the input of the first decoder layer. zero by default.
        q_pos = self.sa_qpos_proj(query_pos)
        k_content = self.sa_kcontent_proj(fusion_query_feat)
        k_pos = self.sa_kpos_proj(query_pos)
        v = self.sa_v_proj(fusion_query_feat)

        num_queries, bs, n_model = q_content.shape

        q = q_content + q_pos
        k = k_content + k_pos

        fusion_query_feat2 = self.self_attn(q, k, value=v)[0]

        fusion_query_feat2 = fusion_query_feat + self.dropout(fusion_query_feat2)
        fusion_query_feat2 = self.norm(fusion_query_feat2) # [n,b,dim]
        # ================ End of fusion_query_feat Self-attention ==================

        return fusion_query_feat2




class RefineDecoderV2(nn.Module):
    def __init__(self, refine_layer, num_layers) -> None:
        super().__init__()
        self.layers = num_layers
        self.layers = _get_clones(refine_layer, num_layers)

    def forward(self, query_feat, video_feat, roi_segment_feat,
                video_feat_key_padding_mask: Optional[Tensor] = None,
                video_pos: Optional[Tensor] = None,
                roi_pos: Optional[Tensor] = None):
        
        output = query_feat
        for layer in self.layers:
            output = layer(query_feat, video_feat, roi_segment_feat, video_feat_key_padding_mask, video_pos, roi_pos)
        
        return output

class RefineDecoderV2_layer(nn.Module):

    def __init__(self, nheads=4, d_model=256, args=None):
        super().__init__()
        self.d_model = d_model
        self.cross_attn_local = nn.MultiheadAttention(d_model,nheads)
        self.refine_drop_saResidual = args.refine_drop_saResidual
        self.refine_drop_sa = args.refine_drop_sa
        self.refine_fusion_type = args.refine_fusion_type
        self.refine_cat_type = args.refine_cat_type
        if "concat" in self.refine_cat_type:
            self.proj_head = nn.Linear(2*d_model,d_model)
            self.self_attn = nn.MultiheadAttention(2*d_model,nheads)
        else:
            self.self_attn = nn.MultiheadAttention(d_model,nheads)


    def forward(self, query_feat, video_feat, roi_segment_feat,
                video_feat_key_padding_mask: Optional[Tensor] = None,
                video_pos: Optional[Tensor] = None,
                roi_pos: Optional[Tensor] = None):
        '''
            query_feat: [b,n,c]
            roi_segment_feat: [b,n,l,c]
            video_feat: [b,t,c]
            video_feat_key_padding_mask: equal the mask [b,t]
            video_pos: [b,t,c]
            roi_pos: [b,n,l,c]
        '''
        if self.refine_fusion_type == "ca":
            # cross-attetion in query_embed and sement feat
            query_feat = query_feat.permute(1,0,2) # [num_queries,b,dim]
            segment_feat = roi_segment_feat.permute(2,1,0,3) # [l,num_queries,b,dim]
            l,n,b,dim = segment_feat.shape
            segment_feat = segment_feat.reshape(l,n*b,dim) # [l,n*b,dim]
            query_feat_seg = query_feat.reshape(1,n*b,dim) # [1,n*b,dim]

            tgt1 = self.cross_attn_local(query=query_feat_seg,
                                        key=segment_feat,
                                        value=segment_feat)[0] # [1,n*b,dim]
            tgt1 = tgt1.reshape(n,b,dim)
        elif self.refine_fusion_type == "mean":
            query_feat = query_feat.permute(1,0,2) # [num_queries,b,dim]
            segment_feat = roi_segment_feat.permute(2,1,0,3) # [l,num_queries,b,dim]
            tgt1 = segment_feat.mean(0) # [n,b,dim]
        elif self.refine_fusion_type == "max":
            query_feat = query_feat.permute(1,0,2) # [num_queries,b,dim]
            segment_feat = roi_segment_feat.permute(2,1,0,3) # [l,num_queries,b,dim]
            tgt1 = segment_feat.max(0)[0] # [n,b,dim]
        else:
            raise NotImplementedError



        if self.refine_drop_sa:
            query_feat = tgt1
        else:
            if "concat" in self.refine_cat_type:
                query_feat = torch.cat([query_feat, tgt1], dim=-1) # [n,b,2*dim]
            elif self.refine_cat_type == "sum":
                query_feat = query_feat + tgt1
            else:
                raise ValueError

            # self-attetnion between different query
            tgt2 = self.self_attn(query=query_feat,
                                key=query_feat,
                                value=query_feat)[0]
            if self.refine_drop_saResidual:
                if self.refine_cat_type == "concat1":
                    n,b,dim = tgt2.shape
                    query_feat = tgt2[:,:,0:self.d_model]
                elif self.refine_cat_type == "concat2":
                    query_feat = self.proj_head(query_feat) # [n,b,2*dim]->[n,b,dim]
                elif self.refine_cat_type == "sum":
                    query_feat = tgt2
            else:
                query_feat = query_feat + tgt2

        query_feat = query_feat.permute(1,0,2) # [b,n,dim]
        return query_feat


def build_refine_decoder(args):

    refine_layer = RefineDecoderV2_layer(nheads=args.nheads,
                           d_model=args.hidden_dim,
                           args=args)
    return RefineDecoderV2(refine_layer=refine_layer,
                           num_layers=args.refine_layer_num)
