"""Define Graph Pooling."""
import math
from functools import partial
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import dgl.geometry as dgl_geo
from .utils import segment_mean_nd
import time
import sys
import os
class Attention(nn.Module):
    """Similar to timm.models.vision_transformer.Attention but we do not use
    additional Fully Connected Layers.
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale # (B, num_head, L, L)
        if attn_mask is not None:
            attn = attn + attn_mask.unsqueeze(1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    """Same as timm.models.vision_transformer.Block
    """

    def __init__(self, dim, num_heads, qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.bias = nn.Parameter(torch.zeros(dim).normal_(0, 1e-2))

    def forward(self, x, attn_mask=None):
        x = x + self.drop_path(self.attn(self.norm(x), attn_mask=attn_mask))

        x = x - torch.mean(x, dim=1, keepdim=True) + self.bias.view(1, 1, -1)

        return x

class LayerScale(nn.Module):
    def __init__(self, dim, init_value=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones(dim))

    def forward(self, x):
        return self.gamma * x

class GraphPooling(nn.Module):

    def __init__(self,
                 num_clusters=4,
                 d_model=512,
                 dropout=0.1,
                 l2_normalize_for_fps=True,
                 num_heads=12,
                 qkv_bias=True,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        """Perfrom Graph Pooling.

        Args:
          num_clusters: A scalar indicates the number of centroids
          d_model: A scalar indicates the input channels to Transformer
          dropout: A `float` indicates the dropout rate
          l2_normalize_for_fps: Enable/disable L2-noramlization before performing
                                Farthest Point Sampling
          num_heads: A scalar indicates the number of attention head
          qkv_bias: Enable/disable bias in the attention layer
          norm_layer: A Norm layer used in the attention layer
        """
        super().__init__()
        self.centroid_fc = Block(dim=d_model,
                                 num_heads=num_heads,
                                 qkv_bias=qkv_bias,
                                 norm_layer=norm_layer)
        self.fc1 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4, bias=True),
            nn.GELU(),
            nn.Dropout(dropout))
        self.fc2 = nn.Sequential(
            nn.LayerNorm(d_model * 4),
            nn.Linear(d_model * 4, d_model, bias=True))

        self._num_clusters = num_clusters
        self._l2_normalize_for_fps = l2_normalize_for_fps

    def _fill_with_mean(self, src, mask):
        """A helper function to fill invalid entries with mean values.
        """
        bs, sl, cs = src.shape
        if mask is not None:
            mean_src = valid_mean(src, ~mask).unsqueeze(1).type_as(src)
            # Fill padded entries with mean values.
            fill_mask = mask.unsqueeze(2).expand(-1, -1, cs)
            filled_src = torch.where(fill_mask, mean_src.expand(-1, sl, -1), src)
        else:
            mean_src = torch.mean(src, dim=1, keepdim=True).type_as(src)
            filled_src = src

        return filled_src, mean_src

    def forward(self, cls_token, src, mask):
        """Feedforward for clustering with Transformer.

        Args:
          cls_token: A `tensor` of shape `[batch_size, 1, channels]`
          src: A `tensor` of shape `[batch_size, source_sequence_length, channels]`
          mask: A bool `tensor` of shape `[batch_size, sequence_length]`, where
                `True` indicates empty/padded elements.

        Returns:
          cls_token: A `tensor` of shape `[batch_size, 1, channels]`
          centroids: A `tensor` of shape `[batch_size, num_clusters, channels]`
          logits: A `tensor` of shape
            `[batch_size, source_sequence_length, num_clusters]`
          sampled_inds: A `tensor` of shape
            `[batch_size, num_clusters]`
        """
        bs, sl, cs = src.shape

        # Sample query by Farthest Point Sampling.
        # `centroids` is of shape `[batch_size, target_sequence_length, channels]`.
        filled_src, mean_src = self._fill_with_mean(src, mask)
        padded_src = torch.cat([mean_src, filled_src], dim=1)

        if self._l2_normalize_for_fps:
            sampling_src = F.normalize(padded_src, dim=-1)
        else:
            sampling_src = padded_src

        # torch.cuda.synchronize()
        # start_time = time.time()

        # print(sampling_src[0, 55:58, :10])
        sampled_inds = dgl_geo.farthest_point_sampler(
            sampling_src.to(torch.float64),
            self._num_clusters + 1,
            0).long()
        sampled_inds = sampled_inds[:, 1:] - 1

        assert((sampled_inds >= 0).all()) # Make sure sampling from the squence
        unfold_sampled_inds = sampled_inds.unsqueeze(2).expand(-1, -1, cs)

        # Apply attention layer to predict grouping
        node_features = self.centroid_fc(src)
        centroid_features = torch.gather(node_features, 1, unfold_sampled_inds)

        # Group squence of tokens into clusters
        normed_centroid_features = F.normalize(centroid_features, dim=-1)
        normed_node_features = F.normalize(node_features, dim=-1)
        logits = torch.einsum(
            'bij,bjk->bik', normed_node_features, normed_centroid_features.transpose(1, 2))
        logits = logits * 5
        assignments = torch.softmax(logits, dim=-1)

        # Average pooling within clusters.
        fc1_cls_token_src = self.fc1(torch.cat([cls_token, src], dim=1))
        fc1_cls_token, fc1_src = fc1_cls_token_src[:, :1], fc1_cls_token_src[:, 1:]
        normalizer = torch.einsum('bij,bjk->bik', assignments.transpose(1, 2),
                                  torch.ones((bs, sl, 1), dtype=src.dtype, device=src.device))
        centroids = torch.einsum('bij,bjk->bik', assignments.transpose(1, 2), fc1_src)
        centroids /= normalizer

        fc2_cls_token_centroids = self.fc2(torch.cat([fc1_cls_token, centroids], dim=1))
        centroids = fc2_cls_token_centroids[:, 1:, :] + torch.gather(src, 1, unfold_sampled_inds)
        cls_token = fc2_cls_token_centroids[:, :1, :] + cls_token

        return cls_token, centroids, logits, sampled_inds

class GraphPoolingReg(GraphPooling):
    def __init__(self, layerscale=False, **kwargs):
        super().__init__(**kwargs)
        self.ls = LayerScale(kwargs['d_model']) if layerscale else nn.Identity()

    def forward(self, cls_token, src, mask, num_clusters=None):
        bs, sl, cs = src.shape

        # Sample query by Farthest Point Sampling.
        # `centroids` is of shape `[batch_size, target_sequence_length, channels]`.
        filled_src, mean_src = self._fill_with_mean(src, mask)
        padded_src = torch.cat([mean_src, filled_src], dim=1)

        if self._l2_normalize_for_fps:
            sampling_src = F.normalize(padded_src, dim=-1)
        else:
            sampling_src = padded_src

        num_clusters = num_clusters or self._num_clusters
        sampled_inds = dgl_geo.farthest_point_sampler(
            sampling_src.to(torch.float64),
            num_clusters+1,
            0).long()
        sampled_inds = sampled_inds[:, 1:] - 1

        assert((sampled_inds >= 0).all()) # Make sure sampling from the squence
        unfold_sampled_inds = sampled_inds.unsqueeze(2).expand(-1, -1, cs)

        # Apply attention layer to predict grouping
        node_features = self.centroid_fc(src)
        centroid_features = torch.gather(node_features, 1, unfold_sampled_inds)

        # Group squence of tokens into clusters
        normed_centroid_features = F.normalize(centroid_features, dim=-1) # (B, N, C)
        normed_node_features = F.normalize(node_features, dim=-1) # (B, L, C)
        logits = torch.einsum(
            'bij,bjk->bik', normed_node_features, normed_centroid_features.transpose(1, 2)) # (B, L, N)
        logits = logits * 5
        assignments = torch.softmax(logits, dim=-1) # N-dim: attn_up

        # Average pooling within clusters.
        num_prefix = cls_token.shape[1] # B, n, C -> n
        fc1_cls_reg_token_src = self.fc1(torch.cat([cls_token, src], dim=1))
        fc1_cls_reg_token, fc1_src = fc1_cls_reg_token_src[:, :num_prefix], fc1_cls_reg_token_src[:, num_prefix:]
        normalizer = torch.einsum('bij,bjk->bik', assignments.transpose(1, 2),
                                  torch.ones((bs, sl, 1), dtype=src.dtype, device=src.device))  #(B, N, 1)
        centroids = torch.einsum('bij,bjk->bik', assignments.transpose(1, 2), fc1_src)
        centroids /= normalizer

        fc2_cls_reg_token_centroids = self.ls(self.fc2(torch.cat([fc1_cls_reg_token, centroids], dim=1)))
        centroids = fc2_cls_reg_token_centroids[:, num_prefix:, :] + torch.gather(src, 1, unfold_sampled_inds)
        cls_reg_token = fc2_cls_reg_token_centroids[:, :num_prefix, :] + cls_token

        return cls_reg_token, centroids, logits, sampled_inds

    def get_intermediate_layers(self, cls_token, src, mask, num_clusters=None, index=0):
        bs, sl, cs = src.shape
        out = {}
        # Sample query by Farthest Point Sampling.
        # `centroids` is of shape `[batch_size, target_sequence_length, channels]`.
        filled_src, mean_src = self._fill_with_mean(src, mask)
        padded_src = torch.cat([mean_src, filled_src], dim=1)

        if self._l2_normalize_for_fps:
            sampling_src = F.normalize(padded_src, dim=-1)
        else:
            sampling_src = padded_src

        num_clusters = num_clusters or self._num_clusters
        sampled_inds = dgl_geo.farthest_point_sampler(
            sampling_src.to(torch.float64),
            num_clusters+1,
            0).long()
        sampled_inds = sampled_inds[:, 1:] - 1

        assert((sampled_inds >= 0).all()) # Make sure sampling from the squence
        unfold_sampled_inds = sampled_inds.unsqueeze(2).expand(-1, -1, cs)

        # Apply attention layer to predict grouping
        node_features = self.centroid_fc(src)
        centroid_features = torch.gather(node_features, 1, unfold_sampled_inds)
        out[f'p{index}'] = node_features
        # Group squence of tokens into clusters
        normed_centroid_features = F.normalize(centroid_features, dim=-1) # (B, N, C)
        normed_node_features = F.normalize(node_features, dim=-1) # (B, M, C)
        logits = torch.einsum(
            'bij,bjk->bik', normed_node_features, normed_centroid_features.transpose(1, 2)) # (B, N, M)
        logits = logits * 5
        assignments = torch.softmax(logits, dim=-1) # M-dim: attn_up, 

        # Average pooling within clusters.
        num_prefix = cls_token.shape[1] # B, n, C -> n
        fc1_cls_reg_token_src = self.fc1(torch.cat([cls_token, src], dim=1))
        fc1_cls_reg_token, fc1_src = fc1_cls_reg_token_src[:, :num_prefix], fc1_cls_reg_token_src[:, num_prefix:]
        normalizer = torch.einsum('bij,bjk->bik', assignments.transpose(1, 2),
                                  torch.ones((bs, sl, 1), dtype=src.dtype, device=src.device))
        centroids = torch.einsum('bij,bjk->bik', assignments.transpose(1, 2), fc1_src)
        centroids /= normalizer
        out[f'r{index}'] = centroids

        fc2_cls_reg_token_centroids = self.ls(self.fc2(torch.cat([fc1_cls_reg_token, centroids], dim=1)))
        centroids = fc2_cls_reg_token_centroids[:, num_prefix:, :] + torch.gather(src, 1, unfold_sampled_inds)
        out[f'f{index}'] = torch.gather(src, 1, unfold_sampled_inds)
        out[f'g{index}'] =  centroids
        cls_reg_token = fc2_cls_reg_token_centroids[:, :num_prefix, :] + cls_token
        

        return cls_reg_token, centroids, logits, sampled_inds, out

class GraphPoolingEM(GraphPooling):
    def __init__(self, n_iters=2, layerscale=False, **kwargs):
        super().__init__(**kwargs)
        self.ls = LayerScale(kwargs['d_model']) if layerscale else nn.Identity()
        self.n_iters = n_iters

    def forward(self, cls_token, src, mask, num_clusters=None):
        bs, sl, cs = src.shape

        # Sample query by Farthest Point Sampling.
        # `centroids` is of shape `[batch_size, target_sequence_length, channels]`.
        filled_src, mean_src = self._fill_with_mean(src, mask)
        padded_src = torch.cat([mean_src, filled_src], dim=1)

        if self._l2_normalize_for_fps:
            sampling_src = F.normalize(padded_src, dim=-1)
        else:
            sampling_src = padded_src

        num_clusters = num_clusters or self._num_clusters
        sampled_inds = dgl_geo.farthest_point_sampler(
            sampling_src.to(torch.float64),
            num_clusters+1,
            0).long()
        sampled_inds = sampled_inds[:, 1:] - 1

        assert((sampled_inds >= 0).all()) # Make sure sampling from the squence
        unfold_sampled_inds = sampled_inds.unsqueeze(2).expand(-1, -1, cs)

        # Apply attention layer to predict grouping
        node_features = self.centroid_fc(src)
        centroid_features = torch.gather(node_features, 1, unfold_sampled_inds)
        normed_node_features = F.normalize(node_features, dim=-1) # (B, L, C)

        # perform Spherical EM with residual to get better assignment
        for i in range(self.n_iters):
            # E-step
            normed_centroid_features = F.normalize(centroid_features, dim=-1) # (B, N, C)
            logits = torch.einsum(
                'bij,bjk->bik', normed_node_features, normed_centroid_features.transpose(1, 2)) # (B, L, N)
            logits = logits * 5
            assignments = torch.softmax(logits, dim=-1) # N-dim: (B, L, N)

            # M-step
            if i == self.n_iters - 1:
                break
            denom = assignments.transpose(1, 2).sum(dim=-1, keepdim=True).clamp(min=1e-6) # L-dim: (B, N, 1)
            update = torch.einsum('bln,blc->bnc', assignments, node_features) / denom # (B, N, C)
            centroid_features = centroid_features + update
        
        num_prefix = cls_token.shape[1] # B, n, C -> n
        fc1_cls_reg_token_src = self.fc1(torch.cat([cls_token, src], dim=1))
        fc1_cls_reg_token, fc1_src = fc1_cls_reg_token_src[:, :num_prefix], fc1_cls_reg_token_src[:, num_prefix:]
        normalizer = torch.einsum('bij,bjk->bik', assignments.transpose(1, 2),
                                torch.ones((bs, sl, 1), dtype=src.dtype, device=src.device))  #(B, N, 1)
        centroids = torch.einsum('bij,bjk->bik', assignments.transpose(1, 2), fc1_src)
        centroids /= normalizer

        fc2_cls_reg_token_centroids = self.ls(self.fc2(torch.cat([fc1_cls_reg_token, centroids], dim=1)))
        centroids = fc2_cls_reg_token_centroids[:, num_prefix:, :] + torch.gather(src, 1, unfold_sampled_inds)
        cls_reg_token = fc2_cls_reg_token_centroids[:, :num_prefix, :] + cls_token

        return cls_reg_token, centroids, logits, sampled_inds

class GraphPoolingSK(GraphPooling):
    def __init__(self, n_iters=2, layerscale=False, **kwargs):
        super().__init__(**kwargs)
        self.ls = LayerScale(kwargs['d_model']) if layerscale else nn.Identity()
        self.n_iters = n_iters

    def forward(self, cls_token, src, mask, num_clusters=None):
        bs, sl, cs = src.shape

        # Sample query by Farthest Point Sampling.
        # `centroids` is of shape `[batch_size, target_sequence_length, channels]`.
        filled_src, mean_src = self._fill_with_mean(src, mask)
        padded_src = torch.cat([mean_src, filled_src], dim=1)

        if self._l2_normalize_for_fps:
            sampling_src = F.normalize(padded_src, dim=-1)
        else:
            sampling_src = padded_src

        num_clusters = num_clusters or self._num_clusters
        sampled_inds = dgl_geo.farthest_point_sampler(
            sampling_src.to(torch.float64),
            num_clusters+1,
            0).long()
        sampled_inds = sampled_inds[:, 1:] - 1

        assert((sampled_inds >= 0).all()) # Make sure sampling from the squence
        unfold_sampled_inds = sampled_inds.unsqueeze(2).expand(-1, -1, cs)

        # Apply attention layer to predict grouping
        node_features = self.centroid_fc(src)
        centroid_features = torch.gather(node_features, 1, unfold_sampled_inds)
        normed_node_features = F.normalize(node_features, dim=-1) # (B, L, C)
        normed_centroid_features = F.normalize(centroid_features, dim=-1) # (B, N, C)
        logits = torch.einsum(
            'bij,bjk->bik', normed_node_features, normed_centroid_features.transpose(1, 2)) # (B, L, N)
        logits = logits * 8
        assignments = torch.softmax(logits, dim=-1) # N-dim: (B, L, N)
        assignments = self.sinkhorn(assignments, n_iters=self.n_iters)
        
        num_prefix = cls_token.shape[1] # B, n, C -> n
        fc1_cls_reg_token_src = self.fc1(torch.cat([cls_token, src], dim=1))
        fc1_cls_reg_token, fc1_src = fc1_cls_reg_token_src[:, :num_prefix], fc1_cls_reg_token_src[:, num_prefix:]
        normalizer = torch.einsum('bij,bjk->bik', assignments.transpose(1, 2),
                                torch.ones((bs, sl, 1), dtype=src.dtype, device=src.device))  #(B, N, 1)
        centroids = torch.einsum('bij,bjk->bik', assignments.transpose(1, 2), fc1_src)
        centroids /= normalizer

        fc2_cls_reg_token_centroids = self.ls(self.fc2(torch.cat([fc1_cls_reg_token, centroids], dim=1)))
        centroids = fc2_cls_reg_token_centroids[:, num_prefix:, :] + torch.gather(src, 1, unfold_sampled_inds)
        cls_reg_token = fc2_cls_reg_token_centroids[:, :num_prefix, :] + cls_token

        return cls_reg_token, centroids, logits, sampled_inds

    def sinkhorn(self, A, n_iters=3, eps=0.05):
        # A: B, L, N >=0
        for _ in range(n_iters):
            A = A / (A.sum(dim=1, keepdim=True) + 1e-9)
            A = A / (A.sum(dim=2, keepdim=True) + 1e-9)
        return A.clamp(min=1e-9)

class GroupingSlotAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        dropout=0.1,
        mlp_ratio= 4.0,
        n_iters = 3,
        layerscale = False
    ):
        super().__init__()
        
        self.n_iters = n_iters
        # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio), bias=qkv_bias),
            nn.GELU(),
            nn.Dropout(dropout))
        self.r_proj = nn.Sequential(
            nn.LayerNorm(int(dim * mlp_ratio)),
            nn.Linear(int(dim * mlp_ratio), dim, bias=True))
        
        self.norm_src = nn.LayerNorm(dim, eps=1e-6)
        self.norm_centroid = nn.LayerNorm(dim, eps=1e-6)
        self.norm_update = nn.LayerNorm(dim, eps=1e-6)
        
        scale = qk_scale or dim**-0.5 
        self.tau = nn.Parameter(torch.ones([]) * np.log(scale * 10)) # scalar
        self.epsilon = 1e-8

        self.attn_drop_rate = attn_drop
        self.attn_drop = nn.Dropout(self.attn_drop_rate)

        self.ls1 = LayerScale(dim) if layerscale else nn.Identity()

    def forward(self, src, centroid):
        # reshaping x_in, x_out
        B, L, C_ = src.shape
        B, N, C = centroid.shape
        assert L > N
        
        # key value preperation
        src = self.norm_src(src)
        k = self.k_proj(src) # (B, L, C)
        v = self.v_proj(src) # (B, L, 4C) 

        for _ in range(self.n_iters):
            centroid = self.norm_centroid(centroid)
            q = self.q_proj(centroid) # (B, N, C)
        
            # get attn_up
            attn = torch.einsum('bnd,bld->bnl', q, k)  * self.tau.exp() # (B, N, L)   
            attn_up = torch.softmax(attn, dim=1) + self.epsilon # hw dim

            # get attn_down
            attn_down = attn_up / (attn_up.sum(dim=-1, keepdim=True) + self.epsilon) # (B, hw, HW)        

            # calcuate Av
            attn_down = self.attn_drop(attn_down)
            updates = torch.einsum('bld,bnl->bnd', v, attn_down) # (B, N, 4C)
            updates = self.r_proj(updates)

            centroid = centroid + self.ls1(self.norm_update(updates)) 
            # centroid = centroid + self.ls2(self.norm2(self.mlp(centroid)))

        return centroid, attn_up, attn_down

class GraphPoolingSlot(nn.Module):
    def __init__(self,
                 num_clusters=4,
                 d_model=512,
                 l2_normalize_for_fps=True,
                 n_iters=3,
                 layerscale=False,
                 hard_group=False):
        super().__init__()

        self._num_clusters = num_clusters
        self._l2_normalize_for_fps = l2_normalize_for_fps
        self.hard_group = hard_group
        self.slot_attn = GroupingSlotAttention(dim= d_model,  n_iters = n_iters)

    def _fill_with_mean(self, src, mask):
        bs, sl, cs = src.shape
        if mask is not None:
            mean_src = valid_mean(src, ~mask).unsqueeze(1).type_as(src)
            # Fill padded entries with mean values.
            fill_mask = mask.unsqueeze(2).expand(-1, -1, cs)
            filled_src = torch.where(fill_mask, mean_src.expand(-1, sl, -1), src)
        else:
            mean_src = torch.mean(src, dim=1, keepdim=True).type_as(src)
            filled_src = src

        return filled_src, mean_src

    def forward(self, src, mask, num_clusters=None):
        bs, sl, cs = src.shape

        # Sample query by Farthest Point Sampling.
        # `centroids` is of shape `[batch_size, target_sequence_length, channels]`.
        filled_src, mean_src = self._fill_with_mean(src, mask)
        padded_src = torch.cat([mean_src, filled_src], dim=1)

        if self._l2_normalize_for_fps:
            sampling_src = F.normalize(padded_src, dim=-1)
        else:
            sampling_src = padded_src

        num_clusters = num_clusters or self._num_clusters
        sampled_inds = dgl_geo.farthest_point_sampler(
            sampling_src.to(torch.float64),
            num_clusters+1,
            0).long()
        sampled_inds = sampled_inds[:, 1:] - 1

        assert((sampled_inds >= 0).all()) # Make sure sampling from the squence
        unfold_sampled_inds = sampled_inds.unsqueeze(2).expand(-1, -1, cs)

        # Apply attention layer to predict grouping
        centroid = torch.gather(src, 1, unfold_sampled_inds)

        centroid, attn_up, attn_down = self.slot_attn(src, centroid)
        return centroid, attn_up, attn_down


def valid_mean(x, mask):
     """Compute mean of x given valid mask.

     Args:
         x: A `float` tensor of shape `[batch_size, num_nodes, channels]`
         mask: A `bool` tensor of shape `[batch_size, num_nodes]`, where
             `True` indicates the entry is valid/padded

     Returns:
         mean_x: A `float` tensor of shape `[batch_size, channels]`
     """
     mask = mask.type_as(x).unsqueeze(2)
     sum_mask = torch.clamp(torch.sum(mask, dim=1), min=1)
     masked_x = x * mask
     mean_x = torch.sum(masked_x, dim=1) / sum_mask

     return mean_x


class TextGraphPooling(nn.Module):

    def __init__(self,
                 num_clusters=3,
                 d_model=512,
                 dropout=0.1,
                 l2_normalize_for_fps=True,
                 num_heads=8,
                 qkv_bias=True,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6)):

        super().__init__()
        self.centroid_fc = Block(dim=d_model,
                                 num_heads=num_heads,
                                 qkv_bias=qkv_bias,
                                 norm_layer=norm_layer)
        self.fc1 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4, bias=True),
            nn.GELU(),
            nn.Dropout(dropout))
        self.fc2 = nn.Sequential(
            nn.LayerNorm(d_model * 4),
            nn.Linear(d_model * 4, d_model, bias=True))

        self._num_clusters = num_clusters
        self._l2_normalize_for_fps = l2_normalize_for_fps

    def _fill_with_mean(self, src, mask):
        """A helper function to fill invalid entries with mean values.
        """
        bs, sl, cs = src.shape
        if False:#mask is not None:
            mean_src = valid_mean(src, ~mask).unsqueeze(1).type_as(src)
            # Fill padded entries with mean values.
            fill_mask = mask.unsqueeze(2).expand(-1, -1, cs)
            filled_src = torch.where(fill_mask, mean_src.expand(-1, sl, -1), src)
        else:
            mean_src = torch.mean(src, dim=1, keepdim=True).type_as(src)
            filled_src = src

        return filled_src, mean_src

    def forward(self, cls_token, src, mask):
        bs, sl, cs = src.shape

        # print(src.shape)

        mask_3d = mask[:, 1:, 1:] #(B, 76, 76) # exclude <sot>
        mask_2d = mask_3d[:, 0, :] #(B, 76)
        valid = torch.isfinite(mask_2d) #(B, 76)
        valid_counts = valid.sum(dim=1) # (B,)

        filled_src, mean_src = self._fill_with_mean(src, ~valid)
        padded_src = torch.cat([mean_src, filled_src], dim=1)

        if self._l2_normalize_for_fps:
            sampling_src = F.normalize(padded_src, dim=-1)
        else:
            sampling_src = padded_src

        # print(mask_2d[0])
        # print(sampling_src[0, :10, :2])


        sampled_inds = dgl_geo.farthest_point_sampler(
            sampling_src.to(torch.float64),
            self._num_clusters + 1,
            0).long()
        sampled_inds = sampled_inds[:, 1:] - 1

        # print(sampled_inds)

        assert((sampled_inds >= 0).all()) # Make sure sampling from the squence
        unfold_sampled_inds = sampled_inds.unsqueeze(2).expand(-1, -1, cs)

        # Apply attention layer to predict grouping
        node_features = self.centroid_fc(src, attn_mask=mask_3d)
        centroid_features = torch.gather(node_features, 1, unfold_sampled_inds)

        # Group squence of tokens into clusters
        normed_centroid_features = F.normalize(centroid_features, dim=-1)
        normed_node_features = F.normalize(node_features, dim=-1)
        logits = torch.einsum( # B, L=76, N=4
            'bij,bjk->bik', normed_node_features, normed_centroid_features.transpose(1, 2))
        logits = logits * 5
        logits = logits + mask_2d[:, :self._num_clusters].unsqueeze(1)
        assignments = torch.softmax(logits, dim=-1)
        assignments = assignments.masked_fill(~valid.unsqueeze(2), 0)

        # Average pooling within clusters.
        fc1_cls_token_src = self.fc1(torch.cat([cls_token, src], dim=1))
        fc1_cls_token, fc1_src = fc1_cls_token_src[:, :1], fc1_cls_token_src[:, 1:]
        normalizer = torch.einsum('bij,bjk->bik', assignments.transpose(1, 2),
                                  torch.ones((bs, sl, 1), dtype=src.dtype, device=src.device)).clamp(min=0.01)
        centroids = torch.einsum('bij,bjk->bik', assignments.transpose(1, 2), fc1_src)
        centroids /= normalizer

        fc2_cls_token_centroids = self.fc2(torch.cat([fc1_cls_token, centroids], dim=1))
        centroids = fc2_cls_token_centroids[:, 1:, :] + torch.gather(src, 1, unfold_sampled_inds)
        cls_token = fc2_cls_token_centroids[:, :1, :] + cls_token

        return torch.cat([cls_token, centroids], dim=1), logits, self.slice_mask(mask)

    def slice_mask(self, mask):
        return mask[:, :self._num_clusters+1, :self._num_clusters+1]

