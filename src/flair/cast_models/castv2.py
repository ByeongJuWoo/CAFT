# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Mostly copy-paste from timm library.
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""
import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os
# sfcn_path = os.path.abspath('./')
# sys.path.append(sfcn_path)
from sfcn.models import SFCN
from sfcn.train_util import update_spixl_map, shift9pos, poolfeat
from .utils import segment_mean_nd, segment_mean_max_nd
from .cast_modules import Pooling
from .graph_pool import GraphPoolingReg
import numpy as np
from timm.models.layers import to_2tuple, trunc_normal_

from .basic_modules import Block, PatchEmbed, ConvStemV1, DropPath

class CAST(nn.Module):
    def __init__(self, img_size=[224], patch_size=8, in_chans=3, embed_dim=768,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, 
                 embed_layer=ConvStemV1, 
                 depth=[3, 3, 3, 3],
                 num_clusters = [64, 32, 16],
                 global_pool = 'cls',
                 reg_tokens=0,
                 output_dim=512,
                 fill_dead_tokens=False,
                 text_con=True,
                 hierarchy_flair=False,
                 **kwargs):
        super().__init__()
        self.num_features = self.embed_dim = self.width = embed_dim
        self.global_pool = global_pool
        self.num_clusters = num_clusters

        self.patch_embed = embed_layer(
            img_size=img_size[0], patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.build_2d_sincos_position_embedding()
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.fill_dead_tokens = fill_dead_tokens
        self.text_con = text_con
        self.hierarchy_flair = hierarchy_flair
        self.num_prefix_tokens = reg_tokens + 1
        self.reg_tokens = nn.Parameter(torch.zeros(1, reg_tokens, embed_dim)) if reg_tokens > 0 else None

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[ # from ModuleList to Sequential
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(sum(depth))])
        self.nn_blocks = []
        start = 0
        for d in depth:
            end = start + d
            self.nn_blocks.append(nn.Sequential(self.blocks[start:end]))
            start = end
        self.depth=depth

        self.pools = nn.ModuleList()
        for i, num_cluster in enumerate(num_clusters):
            pool = Pooling(
                    pool_block=GraphPoolingReg(
                        layerscale=False,
                        num_clusters=num_cluster,
                        d_model=embed_dim,
                        l2_normalize_for_fps=False
                    )
            )
            self.pools.append(pool)
        # self.ln_pre = norm_layer(embed_dim)
        if self.hierarchy_flair:
            assert self.text_con
            self.norm = norm_layer(embed_dim)
            self.norm_fine = norm_layer(embed_dim)
        elif self.text_con:
            self.norm = norm_layer(embed_dim)
        else: 
            self.norm = norm_layer(embed_dim)
            scale = self.embed_dim ** -0.5
            self.proj = nn.Parameter(scale * torch.randn(self.embed_dim, output_dim))

        # trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

        self.n_spix = (img_size[0]// 16)**2

        # TODO: REPLACE below PATH if needed
        checkpoint = torch.load('./sfcn/pretrain_ckpt/SpixelNet_bsd_ckpt.tar')
        self.superpixel_generator = SFCN(n_spix = self.n_spix, data=checkpoint)
        for param in self.superpixel_generator.parameters():
            param.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def build_2d_sincos_position_embedding(self, temperature=10000.):
        h, w = self.patch_embed.grid_size
        grid_w = torch.arange(w, dtype=torch.float32)
        grid_h = torch.arange(h, dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h)
        assert self.embed_dim % 4 == 0, 'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = self.embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature**omega)
        out_w = torch.einsum('m,d->md', [grid_w.flatten(), omega])
        out_h = torch.einsum('m,d->md', [grid_h.flatten(), omega])
        pos_emb = torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], dim=1)[None, :, :]

        pe_token = torch.zeros([1, 1, self.embed_dim], dtype=torch.float32)
        self.pos_embed = nn.Parameter(pos_emb)
        self.pos_embed.requires_grad = False
        self.pe_token = nn.Parameter(pe_token)
        self.pe_token.requires_grad = False

    def prepare_tokens(self, x):
        with torch.no_grad():
            y_ = self.superpixel_generator(x)
        x = self.patch_embed(x)
        B, H, W, C = x.shape
        with torch.no_grad():
            H_ori, W_ori = H * self.patch_embed.patch_size[0], W * self.patch_embed.patch_size[1]
            n_spix = (H_ori // 16) *  (W_ori // 16)
            y = self.superpixel_generator._get_hard_label(y_, H, W, n_spix=n_spix)
            y_ori = self.superpixel_generator._get_hard_label(y_, H_ori, W_ori, n_spix=n_spix)

        # Create padding mask
        ones = torch.ones((B, H, W, 1), dtype=x.dtype, device=x.device)
        avg_ones = segment_mean_nd(ones, y).squeeze(-1)
        pad_mask = avg_ones <= 0.5

        # Add positional encodings
        pos_embed = self.pos_embed.view(1,
                                   self.patch_embed.grid_size[0],
                                   self.patch_embed.grid_size[1],
                                   self.embed_dim)
        pos_embed = F.interpolate(pos_embed.permute(0, 3, 1, 2),
                                  size=(H, W), mode='bicubic', align_corners=False)
        pos_embed = pos_embed.permute(0, 2, 3, 1).contiguous()
        pos_embed = pos_embed.expand(B, -1, -1, -1)
        x = x + pos_embed

        # Add positional encodings
        x = segment_mean_nd(x, y)
        if self.fill_dead_tokens:
            device, N = y_ori.device, n_spix
            labels_flat = y_ori.reshape(B, -1)
            y_coord = torch.arange(H_ori, device=device).view(1, H_ori, 1).expand(B, H_ori, W_ori).reshape(B, -1).float()
            x_coord = torch.arange(H_ori, device=device).view(1, 1, W_ori).expand(B, H_ori, W_ori).reshape(B, -1).float()
            sums_y = torch.zeros(B, N, device=device)
            sums_x = torch.zeros(B, N, device=device)
            count_orig = torch.zeros(B, N, device=device)
            sums_y.scatter_add_(1, labels_flat, y_coord)
            sums_x.scatter_add_(1, labels_flat, x_coord)
            count_orig.scatter_add_(1, labels_flat, torch.ones_like(labels_flat, dtype=torch.float))
            cent_y = sums_y / count_orig.clamp(min=1.0) #(B, N)
            cent_x = sums_x / count_orig.clamp(min=1.0) #(B, N)
            coords = torch.stack((cent_y, cent_x), dim=2) #(B, N, 2)
            diff = coords.unsqueeze(2) - coords.unsqueeze(1) # (B, N, N 2)
            dist2 = (diff*diff).sum(dim=-1)
            dist2 = dist2.masked_fill(pad_mask.unsqueeze(1), float('inf'))
            nearest = torch.argmin(dist2, dim=2) #(B, N)
            nearest_exp = nearest.unsqueeze(-1).expand(-1, -1, C)
            x = x.gather(1, nearest_exp)

        x = self.pos_drop(x)

        # Add class token
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        cls_token = cls_token + self.pe_token.expand(x.shape[0], -1, -1)
        if self.reg_tokens is not None:
            reg_tokens = self.reg_tokens.expand(x.shape[0], -1, -1)
            cls_token = torch.cat([cls_token, reg_tokens], dim=1)

        # layernorm before feeding vit
        if hasattr(self, 'ln_pre'):
            x = self.ln_pre(x)
            cls_token = self.ln_pre(cls_token)

        return x, cls_token, pad_mask

    def _global_pool(self, x):
        if self.global_pool == 'cls':
            return x[:, 0, :]
        elif self.global_pool == 'max':
            x = x[:, 1:, :]
            return torch.max(x, dim=1)[0]
        elif self.global_pool == 'gap':
            x = x[:, 1:, :]
            return torch.mean(x, dim=1)
        elif self.global_pool == 'topk':
            x = x[:, 1:, :].topk(5, dim=1)[0]
            return torch.mean(x, dim=1)
        elif self.global_pool == 'topk3':
            x = x[:, 1:, :].topk(3, dim=1)[0]
            return torch.mean(x, dim=1)

    def forward_features(self, x, return_intermediates=False):
        x, cls_token, pad_mask = self.prepare_tokens(x)
        intermediates = {}
        for i in range(len(self.pools)):
            intermediates.update({f'padding_mask{i+1}':pad_mask})
            cls_x = torch.cat([cls_token, x], dim=1)
            cls_x_ = (self.nn_blocks[i])(cls_x).type_as(x)
            cls_token, x = cls_x_[:, :self.num_prefix_tokens, :], cls_x_[:, self.num_prefix_tokens:, :]
            cls_token, logit, x, pad_mask, sampled_inds = (self.pools[i])(cls_token, x, pad_mask)
            intermediate = {
                f'logit{i+1}': logit, f'centroid{i+1}': x, f'block{i+1}': cls_x_, 
                f'sampled_inds{i+1}': sampled_inds, f'cls_token{i+1}': cls_token,
            }; intermediates.update(intermediate)
        cls_x = torch.cat([cls_token, x], dim=1)
        cls_x = self.nn_blocks[-1](cls_x).type_as(x)
        cls_x = self.norm(cls_x)
        intermediates.update({f'out_cls_x':cls_x})

        if return_intermediates:
            return cls_x, intermediates
        return cls_x

    def forward_head(self, x):
        if self.text_con:
            return x[:, 0], x[:, 1:]
        if self.global_pool == 'topk' or self.global_pool == 'topk3':
            x = x @ self.proj
            x = self._global_pool(x)
        else:
            x = self._global_pool(x)
            x = x @ self.proj
        return x

    def forward(self, x, return_intermediates=False):
        x, intermediates = self.forward_features(x, return_intermediates=True)
        if self.hierarchy_flair:
            # image_fine = intermediates[f'block{len(self.pools)}']
            image_fine = intermediates['block2']
            image_fine = self.norm_fine(image_fine)
            image_fine_local, image_fine_global = image_fine[:, 1:], image_fine[:, 0]
            image_coarse_local, image_coarse_global = x[:, 1:], x[:, 0]
            if return_intermediates:
                return image_fine_local, image_fine_global, image_coarse_local, image_coarse_global, intermediates
            return image_fine_local, image_fine_global, image_coarse_local, image_coarse_global
        else:
            if return_intermediates:
                return self.forward_head(x), intermediates
            else:
                return self.forward_head(x)

    def forward_global_local(self, x):
        x = self.forward_features(x)
        local_image_features = x[:, 1:, :] @ self.proj
        if self.global_pool == 'topk':
            x = local_image_features.topk(5, dim=1)[0]
            global_image_features = torch.mean(x, dim=1)
        elif self.global_pool == 'topk3':
            x = local_image_features.topk(3, dim=1)[0]
            global_image_features = torch.mean(x, dim=1)
        else:
            global_image_features = self.forward_head(x)
        return global_image_features, local_image_features # (B, 1, C), (B, L, C)


    def get_last_selfattention(self, x):
        x, cls_token, pad_mask = self.prepare_tokens(x)
        for i in range(len(self.pools)):
            cls_x = torch.cat([cls_token, x], dim=1)
            cls_x_ = (self.nn_blocks[i])(cls_x).type_as(x)
            cls_token, x = cls_x_[:, :self.num_prefix_tokens, :], cls_x_[:, self.num_prefix_tokens:, :]
            cls_token, logit, x, pad_mask, sampled_inds = (self.pools[i])(cls_token, x, pad_mask)
        cls_x = torch.cat([cls_token, x], dim=1)
        for i, blk in enumerate(self.nn_blocks[-1][0]):
            if i < self.depth[-1] - 1:
                cls_x = blk(cls_x)
            else:
                cls_x, attn = blk(cls_x, return_attention=True)
                return attn

    def get_every_selfattention(self, x):
        x, cls_token, pad_mask = self.prepare_tokens(x)
        attns = []
        for i in range(len(self.pools)):
            cls_x = torch.cat([cls_token, x], dim=1)
            # cls_x_ = (self.nn_blocks[i])(cls_x).type_as(x)
            for j, blk in enumerate(self.nn_blocks[i][0]):
                if j < self.depth[j] - 1:
                    cls_x = blk(cls_x)
                else:
                    cls_x, attn = blk(cls_x, return_attention=True)
                    attns.append(attn)
            cls_x_ = cls_x.type_as(x)
            cls_token, x = cls_x_[:, :self.num_prefix_tokens, :], cls_x_[:, self.num_prefix_tokens:, :]
            cls_token, logit, x, pad_mask, sampled_inds = (self.pools[i])(cls_token, x, pad_mask)
        cls_x = torch.cat([cls_token, x], dim=1)
        for i, blk in enumerate(self.nn_blocks[-1][0]):
            if i < self.depth[-1] - 1:
                cls_x = blk(cls_x)
            else:
                cls_x, attn = blk(cls_x, return_attention=True)
                attns.append(attn)
        return attns

    def get_intermediate_layers(self, x):
        x, cls_token, pad_mask = self.prepare_tokens(x)
        # we return the output tokens from the `n` last blocks
        out = {}
        cnt = 0
        for i in range(len(self.pools)):
            cls_x = torch.cat([cls_token, x], dim=1)
            for block in self.nn_blocks[i][0]:
                cls_x = block(cls_x).type_as(x)
                out[f'b{cnt}'] = cls_x[:, 1:, :]
                cnt = cnt + 1
            cls_token, x_ = cls_x[:, :1, :], cls_x[:, 1:, :]
            cls_token, x, logit, sampled_inds, out_pool = self.pools[i].pool_block.get_intermediate_layers(
                cls_token=cls_token, src=x_, mask=pad_mask, num_clusters=None, index=cnt)
            pad_mask = torch.zeros((logit.shape[0], logit.shape[-1]), dtype=torch.bool, device=logit.device)
            out.update(out_pool)
        cls_x = torch.cat([cls_token, x], dim=1)
        for block in self.nn_blocks[-1][0]:
            cls_x = block(cls_x).type_as(x)
            out[f'b{cnt}'] = cls_x[:, 1:, :]
            cnt = cnt + 1
        cls_x = self.norm(cls_x)
        return out

    def get_superpixel_label(self, x, n_spix=None):
        n_spix = n_spix or self.n_spix
        with torch.no_grad():
            y = self.superpixel_generator.get_hard_label(x, n_spix=n_spix)
        return y


def cast_base(patch_size=8, **kwargs):
    model = CAST(
        patch_size=patch_size, embed_dim=768, depth=[3, 3, 3, 3], num_clusters = [64, 32, 16],
        embed_layer=ConvStemV1, global_pool = 'cls',
        num_heads=12, mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),  **kwargs)
    return model

