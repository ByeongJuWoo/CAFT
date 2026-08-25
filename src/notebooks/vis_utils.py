import os
import sys
from functools import partial

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import skimage.color as sk_color
import skimage.morphology as sk_morph
import scipy.io

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

def generate_contour(label, size=2, is_symmetric=True):
    if is_symmetric:
        label = np.pad(label, ((1, 1), (1, 1)), mode='symmetric')
    else:
        label = np.pad(label, ((1, 1), (1, 1)), mode='constant', constant_values=0)
    l_diff = label[1:-1, 1:-1] != label[1:-1, :-2]
    r_diff = label[1:-1, 1:-1] != label[1:-1, 2:]
    t_diff = label[1:-1, 1:-1] != label[:-2, 1:-1]
    b_diff = label[1:-1, 1:-1] != label[2:, 1:-1]

    edge = (l_diff + r_diff + t_diff + b_diff).astype(label.dtype)

    # dilation
    if size > 0:
        disk = sk_morph.disk(size)
        edge = edge.astype(np.int32)
        edge = sk_morph.dilation(edge, disk).astype(label.dtype)
    return edge

def label2color(label, img):
    out = sk_color.label2rgb(label, img, kind='avg', bg_label=-1)
    edge = generate_contour(label, 0).astype(bool)
    out[edge] = 1
    return out

def img2edged(label, img):
    label = label[0].cpu().data.numpy()
    label = cv2.resize(label, (448, 448), interpolation=cv2.INTER_NEAREST)
    edge = generate_contour(label, 0).astype(bool)
    out = img.transpose(1, 2, 0)
    out[edge] = 1
    return out

def label2color_wo_line(label, img):
    out = sk_color.label2rgb(label, img, kind='avg', bg_label=-1)
    return out

def label2color_wo_avg(label, img):
    # out = sk_color.label2rgb(label, img, kind='avg', bg_label=-1)
    out = img.copy()
    edge = generate_contour(label, 0).astype(bool)
    out[edge] = 1
    return out

def label2vis(img, suppixel, with_line = True, resize=448):
    suppixel = suppixel[0].cpu().data.numpy()
    suppixel = cv2.resize(suppixel, (resize, resize), interpolation=cv2.INTER_NEAREST)
    img = img.transpose(1, 2, 0)
    img = cv2.resize(img, (resize, resize), interpolation=cv2.INTER_NEAREST)
    if with_line:
        suppixel = label2color(suppixel, img)
    else: 
        suppixel = label2color_wo_line(suppixel, img)
    return suppixel

def colorize_segmentations(seg1, seg2, seg3, resize=224):
    seg1 = seg1.copy().astype(np.int32)
    seg2 = seg2.copy().astype(np.int32)
    seg3 = seg3.copy().astype(np.int32)
    
    seg1 = cv2.resize(seg1, (resize, resize), interpolation=cv2.INTER_NEAREST)
    seg2 = cv2.resize(seg2, (resize, resize), interpolation=cv2.INTER_NEAREST)
    seg3 = cv2.resize(seg3, (resize, resize), interpolation=cv2.INTER_NEAREST)

    edge1 = generate_contour(seg1, 1).astype(bool)
    edge2 = generate_contour(seg2, 1).astype(bool)
    edge3 = generate_contour(seg3, 1).astype(bool)
    
    # Reorder, such that segment indices range from 0~N within each subregion.
    for seg2_ind in np.unique(seg2):
        mask = seg2 == seg2_ind
        seg1[mask] = np.unique(seg1[mask], return_inverse=True)[-1]
    
    for seg3_ind in np.unique(seg3):
        mask = seg3 == seg3_ind
        seg2[mask] = np.unique(seg2[mask], return_inverse=True)[-1]

    # Root nodes, evenly distribute over hue.
    hue = np.linspace(0, 1, seg3.max()+2, dtype=np.float32)[:-1]
    saturation = np.linspace(0, 1, seg2.max()+5, dtype=np.float32)[3:-1][::-1]
    value = np.linspace(0, 1, seg1.max()+5, dtype=np.float32)[3:-1][::-1]
    
    ones = np.ones_like(seg1, dtype=np.float32)
    zeros = np.zeros_like(seg1, dtype=np.float32)
    hue_seg3 = hue[seg3]
    saturation_seg2 = saturation[seg2]
    value_seg1 = value[seg1]
    
    seg3_rgb = colors.hsv_to_rgb(np.stack([hue_seg3, ones, ones], axis=-1))
    seg2_rgb = colors.hsv_to_rgb(np.stack([hue_seg3, saturation_seg2, ones], axis=-1))
    seg1_rgb = colors.hsv_to_rgb(np.stack([hue_seg3, saturation_seg2, value_seg1], axis=-1))
    
    seg3_rgb = (seg3_rgb * 255).astype(np.uint8)
    seg2_rgb = (seg2_rgb * 255).astype(np.uint8)
    seg1_rgb = (seg1_rgb * 255).astype(np.uint8)
    seg3_rgb[edge3] = 255
    seg2_rgb[edge2] = 255
    seg1_rgb[edge1] = 255
    
    return seg1_rgb, seg2_rgb, seg3_rgb

def colorize_segmentations_v2(seg1, seg2, seg3, seg4, resize=224):
    seg1 = seg1.copy().astype(np.int32)
    seg2 = seg2.copy().astype(np.int32)
    seg3 = seg3.copy().astype(np.int32)
    seg4 = seg4.copy().astype(np.int32)
    
    seg1 = cv2.resize(seg1, (resize, resize), interpolation=cv2.INTER_NEAREST)
    seg2 = cv2.resize(seg2, (resize, resize), interpolation=cv2.INTER_NEAREST)
    seg3 = cv2.resize(seg3, (resize, resize), interpolation=cv2.INTER_NEAREST)
    seg4 = cv2.resize(seg4, (resize, resize), interpolation=cv2.INTER_NEAREST)

    edge1 = generate_contour(seg1, 1).astype(bool)
    edge2 = generate_contour(seg2, 1).astype(bool)
    edge3 = generate_contour(seg3, 1).astype(bool)
    edge4 = generate_contour(seg4, 1).astype(bool)
    
    # Reorder, such that segment indices range from 0~N within each subregion.
    for seg2_ind in np.unique(seg2):
        mask = seg2 == seg2_ind
        seg1[mask] = np.unique(seg1[mask], return_inverse=True)[-1]
    
    for seg3_ind in np.unique(seg3):
        mask = seg3 == seg3_ind
        seg2[mask] = np.unique(seg2[mask], return_inverse=True)[-1]

    for seg4_ind in np.unique(seg4):
        mask = seg4 == seg4_ind
        seg3[mask] = np.unique(seg3[mask], return_inverse=True)[-1]

    # Root nodes, evenly distribute over hue.
    hue = np.linspace(0, 1, seg4.max()+2, dtype=np.float32)[:-1]
    saturation = np.linspace(0, 1, seg3.max()+5, dtype=np.float32)[3:-1][::-1]
    value = np.linspace(0, 1, seg2.max()+5, dtype=np.float32)[3:-1][::-1]
    
    
    ones = np.ones_like(seg1, dtype=np.float32)
    zeros = np.zeros_like(seg1, dtype=np.float32)
    hue_seg3 = hue[seg3]
    saturation_seg2 = saturation[seg2]
    value_seg1 = value[seg1]
    
    seg4_rgb = colors.hsv_to_rgb(np.stack([hue_seg4, ones, ones], axis=-1))
    seg3_rgb = colors.hsv_to_rgb(np.stack([hue_seg3, ones, ones], axis=-1))
    seg2_rgb = colors.hsv_to_rgb(np.stack([hue_seg3, saturation_seg2, ones], axis=-1))
    seg1_rgb = colors.hsv_to_rgb(np.stack([hue_seg3, saturation_seg2, value_seg1], axis=-1))
    seg4_rgb = (seg4_rgb * 255).astype(np.uint8)
    seg3_rgb = (seg3_rgb * 255).astype(np.uint8)
    seg2_rgb = (seg2_rgb * 255).astype(np.uint8)
    seg1_rgb = (seg1_rgb * 255).astype(np.uint8)
    seg4_rgb[edge4] = 255
    seg3_rgb[edge3] = 255
    seg2_rgb[edge2] = 255
    seg1_rgb[edge1] = 255
    
    return seg1_rgb, seg2_rgb, seg3_rgb