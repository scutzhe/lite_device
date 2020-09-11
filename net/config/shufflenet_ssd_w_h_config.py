# -*- coding: utf-8 -*-
"""
@author  : 秦丹峰
@contact : daneven.jim@gmail.com
@time    : 2020/8/4 23:38
@file    : ssd_config.py
@desc    : 
"""
import sys
sys.path.append("../..")
import numpy as np
from utils.box_utils import SSDSpec_w_h, SSDBoxSizes, generate_ssd_priors_w_h


image_width = 384
image_height = 192
image_mean = np.array([127, 127, 127])  # RGB layout
image_std = 128.0
iou_threshold = 0.45
center_variance = 0.1
size_variance = 0.2

specs = [
    SSDSpec_w_h(24,12, 20, SSDBoxSizes(76, 134),[]),
    SSDSpec_w_h(12,6, 40, SSDBoxSizes(134, 192), []),
    SSDSpec_w_h(6,3, 81, SSDBoxSizes(192, 250), []),
    SSDSpec_w_h(3,2, 128, SSDBoxSizes(250, 307), []),
    SSDSpec_w_h(2,1, 192, SSDBoxSizes(307, 364), []),
    SSDSpec_w_h(1,1, 384, SSDBoxSizes(364, 422), [])
]

# 取宽高中较小者计算anchors
priors = generate_ssd_priors_w_h(specs, image_width,image_height)