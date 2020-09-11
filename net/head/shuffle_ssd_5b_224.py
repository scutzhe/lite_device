# -*- coding: utf-8 -*-
"""
@author  : 秦丹峰
@contact : daneven.jim@gmail.com
@time    : 2020/8/4 23:17
@file    : ssd.py
@desc    : 
"""
import sys
sys.path.append("../..")
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from typing import List, Tuple
from collections import namedtuple
from utils import box_utils

class SSD(nn.Module):
    def __init__(self, num_classes: int, base_net: nn.ModuleList,
                 extras: nn.ModuleList, classification_headers: nn.ModuleList,
                 regression_headers: nn.ModuleList, is_test=False, config=None, device=None):
        """Compose a SSD model using the given components.
        """
        super(SSD, self).__init__()

        self.num_classes = num_classes
        self.base_net = base_net
        self.extras = extras
        self.classification_headers = classification_headers
        self.regression_headers = regression_headers
        self.is_test = is_test
        self.config = config

        if device:
            self.device = device
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if is_test:
            self.config = config
            self.priors = config.priors.to(self.device)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        confidences = []
        locations = []
        header_index = 0
        # for layer_index in range(len(self.base_net)):
        #     x = self.base_net[layer_index](x)
        s3_x,c5_x,fc_x = self.base_net(x)
        # for x in [s3_x,c5_x]:
        for x in [c5_x]:
            print('输入至检测头的特征图的大小：', x.shape)
            confidence, location = self.compute_header(header_index, x)
            header_index += 1
            print('第', header_index, '次compute_header后，confidences的维度：', confidence.shape)
            confidences.append(confidence)
            locations.append(location)
        for layer in self.extras:
            x = layer(x)
            print('输入至检测头的特征图的大小：', x.shape)
            confidence, location = self.compute_header(header_index, x)
            header_index += 1
            print('第', header_index, '次compute_header后，confidences的维度：', confidence.shape)
            confidences.append(confidence)
            locations.append(location)

        confidences = torch.cat(confidences, 1)
        locations = torch.cat(locations, 1)

        if self.is_test:
            confidences = F.softmax(confidences, dim=2)
            boxes = box_utils.convert_locations_to_boxes(
                locations, self.priors, self.config.center_variance, self.config.size_variance
            )
            boxes = box_utils.center_form_to_corner_form(boxes)
            return confidences, boxes
        else:
            return confidences, locations

    def compute_header(self, i, x):
        confidence = self.classification_headers[i](x)
        confidence = confidence.permute(0, 2, 3, 1).contiguous()
        confidence = confidence.view(confidence.size(0), -1, self.num_classes)

        location = self.regression_headers[i](x)
        location = location.permute(0, 2, 3, 1).contiguous()
        location = location.view(location.size(0), -1, 4)

        return confidence, location

    def init_from_base_net(self, model):
        self.base_net.load_state_dict(torch.load(model, map_location=lambda storage, loc: storage), strict=True)
        # self.source_layer_add_ons.apply(_xavier_init_)
        self.extras.apply(_xavier_init_)
        self.classification_headers.apply(_xavier_init_)
        self.regression_headers.apply(_xavier_init_)

    def init_from_pretrained_ssd(self, model):
        state_dict = torch.load(model, map_location=lambda storage, loc: storage)
        model_dict = self.state_dict()
        model_dict.update(state_dict)
        self.load_state_dict(model_dict)

    def fine_tune(self, model):
        state_dict = torch.load(model, map_location=lambda storage, loc: storage)
        state_dict = {k: v for k, v in state_dict.items() if
                      not (k.startswith("classification_headers") or k.startswith("regression_headers"))}
        model_dict = self.state_dict()
        model_dict.update(state_dict)
        self.load_state_dict(model_dict)
        self.classification_headers.apply(_xavier_init_)
        self.regression_headers.apply(_xavier_init_)

    def init(self):
        self.base_net.apply(_xavier_init_)
        self.extras.apply(_xavier_init_)
        self.classification_headers.apply(_xavier_init_)
        self.regression_headers.apply(_xavier_init_)

    def load(self, model):
        self.load_state_dict(torch.load(model, map_location=lambda storage, loc: storage))

    def load_model(self, model):
        self.load_state_dict(torch.load(model, map_location=lambda storage, loc: storage)['model'])

    def load_weight(self, state_dict):
        self.load_state_dict(state_dict)

    def save(self, model_path):
        torch.save(self.state_dict(), model_path)


def _xavier_init_(m: nn.Module):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
