# -*- coding: utf-8 -*-
"""
@author  : 詹金豪
@contact : zhanjinhao5@outlook.com
@time    : 2020/8/27 10:08
@file    : shufflenet_v2_ssdlite.py
@desc    :
"""

import sys

sys.path.append("..")
import torch
from torch.nn import Conv2d, Sequential, ModuleList, BatchNorm2d
from torch import nn

from net.head.shuffle_ssd import SSD
from net.predictor import Predictor_w_h
from net.config import shufflenet_ssd_w_h_config as config
from net.backbone.shuffleNet_v2 import shufflenet_v2_x0_5, InvertedResidual,SSD_InvertedResidual

import time

def SeperableConv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, onnx_compatible=False):
    """Replace Conv2d with a depthwise Conv2d and Pointwise Conv2d.
    """
    ReLU = nn.ReLU if onnx_compatible else nn.ReLU6
    return Sequential(
        Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
               groups=in_channels, stride=stride, padding=padding),
        BatchNorm2d(in_channels),
        ReLU(),
        Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
    )

def create_shufflenetv2_ssd_lite(num_classes,is_test=False):
    base_net = shufflenet_v2_x0_5(num_classes = num_classes)

    extras = ModuleList([
        SSD_InvertedResidual(1024, 512, stride=2, expand_ratio=0.2),
        SSD_InvertedResidual(512, 256, stride=2, expand_ratio=0.25),
        SSD_InvertedResidual(256, 256, stride=2, expand_ratio=0.5),
        SSD_InvertedResidual(256, 64, stride=2, expand_ratio=0.25)
    ])

    regression_headers = ModuleList([
        SeperableConv2d(in_channels=96, out_channels=2 * 4,kernel_size=3, padding=1, onnx_compatible=False),
        SeperableConv2d(in_channels=1024, out_channels=2 * 4, kernel_size=3, padding=1, onnx_compatible=False),
        SeperableConv2d(in_channels=512, out_channels=2 * 4, kernel_size=3, padding=1, onnx_compatible=False),
        SeperableConv2d(in_channels=256, out_channels=2 * 4, kernel_size=3, padding=1, onnx_compatible=False),
        SeperableConv2d(in_channels=256, out_channels=2 * 4, kernel_size=3, padding=1, onnx_compatible=False),
        Conv2d(in_channels=64, out_channels=2 * 4, kernel_size=1),
    ])

    classification_headers = ModuleList([
        SeperableConv2d(in_channels=96, out_channels=2 * num_classes, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=1024, out_channels=2 * num_classes, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=512, out_channels=2 * num_classes, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=256, out_channels=2 * num_classes, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=256, out_channels=2 * num_classes, kernel_size=3, padding=1),
        Conv2d(in_channels=64, out_channels=2 * num_classes, kernel_size=1),
    ])

    return SSD(num_classes, base_net,
               extras, classification_headers, regression_headers, is_test=is_test, config=config)


def create_shufflenetv2_ssd_lite_predictor(net, candidate_size=200, nms_method=None, sigma=0.5,
                                          device=torch.device('cpu')):
    predictor = Predictor_w_h(net, config.image_width,config.image_height, config.image_mean,
                          config.image_std,
                          nms_method=nms_method,
                          iou_threshold=config.iou_threshold,
                          candidate_size=candidate_size,
                          sigma=sigma,
                          device=device)
    return predictor
if __name__ == '__main__':
    # 预训练模型：'/home/zhanjinhao/models/shufflenetv2_x0.5-f707e7126e.pth'
    import cv2
    net = create_shufflenetv2_ssd_lite(1)
    net.eval()
    print('已创建网络')
    # print(net)
    torch.save({'net':net.state_dict()},'shufflenet_v2_ssdlite_w_h.pth')
    predictor = create_shufflenetv2_ssd_lite_predictor(net,device=torch.device('cuda'))
    image = cv2.imread('/home/zhanjinhao/codes/车辆检测/ssdlite_mobilenet_v2/assets/image-20200808162742174.png')
    st = time.time()
    boxes, labels, probs = predictor.predict(image)
    et = time.time()-st
    print('推断一张图片总用时:',et)
    # print('boxes', boxes)
    # print('lables', labels)
    # print('probs', probs)

