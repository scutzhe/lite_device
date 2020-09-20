#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
# @author  : 郑祥忠
# @license : (C) Copyright,2016-2020,广州海格星航科技
# @contact : dylenzheng@gmail.com
# @file    : convert_to_onnx.py
# @time    : 9/11/20 3:39 PM
# @desc    : 
'''

import torch
from net.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite

def create_net(model_path):
    """
    @param model_path:
    @return:
    """
    checkpoint = torch.load(model_path)
    net = create_mobilenetv2_ssd_lite(1, width_mult=1.0)
    net.load_state_dict(checkpoint)
    net.eval()
    net.to("cuda")
    return net

def model_onnx(net,dummy_input,onnx_path,inputs,outputs):
    """
    @param net:
    @param dummy_input:
    @param onnx_path:
    @param inputs:
    @param outputs:
    @return:
    """
    torch.onnx._export(net, dummy_input, onnx_path,
                      export_params=True, verbose=False, input_names=inputs,output_names=outputs)


if __name__ == "__main__":
    model_path = ""
    model_name = ""
    onnx_path = ""
    dummy_input = torch.randn(1,3,300,300).to("cuda")
    inputs, outputs = ["input"], ["output"]
    net = create_net(model_path)
    model_onnx(net, dummy_input, onnx_path, inputs, outputs)