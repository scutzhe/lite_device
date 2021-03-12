#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
# @author  : dylen
# @license :
# @contact : dylenzheng@gmail.com
# @file    : convert_to_onnx.py
# @time    : 9/18/20 3:59 PM
# @desc    : 
'''

import torch
from net.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite


def create_model(model_path):
    """
    @param model_path:
    @return:
    """
    checkpoint = torch.load(model_path)
    net = create_mobilenetv2_ssd_lite(2, is_test=True)
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
    model_path = "models/100_1.826955000559489.pth"
    onnx_path = "onnx_model/electronic_tag.onnx"
    dummy_input = torch.randn(1, 3, 300, 300).to("cuda")
    inputs, outputs = ["input"],["scores","boxes"]
    net = create_model(model_path)
    model_onnx(net,dummy_input,onnx_path,inputs,outputs)
