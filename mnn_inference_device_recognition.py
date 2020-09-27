#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
# @author  : 郑祥忠
# @license : (C) Copyright,2016-2020,广州海格星航科技
# @contact : dylenzheng@gmail.com
# @file    : mnn_inference_device_recognition.py
# @time    : 9/25/20 1:45 PM
# @desc    : 
'''

import os
import cv2
import time
import torch
import torchvision
import numpy as np
import MNN.expr as F
from PIL import Image
from tqdm import tqdm
from torchvision import transforms as T

def inference(mnn_model_path,image_path):
    """
    @param mnn_model_path:
    @param image_path:
    @return:
    """
    # mean std
    image_mean = np.array([123.675, 116.28, 103.53])
    image_std = np.array([0.229, 0.224, 0.225])

    # 加载mnn
    vars = F.load_as_dict(mnn_model_path)
    inputVar = vars["input0"]
    # 查看输入信息
    print('input shape: ', inputVar.shape)

    # input_image = Image.open(image_path)
    input_image = cv2.imread(image_path)
    input_image = cv2.resize(input_image,(224,224),interpolation=cv2.INTER_CUBIC)
    input_image = cv2.cvtColor(input_image,cv2.COLOR_BGR2RGB)
    input_image = (input_image - image_mean) * image_std
    # transform = T.Compose([
    #     T.Resize((300,300)),
    #     T.ToTensor(),
    #     T.Normalize(image_mean, image_std),
    # ])
    transform = T.Compose([
        # T.Resize((224, 224)),
        SubtractMeansStd(image_mean, image_std),
        T.ToTensor(),
    ]
    )
    input_tensor = transform(input_image)
    print("before input_tensor.size()=",input_tensor.size())
    input_tensor = torch.unsqueeze(input_tensor,0)
    print("after input_tensor.size()=", input_tensor.size())
    inputVar.write(input_tensor.tolist())

    # 查看输出结果
    outputVar = vars["output0"]
    # print('output shape: ', outputVar.shape)
    output = outputVar.read()

    print('完成一张图片的推断')

    return output

class SubtractMeansStd(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        image = (image - self.mean) / self.std
        return image


if __name__ == '__main__':
    mnn_model_path = "/home/zhex/data/arm_device_voc/tflite/device_recognition.mnn"
    image_one_path = "/home/zhex/git_me/DeviceRetrieval/resource/1-1.jpg"
    image_two_path = "/home/zhex/git_me/DeviceRetrieval/resource/1-2.jpg"
    output1 = inference(mnn_model_path,image_one_path)
    output2 = inference(mnn_model_path,image_two_path)
    print("outpu1.shape=",output1.shape)
    print("outpu2.shape=",output2.shape)

