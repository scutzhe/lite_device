#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
# @author  : 郑祥忠
# @license : (C) Copyright,2016-2020,广州海格星航科技
# @contact : dylenzheng@gmail.com
# @file    : tflite_inference_device_recognition.py
# @time    : 9/27/20 11:44 AM
# @desc    : 
'''

from __future__ import print_function
import numpy as np
import MNN
import cv2
import time



class SubtractMeansStd(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        image = (image - self.mean) / self.std
        return image

def inference_test(model_path,img_path):
    """
    @param model_path:
    @param img_path:
    @return:
    """
    image_mean = np.array([123.675, 116.28, 103.53])
    image_std = np.array([0.229, 0.224, 0.225])
    ori_image = cv2.imread(img_path)
    resize_image = cv2.resize(ori_image, (224, 224))
    input_image = cv2.cvtColor(resize_image, cv2.COLOR_BGR2RGB)
    image = (input_image - image_mean) * image_std


    # 模型加载
    interpreter = MNN.Interpreter(model_path)
    session = interpreter.createSession()
    input_tensor = interpreter.getSessionInput(session)

    # 输入tensor
    caffe_input = MNN.Tensor((1, 3, 224, 224), MNN.Halide_Type_Float,\
                    image, MNN.Tensor_DimensionType_Caffe)
    tensorflow_input = MNN.Tensor((1, 224, 224, 3), MNN.Halide_Type_Float, \
                           image, MNN.Tensor_DimensionType_Tensorflow)

    input_tensor.copyFrom(tensorflow_input)

    # 执行推断
    ctime = time.time()
    checkInput = np.array(input_tensor.getData())
    interpreter.runSession(session)
    print("inference time:", time.time()-ctime)

    # 获得推断结果
    output = interpreter.getSessionOutputAll(session)
    recognition = output["output0"].getData()
    print("output=",len(recognition))


if __name__ == "__main__":
    model_path = "/home/zhex/data/arm_device_voc/tflite/device_recognition.mnn"
    image_path = "/home/zhex/git_me/DeviceRetrieval/resource/1-1.jpg"
    inference_test(model_path,image_path)
