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
from PIL import Image



class SubtractMeansStd(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        image = (image - self.mean) / self.std
        return image

def inference(model_path,img_path):
    """
    @param model_path:
    @param img_path:
    @return:
    """
    image_mean = np.array([123.675, 116.28, 103.53])
    image_std = np.array([0.229, 0.224, 0.225])
    # ori_image = cv2.imread(img_path)
    ori_image = Image.open(img_path)
    # resize_image = cv2.resize(ori_image, (224, 224))
    resize_image = ori_image.resize((224,224),Image.BICUBIC)
    # resize_image = cv2.resize(ori_image, (64, 64))
    # input_image = cv2.cvtColor(resize_image, cv2.COLOR_BGR2RGB)
    # image = (input_image - image_mean) * image_std
    image = np.array(resize_image)
    image = (image - image_mean) * image_std

    # 模型加载
    interpreter = MNN.Interpreter(model_path)
    session = interpreter.createSession()
    input_tensor = interpreter.getSessionInput(session)

    # 输入tensor
    caffe_input = MNN.Tensor((1, 3, 224, 224), MNN.Halide_Type_Float,\
                    image, MNN.Tensor_DimensionType_Caffe)
    tensorflow_input = MNN.Tensor((1, 224, 224, 3), MNN.Halide_Type_Float, \
                           image, MNN.Tensor_DimensionType_Tensorflow)


    # input_tensor.copyFrom(tensorflow_input)
    input_tensor.copyFrom(caffe_input)

    # 执行推断
    ctime = time.time()
    checkInput = np.array(input_tensor.getData())
    interpreter.runSession(session)
    print("inference time:", time.time()-ctime)

    # 获得推断结果
    output = interpreter.getSessionOutputAll(session)
    recognition = output["output0"].getData()
    # recognition = output["classifier"].getData()
    # print("output=",len(recognition))
    return recognition

def cal_distance_nonormal(tuple1,tuple2):
    """
    @param tuple1:
    @param tuple2:
    @return:
    """
    import math
    middle1 = 0
    middle2_1 = 0
    middle2_2 = 0
    for i in range(len(tuple1)):
        middle1 += tuple1[i] * tuple2[i]
        middle2_1 += math.pow(tuple1[i],2)
        middle2_2 += math.pow(tuple2[i],2)
    distance = middle1 / (math.sqrt(middle2_1) * math.sqrt(middle2_2))
    return distance


if __name__ == "__main__":
    model_path = "/home/zhex/share_folder/mobilenetv3_large_100.mnn"
    image_path1 = "/home/zhex/git_me/DeviceRetrieval/resource/1-1.jpg"
    image_path2 = "/home/zhex/git_me/DeviceRetrieval/resource/2-2.jpg"
    feature1 = inference(model_path,image_path1)
    feature2 = inference(model_path,image_path2)
    distance = cal_distance_nonormal(feature1,feature2)
    print("distance=",distance)
