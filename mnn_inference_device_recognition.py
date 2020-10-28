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
from scipy.spatial.distance import cosine, euclidean


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
    # inputVar = vars["input"]
    # 查看输入信息
    print('input shape: ', inputVar.shape)

    input_image = Image.open(image_path)
    # input_image = cv2.imread(image_path)
    # input_image = cv2.resize(input_image,(224,224),interpolation=cv2.INTER_CUBIC)
    # input_image = cv2.resize(input_image,(64,64),interpolation=cv2.INTER_CUBIC)
    # input_image = cv2.cvtColor(input_image,cv2.COLOR_BGR2RGB)
    # input_image = (input_image - image_mean) * image_std
    # transform = T.Compose([
    #     # T.Resize((224,224)),
    #     T.Resize((160,160)),
    #     T.ToTensor(),
    #     # T.Normalize(image_mean, image_std),
    # ])
    transform = T.Compose([
        T.Resize((224, 224)),
        SubtractMeansStd(image_mean, image_std),
        T.ToTensor(),
    ])
    input_tensor = transform(input_image)
    # print("before input_tensor.size()=",input_tensor.size())
    # input_tensor = torch.unsqueeze(input_tensor,0)
    # print("after input_tensor.size()=", input_tensor.size())
    # img_bgr = cv2.imread(image_path)
    # img_rgb_tensor1 = cv2.dnn.blobFromImage(img_bgr, size=(224,224), swapRB=True, mean=tuple(255.0 * m for m in (0.485, 0.456, 0.406)),
    #                                         scalefactor=1 / (0.225 * 255.0))

    inputVar.write(input_tensor.tolist())
    # inputVar.write(img_rgb_tensor1.tolist())

    # 查看输出结果
    outputVar = vars["output0"]
    # outputVar = vars["classifier"]
    # print('output shape: ', outputVar.shape)
    output = outputVar.read()

    # print('完成一张图片的推断')

    return output

class SubtractMeansStd(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        image = (image - self.mean) / self.std
        return image

def cal_distance(tuple1,tuple2):
    """
    @param tuple1:
    @param tuple2:
    @return:
    """
    import math
    import numpy as np
    middle1 = 0
    middle2_1 = 0
    middle2_2 = 0
    np1 = np.array(tuple1)
    np2 = np.array(tuple2)
    avg1 = np.mean(np1)
    std1 = np.var(np1)
    avg2 = np.mean(np2)
    std2 = np.var(np2)
    for i in range(len(tuple1)):
        middle1 += (((tuple1[i] - avg1) / std1)  * ((tuple2[i] - avg2) / std2))
        middle2_1 += math.pow((tuple1[i] - avg1) / std1 ,2)
        middle2_2 += math.pow((tuple2[i] - avg2) / std2,2)
    distance = middle1 / (math.sqrt(middle2_1) * math.sqrt(middle2_2))
    return distance

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



if __name__ == '__main__':
    mnn_model_path = "/home/zhex/share_folder/device_recognition_224.mnn"
    # mnn_model_path = "/home/zhex/share_folder/mobilenetv3_large_100.mnn"
    # mnn_model_path = "/home/zhex/share_folder/sigarette_Y.mnn"
    image_one_path = "/home/zhex/git_me/DeviceRetrieval/resource/1-1.jpg"
    image_two_path = "/home/zhex/git_me/DeviceRetrieval/resource/1-2.jpg"
    output1 = inference(mnn_model_path,image_one_path)
    output2 = inference(mnn_model_path,image_two_path)
    # print("output1[0]:",output1[0])
    # print("output1[999]:",output1[999])
    print(len(output1))
    output1 = output1 / np.linalg.norm(output1)
    output2 = output2 / np.linalg.norm(output2)

    distance = euclidean(output1, output2)
    # print("outpu1=",output1)
    # print("outpu2=",output2)
    # distance = cal_distance(output1,output2)
    # distance = cal_distance_nonormal(output1,output2)
    print("distance=",distance)

# if __name__ == "__main__":
#     mnn_model_path = "/home/zhex/share_folder/mobilenetv3_large_100.mnn"
#     image_dir = "/home/zhex/git_me/DeviceRetrieval/resource/lib"
#     names = os.listdir(image_dir)
#     for i in range(len(names)):
#         for j in range(i+1,len(names)):
#             image_one_path = os.path.join(image_dir,names[i])
#             image_two_path = os.path.join(image_dir,names[j])
#             output1 = inference(mnn_model_path, image_one_path)
#             output2 = inference(mnn_model_path,image_two_path)
#             distance = cal_distance(output1, output2)
#             print("{}_{} distance:".format(names[i],names[j]),distance)


