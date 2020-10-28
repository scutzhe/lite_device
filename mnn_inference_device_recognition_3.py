#!/usr/bin/python3
# *_* coding: utf-8 *_*
# @Author: shengyang
# @Email: samonsix@163.com
# @IDE: PyCharm
# @File: mnn_demo.py
# @Modify Time        @Author    @Version    @Desciption
# ----------------    -------    --------    -----------
# 2020.09.27 13:45    shengyang      v0.1        creation
# 2020.09.29 10:09    dylen          v0.2        revision

import sys
sys.path.append("..")
import MNN.expr as F
from torchvision import transforms as T
from PIL import Image
import numpy as np
from scipy.spatial.distance import euclidean


mnn_model_path = '/home/zhex/share_folder/device_recognition_224.mnn'
vars = F.load_as_dict(mnn_model_path)
inputVar = vars["input0"]
# 查看输入信息
print('input shape: ', inputVar.shape)

image_path = "/home/zhex/git_me/DeviceRetrieval/resource/1-1.jpg"
image_path2 = "/home/zhex/git_me/DeviceRetrieval/resource/2-2.jpg"

# 写入数据
input_image = Image.open(image_path)
input_image2 = Image.open(image_path2)
preprocess = T.Compose([
                        T.Resize((224, 224)),
                        T.ToTensor(),
                        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                        ])

input_tensor = preprocess(input_image)
input_tensor2 = preprocess(input_image2)

inputVar.write(input_tensor.tolist())
outputVar = vars['output0']    # 查看输出结果

fea1 = np.array(outputVar.read())
print("fea1[0]:",fea1[0])

inputVar.write(input_tensor2.tolist())
outputVar = vars['output0']    # 查看输出结果
fea2 = np.array(outputVar.read())
print("fea2[0]:",fea2[0])
fea1 = fea1 / np.linalg.norm(fea1)
fea2 = fea2 / np.linalg.norm(fea2)
euc_dis = euclidean(fea1, fea2)
print(euc_dis)
