# -*- coding: utf-8 -*-
"""
@author  : 秦丹峰
@contact : daneven.jim@gmail.com
@time    : 2020/8/11 20:30
@file    : get_small_sample.py
@desc    : 提取小样本
"""
import os
import random

root = "/home/qindanfeng/work/deep_learning/datasets/vehicle_datasets"
img_path = root + '/JPEGImages'
img_list = os.listdir(img_path)

random.shuffle(img_list)
random.shuffle(img_list)
random.shuffle(img_list)

with open(root+'/ImageSets/Main/trainval_small_sample.txt', "w")as f:
    for img in img_list[:5000]:
        f.writelines(img.split('.')[0]+'\n')

with open(root+'/ImageSets/Main/test_small_sample.txt', "w")as f:
    for img in img_list[-1000:]:
        f.writelines(img.split('.')[0]+'\n')