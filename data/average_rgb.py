#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
# @author  : dylen
# @license :
# @contact : dylenzheng@gmail.com
# @file    : average_rgb.py
# @time    : 9/21/20 8:36 PM
# @desc    : 
'''
import os
import cv2
import numpy as np
from tqdm import tqdm
from math import pow

def cal_avg(image_dir):
    """
    @param image_dir:
    @return:
    """
    assert os.path.exists(image_dir),"{} is null !!!".format(image_dir)
    image_Bmean = []
    image_Gmean = []
    image_Rmean = []
    for name in tqdm(os.listdir(image_dir)):
        image_path = os.path.join(image_dir,name)
        image = cv2.imread(image_path)
        image_Bmean.append(np.mean(image[:,:,0]))
        image_Gmean.append(np.mean(image[:,:,1]))
        image_Rmean.append(np.mean(image[:,:,2]))
    B_mean = np.mean(image_Bmean)
    G_mean = np.mean(image_Gmean)
    R_mean = np.mean(image_Rmean)
    return B_mean,G_mean,R_mean

def cal_std(image_dir):
    """
    @param image_dir:
    @return:
    """
    assert os.path.exists(image_dir), "{} is null !!!".format(image_dir)
    # B_mean, G_mean, R_mean = cal_avg(image_dir)
    B_mean, G_mean, R_mean = 94,111,118
    image_Bstd = []
    image_Gstd = []
    image_Rstd = []
    for name in tqdm(os.listdir(image_dir)):
        image_path = os.path.join(image_dir,name)
        image = cv2.imread(image_path)
        image_Bstd.append(pow((np.mean(image[:, :, 0])-B_mean),2))
        image_Gstd.append(pow((np.mean(image[:, :, 1])-G_mean),2))
        image_Rstd.append(pow((np.mean(image[:, :, 2])-R_mean),2))
    B_std = np.mean(image_Bstd)
    G_std = np.mean(image_Gstd)
    R_std = np.mean(image_Rstd)
    return B_std,G_std,R_std

def image_divide_std(image_path,std=np.array([94,111,118])):
    """
    @param image_path:
    @param std:
    @return:
    """
    image = cv2.imread(image_path)
    image = image / std
    return image

# if __name__ == "__main__":
#     image_dir = "/home/zhex/data/arm_device_voc/JPEGImages"
#     # B_mean, G_mean, R_mean = cal_avg(image_dir)
#     # print("B_mean=",B_mean)
#     # print("G_mean=",G_mean)
#     # print("R_mean=",R_mean)
#     B_std, G_std, R_std = cal_std(image_dir)
#     print("B_std=",B_std)
#     print("G_std=",G_std)
#     print("R_std=",R_std)


if __name__ == "__main__":
    image_path = "/home/zhex/test_result/tag_device/01940002_frame_6.jpg"
    image = cv2.imread(image_path)
    value = image_divide_std(image_path)
    # print("value=",value)
    print("image=",image)