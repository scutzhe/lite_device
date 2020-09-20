#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
# @author  : 郑祥忠
# @license : (C) Copyright,2016-2020,广州海格星航科技
# @contact : dylenzheng@gmail.com
# @file    : arm_dataset.py
# @time    : 9/14/20 8:34 AM
# @desc    : 
'''
import os
import cv2
import numpy as np

class ARMDataset:
    def __init__(self,image_dir,annotation_dir,is_train=True,transform=None):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.annotations = os.listdir(self.annotation_dir)
        self.transform = transform
        self.is_train = is_train

    def __getitem__(self, index):
        if self.is_train:
            train_annotation_txt = self.annotations[index]
            image, coordination, label = self.get_info(train_annotation_txt)
            if self.transform:
                image, coordination, label = self.transform(image, coordination, label)
            return image,coordination,label

        else:
            test_annotation_txt = self.annotations[index]
            image, coordination, label = self.get_info(test_annotation_txt)
            if self.transform:
                image, coordination, label = self.transform(image, coordination, label)
            return image, coordination, label

    def __len__(self,):
        return len(self.annotations)

    def get_info(self,txt):
        """
        @param txt:
        @return:
        """
        image_name = txt.strip().split(".")[0] + ".jpg"
        image_path = os.path.join(self.image_dir,image_name)
        image = cv2.imread(image_path)
        H, W = image.shape[:2]
        txt_path = os.path.join(self.annotation_dir,txt)
        annotation_file = open(txt_path,"r").readlines()
        coordination = []
        labels = []
        for line in annotation_file:
            info = line.strip().split(" ")
            class_id = int(info[0])
            x1 = int(float(info[1]) * W)
            y1 = int(float(info[2]) * H)
            x2 = x1 + int(float(info[3]) * W)
            y2 = y1 + int(float(info[4]) * H)
            coordination.append([x1,y1,x2,y2])
            labels.append(class_id)

        return image, np.array(coordination,np.float32), np.array(labels,np.int64)

if __name__ == "__main__":
    import torch
    from torch.utils.data import DataLoader
    from data_preprocessing import TrainAugmentation, TestTransform
    from net.config import ssd_config as config
    train_image_dir = "/home/zhex/data/arm_device/images/train"
    train_annotations = "/home/zhex/data/arm_device/annotations/train"
    test_image_dir = "/home/zhex/data/arm_device/images/test"
    test_annotations = "/home/zhex/data/arm_device/annotations/test"
    transform = TrainAugmentation(config.image_size, config.image_mean, config.image_std)
    dataset = ARMDataset(train_image_dir,train_annotations,is_train=True,transform=transform)
    test_dataLoader = DataLoader(dataset,batch_size=1,shuffle=False,drop_last=True)
    for image,coordination,label in test_dataLoader:
        print("image.size()=",image.size())
        print("coordination.size()=",coordination.size())
        print("label.size()=",label.size())
