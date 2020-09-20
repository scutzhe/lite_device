#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
# @author  : 郑祥忠
# @license : (C) Copyright,2016-2020,广州海格星航科技
# @contact : dylenzheng@gmail.com
# @file    : demo_arm.py
# @time    : 9/15/20 3:07 PM
# @desc    : 
'''
import os
import cv2
from tqdm import tqdm
from net.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor


if __name__ == "__main__":
    model_path = 'models/90_1.83812066167593.pth'
    label_path = '/home/zhex/data/army/labels.txt'
    image_dir = "/home/zhex/test_result/tag_device"
    save_dir = "eval_results"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    class_names = [name.strip() for name in open(label_path).readlines()]
    model_name = model_path.split('/')[-2]
    net = create_mobilenetv2_ssd_lite(len(class_names)+1, is_test=True)
    net.load(model_path)
    predictor = create_mobilenetv2_ssd_lite_predictor(net, candidate_size=200,device="cuda:0")

    for image_name in tqdm(os.listdir(image_dir)):
        image_path = os.path.join(image_dir,image_name)
        image_bgr = cv2.imread(image_path)
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        boxes, labels, prob = predictor.predict(image, 10, 0.4)
        boxes = boxes.numpy()
        if boxes.shape[0] == 0:
            continue
        for i in range(boxes.shape[0]):
            box = boxes[i, :]
            cv2.rectangle(image_bgr, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255,0, 0), 2)
            # label = f"{class_names[labels[i]]}: {prob[i]:.2f}"
            # cv2.imshow("image",image_bgr)
            # cv2.waitKey(1000)
            # cv2.imwrite("eval_results" + "/" +"{}".format(image_name),image_bgr)