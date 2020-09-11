# -*- coding: utf-8 -*-
"""
@author  : 秦丹峰
@contact : daneven.jim@gmail.com
@time    : 2020/8/4 22:52
@file    : detection_demo.py
@desc    : 运行demo
"""
from net.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
from utils.misc import Timer
import cv2
import sys
import os
import torch

# if len(sys.argv) < 5:
#     print('Usage: python run_ssd_example.py <net type>  <model path> <label path> <image path>')
#     sys.exit(0)

model_path = 'saved_model/ori_mbn_v2_ssdlite/best.pth'
label_path = 'saved_model/voc-model-labels.txt'
image_path = '/home/qindanfeng/work/deep_learning/datasets/vehicle_datasets/JPEGImages/'
imgs = [v for v in os.listdir(image_path) if v.endswith('.jpg')]

class_names = [name.strip() for name in open(label_path).readlines()]
model_name = model_path.split('/')[-2]
net = create_mobilenetv2_ssd_lite(len(class_names)+1, is_test=True)
net.load(model_path)
# net.load_model(model_path)
predictor = create_mobilenetv2_ssd_lite_predictor(net, candidate_size=200,device="cuda:0")



for img in imgs[:1]:
    orig_image = cv2.imread(image_path+img)
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    boxes, labels, probs = predictor.predict(image, 10, 0.4)

    for i in range(boxes.size(0)):
        box = boxes[i, :]
        # print('检测到物体，矩形框：', box[0], box[1], box[2], box[3])
        cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)
        #label = f"""{voc_dataset.class_names[labels[i]]}: {probs[i]:.2f}"""
        label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
        cv2.putText(orig_image, label,
                    (box[0] + 20, box[1] + 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,  # font scale
                    (255, 0, 255),
                    2)  # line type
    out_path = 'output/'+model_name
    os.makedirs(out_path,exist_ok=True)
    path = out_path+'/'+img
    cv2.imwrite(path, orig_image)
    print(f"Found {len(probs)} objects. The output image is {path}")
