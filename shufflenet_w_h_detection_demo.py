# -*- coding: utf-8 -*-
"""
@author  : 秦丹峰
@contact : daneven.jim@gmail.com
@time    : 2020/8/4 22:52
@file    : detection_demo.py
@desc    : 运行demo
"""

from net.shufflenet_v2_ssdlite_w_h import create_shufflenetv2_ssd_lite, create_shufflenetv2_ssd_lite_predictor
from net.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
from utils.misc import Timer
import cv2
import sys
import os
import time
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
os.environ["OMP_NUM_THREADS"] = "1"

# if len(sys.argv) < 5:
#     print('Usage: python run_ssd_example.py <net type>  <model path> <label path> <image path>')
#     sys.exit(0)
net_type = 'shufflenet'
model_path = '/home/zhanjinhao/codes/车辆检测/ssdlite_mobilenet_v2/saved_model/shufflenet_ssd_w_h/last.pth'
label_path = 'saved_model/voc-model-labels.txt'
image_path = '/home/qindanfeng/work/deep_learning/datasets/vehicle_datasets/JPEGImages/'
imgs = [v for v in os.listdir(image_path) if v.endswith('.jpg')]
class_names = [name.strip() for name in open(label_path).readlines()]
model_name = model_path.split('/')[-2]

net = create_shufflenetv2_ssd_lite(len(class_names), is_test=True)
# 当保存的模型包含网路参数，训练参数时用这个加载
net.load_model(model_path)
# 当保存的模性只包含网络参数时，用这个加载
# net.load(model_path)
predictor = create_shufflenetv2_ssd_lite_predictor(net, candidate_size=200,device="cuda:0")
predict_cost = 0
process_cost = 0
for img in imgs[:100]:
    process_tic = time.time()
    orig_image = cv2.imread(image_path+img)
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    process_toc = time.time()
    process_cost += process_toc-process_tic
    boxes, labels, probs = predictor.predict(image, 10, 0.4)
    predict_toc = time.time()
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
    out_path = 'output/' + model_name
    os.makedirs(out_path, exist_ok=True)
    path = out_path + '/' + img
    cv2.imwrite(path, orig_image)
    print(f"Found {len(probs)} objects. The output image is {path}")
    predict_cost+=predict_toc-process_tic
print('图片预处理耗时：'+str(process_cost*1000)+'ms')
print('推断一张图片耗时：'+str(predict_cost*1000)+'ms')