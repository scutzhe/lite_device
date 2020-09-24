# Copyright @ 2019 Alibaba. All rights reserved.
# Created by ruhuan on 2019.09.09
""" python demo usage about MNN API """
from __future__ import print_function
import numpy as np
import MNN
import cv2
import time

def inference():
    """ inference mobilenet_v1 using a specific picture """
    interpreter = MNN.Interpreter("/home/zhanjinhao/codes/人脸识别/occlusion_face_recognition/model/mnn/faceRec_mobilenet_v2_cfg_tuned.mnn")
    session = interpreter.createSession()
    input_tensor = interpreter.getSessionInput(session)
    ori_image = cv2.imread('z:/qindanfeng/work/deep_learning/train_ssd_mobilenet/test_img/phone.jpg')
    #cv2 read as bgr format
    # image = image[..., ::-1]
    #change to rgb format
    resize_image = cv2.resize(ori_image, (300, 300))
    #resize to mobile_net tensor size
    image = resize_image.astype(float)
    # image = image - (103.94, 116.78, 123.68)
    # image = image * (0.017, 0.017, 0.017)
    image = (2.0 / 255.0) * image - 1.0
    #preprocess it
    image = image.transpose((2, 0, 1))
    # cv2 read shape is NHWC, Tensor's need is NCHW,transpose it
    tmp_input = MNN.Tensor((1, 3, 300, 300), MNN.Halide_Type_Float,\
                    image, MNN.Tensor_DimensionType_Caffe)
    #construct tensor from np.ndarray
    input_tensor.copyFrom(tmp_input)
    ctime = time.time()
    interpreter.runSession(session)
    print("inference time:", time.time()-ctime)
    output_tensor = interpreter.getSessionOutputAll(session)
    output = output_tensor
    loc = output['TFLite_Detection_PostProcess'].getData()
    cls = output['TFLite_Detection_PostProcess:1'].getData()
    score = output['TFLite_Detection_PostProcess:2'].getData()
    print(output['TFLite_Detection_PostProcess'].getData())
    print(output['TFLite_Detection_PostProcess:1'].getData())
    print(output['TFLite_Detection_PostProcess:2'].getData())
    print(output['TFLite_Detection_PostProcess:3'].getData())
    loc = np.array(loc)*300
    xmin = int(loc[1])
    ymin = int(loc[0])
    xmax = int(loc[3])
    ymax = int(loc[2])
    # cv2.rectangle(resize_image, (xmin,ymin), (xmax, ymax), (0,255,255))
    # cv2.imshow('img', resize_image)
    # cv2.waitKey(10000)

def inference_test(img_path):
    # 图片加载与处理
    ori_image = cv2.imread(img_path)
    resize_image = cv2.resize(ori_image, (300, 300))
    image = resize_image.astype(float)
    image = (2.0 / 255.0) * image - 1.0

    # 读取anchor文件，坐标转换需要用
    anchors = np.load('anchors.npy')
    anchors = anchors.reshape((-1))

    # 模型加载
    interpreter = MNN.Interpreter("z:/qindanfeng/work/deep_learning/train_ssd_mobilenet/tflite/v2-ssdlite-data_aug/tflite_graph_nopostprocess.mnn")
    session = interpreter.createSession()
    input_tensor = interpreter.getSessionInput(session)

    # 输入tensor
    caffe_input = MNN.Tensor((1, 3, 300, 300), MNN.Halide_Type_Float,\
                    image, MNN.Tensor_DimensionType_Caffe)
    tensorflow_input = MNN.Tensor((1, 300, 300, 3), MNN.Halide_Type_Float, \
                           image, MNN.Tensor_DimensionType_Tensorflow)

    input_tensor.copyFrom(tensorflow_input)

    # 执行推断
    ctime = time.time()
    checkInput = np.array(input_tensor.getData())
    interpreter.runSession(session)
    print("inference time:", time.time()-ctime)

    # 获得推断结果
    output = interpreter.getSessionOutputAll(session)
    loc = output['Squeeze'].getData()
    score = output['convert_scores'].getData()
    score = list(score)

    print("输入length:", len(input_tensor.getData())) # tmp_input:360000  test_input:360000
    print("输入维度:", input_tensor.getShape())
    print("输出维度box:", output['Squeeze'].getShape())
    print("输入: 均值{} 最大值{} 最小值{}".format(np.mean(checkInput), np.max(checkInput), np.min(checkInput)))
    print("输出维度score:", output['convert_scores'].getShape())
    print("输出box: 均值{} 最大值{} 最小值{}".format(np.mean(loc), np.max(loc), np.min(loc)))
    print("输出score: 均值{} 最大值{} 最小值{}".format(np.mean(score), np.max(score), np.min(score)))

    # 坐标转换需要用的scale
    x_scale = 10
    y_scale = 10
    h_scale = 5
    w_scale = 5
    score_threshold = 0.300000011921

    result = []
    for i in range(1917):
        info=[]
        noobj = score[3*i]  # 无目标
        phone = score[3*i+1]  # 打电话
        drink = score[3*i+2]  # 喝水

        if phone>noobj and phone>drink and phone>score_threshold:
            cls = '0'
            conf = phone
            info.append(cls)
            info.append(conf)

        elif drink>noobj and drink>phone and drink>score_threshold:
            cls = '1'
            conf = drink
            info.append(cls)
            info.append(conf)

        # 坐标转换
        ycenter = loc[4 * i + 0] / y_scale * anchors[4 * i + 2] + anchors[4 * i + 0]
        xcenter = loc[4 * i + 1] / x_scale * anchors[4 * i + 3] + anchors[4 * i + 1]
        h = np.exp(loc[4 * i + 2] / h_scale) * anchors[4 * i + 2]
        w = np.exp(loc[4 * i + 3] / w_scale) * anchors[4 * i + 3]

        ymin = (ycenter - h * 0.5) * 300
        xmin = (xcenter - w * 0.5) * 300
        ymax = (ycenter + h * 0.5) * 300
        xmax = (xcenter + w * 0.5) * 300

        info.append(xmin)
        info.append(ymin)
        info.append(xmax)
        info.append(ymax)
        result.append(info)

    # 未做NMS

    for i in result:
        x1 = int(i[2])
        y1 = int(i[3])
        x2 = int(i[4])
        y2 = int(i[5])
        cv2.rectangle(resize_image, (x1,y1), (x2, y2), (0,255,255))
    cv2.imshow('img', resize_image)
    cv2.waitKey(1000)

def inference_fast(img_path):
    # 图片加载与处理
    ori_image = cv2.imread(img_path)  # 读取图片
    resize_image = cv2.resize(ori_image, (300, 300))  # resize到300*300
    image = resize_image.astype(float)  # 转为float类型
    image = (2.0 / 255.0) * image - 1.0  # 标准化

    # 读取本地的anchor文件
    anchors = np.load('anchors.npy')
    anchors = anchors.reshape((-1))

    # 模型加载
    interpreter = MNN.Interpreter("z:/qindanfeng/work/deep_learning/train_ssd_mobilenet/tflite/v2-ssdlite-data_aug/tflite_graph_nopostprocess.mnn")
    session = interpreter.createSession()  # 创建会话
    input_tensor = interpreter.getSessionInput(session)  # 创建输入tensor

    # 两种加载方式，1、2两种都是可以用的
    # 1 CAFFE
    image = image.transpose((2, 0, 1))  # HWC->CHW
    tmp_input = MNN.Tensor((1, 3, 300, 300), MNN.Halide_Type_Float, image, MNN.Tensor_DimensionType_Caffe)
    # 2 TENSORFLOW
    input_tensor.copyFrom(tmp_input)

    # 执行推断
    ctime = time.time()
    interpreter.runSession(session)
    print("inference time:", time.time()-ctime)

    # 获得输出
    output = interpreter.getSessionOutputAll(session)
    loc = output['Squeeze'].getData()  # 坐标
    score = output['convert_scores'].getData()  # 置信度

    loc = np.array(loc)
    score = np.array(score)
    score = score.reshape((1917, 3))
    score1 = score[:,0]  # 无目标
    score2 = score[:,1]  # 打电话
    score3 = score[:,2]  # 喝水

    print("score1:",np.max(score1))
    print("score2:",np.max(score2))
    print("score3:",np.max(score3))

    # 提取score最大值对应的索引
    if np.max(score2)>np.max(score3):  # 打电话的score最大值大于喝水
        index = list(score2).index(np.max(score2))
        score = score2[index]
        cls = 'phone'
    else:
        index = list(score3).index(np.max(score3))
        score = score3[index]
        cls  = 'drink'

    # 坐标转换需要用的的scale
    x_scale = 10
    y_scale = 10
    h_scale = 5
    w_scale = 5

    # 坐标转换
    ycenter = loc[4 * index + 0] / y_scale * anchors[4 * index + 2] + anchors[4 * index + 0]
    xcenter = loc[4 * index + 1] / x_scale * anchors[4 * index + 3] + anchors[4 * index + 1]
    h = np.exp(loc[4 * index + 2] / h_scale) * anchors[4 * index + 2]
    w = np.exp(loc[4 * index + 3] / w_scale) * anchors[4 * index + 3]

    ymin = (ycenter - h * 0.5) * 300
    xmin = (xcenter - w * 0.5) * 300
    ymax = (ycenter + h * 0.5) * 300
    xmax = (xcenter + w * 0.5) * 300

    # 检查数组越界
    if xmin<0:
        xmin=0
    if ymin<0:
        ymin=0
    if xmax>300:
        xmax=300
    if ymax>300:
        ymax=300

    print("[INFO] class:{} | confidence:{} | xmin:{} ymin:{} xmax:{} ymax:{} | index:{}".format(cls, score, round(xmin), round(ymin), round(xmax), round(ymax), index))
    if score>0.95:
        cv2.rectangle(resize_image, (int(xmin),int(ymin)), (int(xmax), int(ymax)), (0,255,255))
    cv2.imshow('img', resize_image)
    cv2.waitKey(500)

if __name__ == "__main__":
    import os, json
    # inference() # phone:0.9987421631813049  drink:0.9955020546913147
    inference_test("z:/qindanfeng/work/deep_learning/train_ssd_mobilenet/test_img/drink.jpg")
    # inference_fast("z:/qindanfeng/work/deep_learning/train_ssd_mobilenet/test_img/drink.jpg")

