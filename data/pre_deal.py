#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
# @author  : 郑祥忠
# @license : (C) Copyright,2016-2020,广州海格星航科技
# @contact : dylenzheng@gmail.com
# @file    : pre_deal.py
# @time    : 9/17/20 10:18 AM
# @desc    : 
'''
import os
import shutil
from tqdm import tqdm

#image_dir = "/home/zhex/Downloads/tmp/01940002_output/data/obj"
#/home/zhex/Downloads/tmp/01940002_output
def deal_data(image_dir,txt_file_path,image_save_dir,annotation_save_dir):
    """
    @param image_dir:
    @param txt_file_path:
    @param save_dir:
    @return:
    """
    assert os.path.exists(image_dir),"{} is empty".format(image_dir)
    if not os.path.exists(image_save_dir):
        os.makedirs(image_save_dir)
    if not os.path.exists(annotation_save_dir):
        os.makedirs(annotation_save_dir)

    txt_file = open(txt_file_path,"r")
    index = 0
    middle_dir = image_dir.strip().split("/")
    pre_image_dir = middle_dir[0] + "/" + middle_dir[1] + "/"+ middle_dir[2] + "/"+ \
                    middle_dir[3] + "/" + middle_dir[4] + "/" +middle_dir[5] + "/"
    for line in tqdm(txt_file.readlines()):
        line = line.strip().split(".")
        jpg_name = line[0] + ".jpg"
        txt_name = line[0] + ".txt"
        image_path = pre_image_dir + jpg_name
        # print("image_path=",image_path)
        annotation_path = pre_image_dir + txt_name
        # print("annotation_path=",annotation_path)
        shutil.copy(image_path,image_save_dir)
        shutil.copy(annotation_path,annotation_save_dir)
        index += 1
    print("index=",index)

def check(image_dir,annotation_dir):
    """
    @param image_dir:
    @param annotation_dir:
    @return:
    """
    assert os.path.exists(image_dir),"{} is empty!!!".format(image_dir)
    assert os.path.exists(annotation_dir),"{} is empty !!!".format(annotation_dir)
    common_names = []
    for name in tqdm(os.listdir(annotation_dir)):
        pre_name = name.strip().split(".")[0]
        common_names.append(pre_name)
    for name in os.listdir(image_dir):
        pre_name = name.strip().split(".")[0]
        if pre_name not in common_names:
            print("pre_name=",pre_name)

def check_image(image_dir):
    """
    @param image_dir:
    @return:
    """
    import cv2
    index = 0
    for image_name in tqdm(os.listdir(image_dir)):
        image_path = os.path.join(image_dir,image_name)
        try:
            image = cv2.imread(image_path)
        except Exception as e:
            index += 1
    print("index=",index)


def delete_empty_file(image_dir,annotation_dir):
    """
    @param image_dir:
    @param annotation_dir:
    @return:
    """
    assert os.path.exists(image_dir),"{} is empty !!!".format(image_dir)
    assert os.path.exists(annotation_dir), "{} is empty !!!".format(annotation_dir)
    num = 0
    for name in tqdm(os.listdir(annotation_dir)):
        annotation_path = os.path.join(annotation_dir,name)
        filex = open(annotation_path,"r").readlines()
        if len(filex) == 0:
            # print("path=",annotation_path)
            os.remove(annotation_path)
            image_path = os.path.join(image_dir,name.strip().split(".")[0] + ".jpg")
            # print("image_path=",image_path)
            os.remove(image_path)
            num += 1
    print("delete_num=",num)

def split_dataset(image_dir,annotations_dir,test_image_dir,test_annotations_dir):
    """
    @param image_dir:
    @param annotations_dir:
    @param test_image_dir:
    @param test_annotations_dir:
    @return:
    """
    assert os.path.exists(image_dir),"{} is empty!!!".format(image_dir)
    assert os.path.exists(annotations_dir),"{} is empty!!!".format(image_dir)
    if not os.path.exists(test_image_dir):
        os.makedirs(test_image_dir)
    if not os.path.exists(test_annotations_dir):
        os.makedirs(test_annotations_dir)
    length = len(os.listdir(image_dir))
    test_length = int(0.3 * length)
    index = 0
    for image_name in tqdm(os.listdir(image_dir)):
        image_path = os.path.join(image_dir,image_name)
        annotation_name = image_name.strip().split(".")[0] + ".txt"
        annotation_path = os.path.join(annotations_dir,annotation_name)
        # print("annotation_path=",annotation_path)
        # print("image_path=",image_path)
        shutil.move(image_path,test_image_dir)
        shutil.move(annotation_path,test_annotations_dir)
        index += 1
        if index == test_length:
            break

def get_txt(image_dir):
    """
    @param image_dir:
    @return:
    """
    import random
    assert os.path.exists(image_dir),"{} is empty!!!".format(image_dir)
    train_file = open("train.txt","a")
    val_file = open("val.txt","a")
    image_names = os.listdir(image_dir)
    length = len(image_names)
    train_names = random.sample(image_names,int(0.9 * length))
    val_names = [name for name in image_names if name not in train_names]
    index_train = 0
    for name in tqdm(train_names):
        # train_file.write(name.strip().split(".")[0] + ".txt" + "\n")
        train_file.write(name.strip().split(".")[0]  + "\n")
        index_train += 1
    index_val = 0
    for name in tqdm(val_names):
        # val_file.write(name.strip().split(".")[0] + ".txt" + "\n")
        val_file.write(name.strip().split(".")[0] + "\n")
        index_val += 1
    print("index_train=",index_train)
    print("index_val=",index_val)

def check_xywh_xyxy(annotation_path,image_path):
    """
    @param annotation_path:
    @param image_path:
    @return:
    """
    import cv2
    filex = open(annotation_path,"r")
    image_name = image_path.split("/")[-1]
    image = cv2.imread(image_path)
    H,W = image.shape[:2]
    for line in filex.readlines():
        info = line.strip().split(" ")
        print("info=",info)
        x1 = int((float(info[1])-float(info[3])/2) * W)
        y1 = int((float(info[2])-float(info[4])/2) * H)
        x2 = int((float(info[1])+float(info[3])/2) * W)
        y2 = int((float(info[2])+float(info[4])/2) * H)
        cv2.rectangle(image,(x1,y1),(x2,y2),(0,0,255),2)
    cv2.imshow("image",image)
    # cv2.waitKey(1000)
    cv2.imwrite("result/" + "{}".format(image_name),image)

def get_image(txt_path,eval_dir):
    """
    @param txt_path:
    @return:
    """
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)
    filex = open(txt_path,"r")
    for line in tqdm(filex.readlines()):
        image_name = line.strip() + ".jpg"
        image_path = os.path.join("/home/zhex/data/army/JPEGImages",image_name)
        shutil.copy(image_path,eval_dir)

def cut_image(annotation_dir,image_dir,save_dir):
    """
    @param annotation_dir:
    @param image_dir:
    @param save_dir:
    @return:
    """
    import cv2
    for name in tqdm(os.listdir(annotation_dir)):
        annotation_path = os.path.join(annotation_dir,name)
        middle_name = str(name.split(".")[0])
        image_name = middle_name + ".jpg"
        image_path = os.path.join(image_dir,image_name)
        image = cv2.imread(image_path)
        H,W = image.shape[:2]
        index = 0
        for line in open(annotation_path,"r").readlines():
            info = line.strip().split(" ")
            x1 = int((float(info[1]) - float(info[3]) / 2) * W)
            y1 = int((float(info[2]) - float(info[4]) / 2) * H)
            x2 = int((float(info[1]) + float(info[3]) / 2) * W)
            y2 = int((float(info[2]) + float(info[4]) / 2) * H)

            W0 = x2 -x1
            H0 = y2 - y1
            x1_ = x1 - W0//4
            y1_ = y1 - H0//4
            x2_ = x2 + W0//4
            y2_ = y2 + H0//4

            if x1_ < 0:
                x1_ = 0
            if y1_ < 0:
                y1_ = 0
            if x2_ > 1920:
                x2_ = 1920
            if y2_ > 1080:
                y2_ = 1080
            image_new = image[y1_:y2_,x1_:x2_,:]
            # cv2.rectangle(image,(x1,y1),(x2,y2),(0,0,255),2)
            # cv2.imshow("image",image_new)
            # cv2.waitKey(10000)
            index += 1
            cv2.imwrite(save_dir + "/" +"{}_{}.jpg".format(middle_name,index),image_new)

def get_val_all(image_dir,txt_file_path,save_dir):
    """
    @param image_dir:
    @param txt_file_path:
    @param save_dir:
    @return:
    """
    import cv2
    assert os.path.exists(image_dir),"{} is null !!!".format(image_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    filex = open(txt_file_path)
    for line in tqdm(filex.readlines()):
        annotation_path = "/home/zhex/data/arm_device_all/annotations/" + line.strip() + ".txt"
        # print("annotation_path=",annotation_path)
        image_path = os.path.join(image_dir,line.strip() + ".jpg")
        # print("image_path=",image_path)
        middle_name = line.strip()
        image = cv2.imread(image_path)
        H,W = image.shape[:2]
        index = 0
        for line in open(annotation_path,"r").readlines():
            info = line.strip().split(" ")
            x1 = int((float(info[1]) - float(info[3]) / 2) * W)
            y1 = int((float(info[2]) - float(info[4]) / 2) * H)
            x2 = int((float(info[1]) + float(info[3]) / 2) * W)
            y2 = int((float(info[2]) + float(info[4]) / 2) * H)

            W0 = x2 -x1
            H0 = y2 - y1
            x1_ = x1 - W0//4
            y1_ = y1 - H0//4
            x2_ = x2 + W0//4
            y2_ = y2 + H0//4

            if x1_ < 0:
                x1_ = 0
            if y1_ < 0:
                y1_ = 0
            if x2_ > 1920:
                x2_ = 1920
            if y2_ > 1080:
                y2_ = 1080
            image_new = image[y1_:y2_,x1_:x2_,:]
            cv2.imwrite(save_dir + "/" + "{}_{}.jpg".format(middle_name, index), image_new)
            index += 1

# if __name__ == "__main__":
#     image_dir = "/home/zhex/Downloads/tmp/01940006_output/data/obj"
#     txt_file_path = "/home/zhex/Downloads/tmp/01940006_output/data/test.txt"
#     image_save_dir = "/home/zhex/data/arm_device_all/images/"
#     annotation_save_dir = "/home/zhex/data/arm_device_all/annotations/"
#     deal_data(image_dir,txt_file_path,image_save_dir,annotation_save_dir)


# if __name__ == "__main__":
#     image_dir = "/home/zhex/data/arm_device_all/images"
#     annotation_dir = "/home/zhex/data/arm_device_all/annotations"
#     check(image_dir,annotation_dir)
#     check(annotation_dir,image_dir)


# if __name__ == "__main__":
#     image_dir = "/home/zhex/data/arm_device_all/images"
#     annotation_dir = "/home/zhex/data/arm_device_all/annotations"
#     delete_empty_file(image_dir,annotation_dir)


## 采用voc格式之后用不着这个方法
# if __name__ == "__main__":
#     image_dir = "/home/zhex/data/arm_device/images/train"
#     annotations_dir = "/home/zhex/data/arm_device/annotations/train"
#     test_image_dir = "/home/zhex/data/arm_device/images/test"
#     test_annotations_dir = "/home/zhex/data/arm_device/annotations/test"
#     split_dataset(image_dir,annotations_dir,test_image_dir,test_annotations_dir)


# if __name__ == "__main__":
#     image_dir = "/home/zhex/data/arm_device_voc/JPEGImages"
#     get_txt(image_dir)
#     annotation_path = "/home/zhex/data/arm_device/annotations/01940002_frame_1.txt"
#     image_path = "/home/zhex/data/arm_device/images/01940002_frame_1.jpg"
#     check_xywh_xyxy(annotation_path,image_path)

# if __name__ == "__main__":
#     image_dir = "/home/zhex/data/arm_device/images"
#     annotation_dir = "/home/zhex/data/arm_device/annotations"
#     for annotation_name in os.listdir(annotation_dir):
#         annotation_path = os.path.join(annotation_dir,annotation_name)
#         image_name = annotation_name.split(".")[0] + ".jpg"
#         image_path = os.path.join(image_dir,image_name)
#         print("image_path=",image_path)
#         check_xywh_xyxy(annotation_path,image_path)


# if __name__ == '__main__':
#     val_path = "/home/zhex/data/army/val.txt"
#     save_dir = "/home/zhex/test_result/army_device"
#     get_image(val_path,save_dir)


# if __name__ == "__main__":
#     image_dir = "/home/zhex/data/arm_device_all/images"
#     txt_file_path = "/home/zhex/data/arm_device_voc/val.txt"
#     save_dir = "/home/zhex/data/arm_device_all/mini_images"
#     get_val_all(image_dir,txt_file_path,save_dir)

# if __name__ == '__main__':
#     annotation_dir = "/home/zhex/data/arm_device/annotations"
#     image_dir = "/home/zhex/data/arm_device/images"
#     save_dir = "/home/zhex/data/arm_device/mini_image"
#     cut_image(annotation_dir, image_dir, save_dir)


