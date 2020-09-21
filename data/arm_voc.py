#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
# @author  : 郑祥忠
# @license : (C) Copyright,2016-2020,广州海格星航科技
# @contact : dylenzheng@gmail.com
# @file    : arm_voc.py
# @time    : 9/17/20 5:04 PM
# @desc    : 
'''
import os
import cv2
import logging
import numpy as np
import xml.etree.ElementTree as ET

class ARMDataset:
    def __init__(self, root, is_train=False,transform=None,target_transform=None):
        self.root = root
        self.image_dir = self.root + "/" + "JPEGImages"
        self.annotation_dir = self.root + "/" + "Annotations"
        self.annotations = os.listdir(self.annotation_dir)
        self.transform = transform
        self.target_transform = target_transform
        self.is_train = is_train
        if is_train:
                image_sets_file = self.root + "/" + "train.txt"
        else:
                image_sets_file = self.root + "/" + "val.txt"

        self.ids = ARMDataset._read_image_ids(image_sets_file)

        # if the labels file exists, read in the class names
        label_file_name = self.root + "/" + "labels.txt"

        if os.path.isfile(label_file_name):
            with open(label_file_name, 'r') as infile:
                classes = infile.read().splitlines()

            # prepend BACKGROUND as first class
            classes.insert(0, 'BACKGROUND')
            self.class_names = tuple(classes)
            logging.info("device Labels read from file: " + str(self.class_names))

        else:
            logging.info("No labels file, using default Army classes.")
            self.class_names = ("BACKGROUND","device")

        self.class_dict = {class_name: i for i, class_name in enumerate(self.class_names)}

    def __getitem__(self, index):
        image_id = self.ids[index]
        # print("image_id=",image_id)
        boxes, labels, is_difficult = self._get_annotation(image_id)
        boxes = boxes[is_difficult == 0]
        labels = labels[is_difficult == 0]
        image = self._read_image(image_id)
        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)
        return image, boxes, labels

    def get_image(self, index):
        image_id = self.ids[index]
        image = self._read_image(image_id)
        if self.transform:
            image, _ = self.transform(image)
        return image

    def get_annotation(self, index):
        image_id = self.ids[index]
        return image_id, self._get_annotation(image_id)

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def _read_image_ids(image_sets_file):
        ids = []
        with open(image_sets_file) as f:
            for line in f:
                ids.append(line.rstrip())
        return ids

    def _get_annotation(self, image_id):
        annotation_file = self.root + "/" + "Annotations/{}.xml".format(image_id)
        # print("annotation_file=",annotation_file)
        objects = ET.parse(annotation_file).findall("object")
        boxes = []
        labels = []
        is_difficult = []
        for object in objects:
            class_name = object.find('name').text.lower().strip()
            # we're only concerned with clases in our list
            if class_name in self.class_dict:
                bbox = object.find('bndbox')
                # VOC dataset format follows Matlab, in which indexes start from 0
                x1 = int(bbox.find('xmin').text)
                y1 = int(bbox.find('ymin').text)
                x2 = int(bbox.find('xmax').text)
                y2 = int(bbox.find('ymax').text)
                boxes.append([x1, y1, x2, y2])

                labels.append(self.class_dict[class_name])
                is_difficult_str = object.find('difficult').text
                is_difficult.append(int(is_difficult_str) if is_difficult_str else 0)

        return (np.array(boxes, dtype=np.float32),
                np.array(labels, dtype=np.int64),
                np.array(is_difficult, dtype=np.uint8))

    def _read_image(self, image_id):
        image_path = self.root  + "/" +"JPEGImages/{}.jpg".format(image_id)
        # print("image_path=",image_path)
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image