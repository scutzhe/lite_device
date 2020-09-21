#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
# @author  : 郑祥忠
# @license : (C) Copyright,2016-2020,广州海格星航科技
# @contact : dylenzheng@gmail.com
# @file    : txt2xml.py
# @time    : 9/17/20 5:19 PM
# @desc    :
'''

import os
import shutil
import cv2
from tqdm import tqdm

from xml.dom.minidom import Document
def convert2xml(xml_path,
                txt_name,
				num_bbox,
				bbox,
				image_full_name,
				full_path,
				width_value,
				height_value,
				class_name):

	doc = Document()  #创建DOM文档对象
	annotation = doc.createElement('annotation') #创建根元素
	doc.appendChild(annotation)

	folder = doc.createElement('folder')
	folder_text = doc.createTextNode('JPGImages')
	folder.appendChild(folder_text)
	annotation.appendChild(folder)

	filename = doc.createElement('filename')
	filename_text = doc.createTextNode(image_full_name)
	filename.appendChild(filename_text)
	annotation.appendChild(filename)

	path = doc.createElement('path')
	path_text = doc.createTextNode(full_path)
	path.appendChild(path_text)
	annotation.appendChild(path)

	source = doc.createElement('source')
	database = doc.createElement('database')
	database_text = doc.createTextNode('Unknown')
	database.appendChild(database_text)
	source.appendChild(database)
	annotation.appendChild(source)

	size = doc.createElement('size')
	# 宽
	width = doc.createElement('width')
	width_text = doc.createTextNode(str(width_value))
	width.appendChild(width_text)
	# 高
	height = doc.createElement('height')
	height_text = doc.createTextNode(str(height_value))
	height.appendChild(height_text)
	# 通道数
	depth = doc.createElement('depth')
	depth_text = doc.createTextNode('3')
	depth.appendChild(depth_text)
	size.appendChild(width)
	size.appendChild(height)
	size.appendChild(depth)
	annotation.appendChild(size)

	segmented = doc.createElement('segmented')
	segmented_text = doc.createTextNode('0')
	segmented.appendChild(segmented_text)
	annotation.appendChild(segmented)

	for i in range(int(num_bbox)):
		object = doc.createElement('object')
		# class name
		name = doc.createElement('name')  # 因为我们同一张图片都是同一类别
		name_text = doc.createTextNode(class_name)
		name.appendChild(name_text)
		# pose
		pose = doc.createElement('pose')
		pose_text = doc.createTextNode('Unspecified')
		pose.appendChild(pose_text)
		# truncated
		truncated = doc.createElement('truncated')
		truncated_text = doc.createTextNode('0')
		truncated.appendChild(truncated_text)
		# difficult
		difficult = doc.createElement('difficult')
		difficult_text = doc.createTextNode('0')
		difficult.appendChild(difficult_text)
		# bbox
		box = bbox[4*i:4*(i+1)]
		bndbox = doc.createElement('bndbox')
		xmin = doc.createElement('xmin')
		xmin_text = doc.createTextNode(str(box[0]))
		ymin = doc.createElement('ymin')
		ymin_text = doc.createTextNode(str(box[1]))
		xmax = doc.createElement('xmax')
		xmax_text = doc.createTextNode(str(int(box[2])))
		ymax = doc.createElement('ymax')
		ymax_text = doc.createTextNode(str(int(box[3])))
		xmin.appendChild(xmin_text)
		ymin.appendChild(ymin_text)
		xmax.appendChild(xmax_text)
		ymax.appendChild(ymax_text)
		bndbox.appendChild(xmin)
		bndbox.appendChild(ymin)
		bndbox.appendChild(xmax)
		bndbox.appendChild(ymax)

		object.appendChild(name)
		object.appendChild(pose)
		object.appendChild(truncated)
		object.appendChild(difficult)
		object.appendChild(bndbox)
		annotation.appendChild(object)

	# 输出文件
	f = open('{}/{}.xml'.format(xml_path, txt_name.split('.')[0]),'w')
	doc.writexml(f,indent = '\t',newl = '\n', addindent = '\t',encoding='utf-8')
	f.close()


if __name__ == '__main__':
	annotation_dir = "/home/zhex/data/arm_device_all/annotations"
	image_dir = "/home/zhex/data/arm_device_all/images"
	xml_dir = "/home/zhex/data/arm_device_voc/Annotations"
	image_voc_dir = "/home/zhex/data/arm_device_voc/JPEGImages"
	if not os.path.exists(xml_dir):
		os.makedirs(xml_dir)
	if not os.path.exists(image_voc_dir):
		os.makedirs(image_voc_dir)
	for name in tqdm(os.listdir(annotation_dir)):
		annotation_path = os.path.join(annotation_dir,name)
		annotation_file = open(annotation_path,"r")
		image_path = os.path.join(image_dir,name.split(".")[0]+".jpg")
		shutil.copy(image_path,image_voc_dir)
		image = cv2.imread(image_path)
		H,W = image.shape[:2]
		lines = annotation_file.readlines()

		num_box = len(lines)
		class_name = "device"
		image_name = name.split(".")[0]+".jpg"
		image_path = "arm_device_voc/JPEGImages/" + image_name
		bbox = []
		for line in lines:
			info = line.strip().split(" ")
			x1 = int((float(info[1]) - float(info[3]) / 2) * W)
			y1 = int((float(info[2]) - float(info[4]) / 2) * H)
			x2 = int((float(info[1]) + float(info[3]) / 2) * W)
			y2 = int((float(info[2]) + float(info[4]) / 2) * H)
			bbox.append(x1)
			bbox.append(y1)
			bbox.append(x2)
			bbox.append(y2)
		convert2xml(xml_dir,name,num_box,bbox,image_name,image_path,W,H,class_name)
	print("txt2xml transformation successful！")