#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
# @author  : 郑祥忠
# @license : (C) Copyright,2016-2020,广州海格星航科技
# @contact : dylenzheng@gmail.com
# @file    : onnx_inference.py
# @time    : 9/19/20 11:35 AM
# @desc    :
'''
import os
import cv2
import time
import torch
import torchvision
import onnxruntime
from PIL import Image
from data.transforms import *
from net.config import ssd_config as config



def image_deal(image_path):
    """
    @param image_path:
    @return:
    """
    image = Image.open(image_path)
    image_resize = image.resize((300,300),Image.BICUBIC)
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()
                                    ])
    img_tensor = transform(image_resize)
    img_tensor = img_tensor.unsqueeze(0)
    return img_tensor

def to_numpy(tensor):
    """
    @param tensor:
    @return:
    """
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def to_tensor(numpy):
    """
    @param numpy:
    @return:
    """
    return torch.from_numpy(numpy)


class ONNXModel():
    def __init__(self, onnx_path):
        """
        :param onnx_path:
        """
        self.onnx_session = onnxruntime.InferenceSession(onnx_path)
        self.input_name = self.get_input_name(self.onnx_session)
        self.output_name = self.get_output_name(self.onnx_session)
        print("input_name:{}".format(self.input_name))
        print("output_name:{}".format(self.output_name))

    def get_output_name(self, onnx_session):
        """
        output_name = onnx_session.get_outputs()[0].name
        :param onnx_session:
        :return:
        """
        output_name = []
        for node in onnx_session.get_outputs():
            output_name.append(node.name)
        return output_name

    def get_input_name(self, onnx_session):
        """
        input_name = onnx_session.get_inputs()[0].name
        :param onnx_session:
        :return:
        """
        input_name = []
        for node in onnx_session.get_inputs():
            input_name.append(node.name)
        return input_name

    def get_input_feed(self, input_name, image_numpy):
        """
        input_feed={self.input_name: image_numpy}
        :param input_name:
        :param image_numpy:
        :return:
        """
        input_feed = {}
        for name in input_name:
            input_feed[name] = image_numpy
        return input_feed

    def forward(self, image_numpy):
        '''
        # image_numpy = image.transpose(2, 0, 1)
        # image_numpy = image_numpy[np.newaxis, :]
        # onnx_session.run([output_name], {input_name: x})
        # :param image_numpy:
        # :return:
        '''
        # 输入数据的类型必须与模型一致,以下三种写法都是可以的
        # scores, boxes = self.onnx_session.run(None, {self.input_name: image_numpy})
        # scores, boxes = self.onnx_session.run(self.output_name, input_feed={self.input_name: iimage_numpy})
        input_feed = self.get_input_feed(self.input_name, image_numpy)
        scores, boxes = self.onnx_session.run(self.output_name, input_feed=input_feed)
        return scores,boxes  ## type:numpy

class Post(object):
    def __init__(self,scores,boxes):
        self.scores = scores
        self.boxes = boxes

    def get_final_result(self, top_k=10, prob_threshold=0.01):
        scores = to_tensor(self.scores[0])
        boxes = to_tensor(self.boxes[0])
        nms_st = time.time()
        # print("scores.size()=",scores.size())
        # print("boxes.size()=",boxes.size())
        boxes = boxes.to(torch.device("cpu"))
        scores = scores.to(torch.device("cpu"))
        picked_box_probs = []
        picked_labels = []
        for class_index in range(1,scores.size(1)):
            probs = scores[:, class_index]
            mask = probs > prob_threshold
            probs = probs[mask]
            if probs.size(0) == 0:
                continue
            subset_boxes = boxes[mask, :]
            box_probs = torch.cat([subset_boxes, probs.reshape(-1, 1)], dim=1) # N * 5
            print("box_probs_before.size()=",box_probs.size())
            ## hard_nms效果很烂
            # box_probs = self.hard_nms(box_probs,
            #                           iou_threshold=0.5,
            #                           top_k=top_k,
            #                           )
            ## soft_nms效果好一点,但是还是比较差劲
            box_probs = self.soft_nms(box_probs,0.5)
            print("box_probs_after.size()=", box_probs.size())
            picked_box_probs.append(box_probs)
            picked_labels.extend([class_index] * box_probs.size(0))
        if not picked_box_probs:
            return torch.tensor([]), torch.tensor([]), torch.tensor([])
        picked_box_probs = torch.cat(picked_box_probs)
        picked_box_probs[:, 0] *= 1920 ## W
        picked_box_probs[:, 1] *= 1080 ## H
        picked_box_probs[:, 2] *= 1920 ## W
        picked_box_probs[:, 3] *= 1080 ## H
        nms_et = time.time()-nms_st
        print('nms time cost {} ms:'.format(nms_et*1000))
        return picked_box_probs[:, :4], torch.tensor(picked_labels), picked_box_probs[:, 4]

    def soft_nms(self,box_scores, score_threshold, sigma=0.5, top_k=10):
        picked_box_scores = []
        while box_scores.size(0) > 0:
            max_score_index = torch.argmax(box_scores[:, 4])
            cur_box_prob = torch.tensor(box_scores[max_score_index, :])
            picked_box_scores.append(cur_box_prob)
            if len(picked_box_scores) == top_k > 0 or box_scores.size(0) == 1:
                break
            cur_box = cur_box_prob[:-1]
            box_scores[max_score_index, :] = box_scores[-1, :]
            box_scores = box_scores[:-1, :]
            ious = self.iou_of(cur_box.unsqueeze(0), box_scores[:, :-1])
            box_scores[:, -1] = box_scores[:, -1] * torch.exp(-(ious * ious) / sigma)
            box_scores = box_scores[box_scores[:, -1] > score_threshold, :]
        if len(picked_box_scores) > 0:
            return torch.stack(picked_box_scores)
        else:
            return torch.tensor([])


    def hard_nms(self,box_scores, iou_threshold, top_k=-1, candidate_size=100):
        """
        Args:
            box_scores (N, 5): boxes in corner-form and probabilities.
            iou_threshold: intersection over union threshold.
            top_k: keep top_k results. If k <= 0, keep all the results.
            candidate_size: only consider the candidates with the highest scores.
        Returns:
             picked: a list of indexes of the kept boxes
        """
        scores = box_scores[:, -1]
        boxes = box_scores[:, :-1]
        picked = []

        _, indexes = scores.sort(descending=True)
        indexes = indexes[:candidate_size]
        while len(indexes) > 0:
            current = indexes[0]
            picked.append(current.item())
            if 0 < top_k == len(picked) or len(indexes) == 1:
                break
            current_box = boxes[current, :]
            indexes = indexes[1:]
            rest_boxes = boxes[indexes, :]
            iou = self.iou_of(
                rest_boxes,
                current_box.unsqueeze(0),
            )
            indexes = indexes[iou <= iou_threshold]

        return box_scores[picked, :]

    def iou_of(self,boxes0, boxes1, eps=1e-5):
        """Return intersection-over-union (Jaccard index) of boxes.

        Args:
            boxes0 (N, 4): ground truth boxes.
            boxes1 (N or 1, 4): predicted boxes.
            eps: a small number to avoid 0 as denominator.
        Returns:
            iou (N): IoU values.
        """
        overlap_left_top = torch.max(boxes0[..., :2], boxes1[..., :2])
        overlap_right_bottom = torch.min(boxes0[..., 2:], boxes1[..., 2:])

        overlap_area = self.area_of(overlap_left_top, overlap_right_bottom)
        area0 = self.area_of(boxes0[..., :2], boxes0[..., 2:])
        area1 = self.area_of(boxes1[..., :2], boxes1[..., 2:])
        return overlap_area / (area0 + area1 - overlap_area + eps)

    def area_of(self,left_top, right_bottom) -> torch.Tensor:
        """Compute the areas of rectangles given two corners.

        Args:
            left_top (N, 2): left top corner.
            right_bottom (N, 2): right bottom corner.

        Returns:
            area (N): return the area.
        """
        hw = torch.clamp(right_bottom - left_top, min=0.0)
        return hw[..., 0] * hw[..., 1]
"""
image_size = 300
image_mean = np.array([127, 127, 127])  # RGB layout
image_std = 128.0
"""

class PredictionTransform:
    def __init__(self, size, mean=0.0, std=1.0):
        self.transform = Compose([
            Resize(size),
            SubtractMeans(mean),
            lambda img, boxes=None, labels=None: (img / std, boxes, labels),
            ToTensor()
        ])

    def __call__(self, image):
        image, _, _ = self.transform(image)
        return image

if __name__ == "__main__":
    image_dir = "/home/zhex/test_result/tag_device"
    transform = PredictionTransform(config.image_size, config.image_mean, config.image_std)
    onnx_path = "onnx_model/electronic_tag.onnx"
    model = ONNXModel(onnx_path)

    for name in os.listdir(image_dir):
        image_path = os.path.join(image_dir,name)
        image_bgr = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image_bgr,cv2.COLOR_BGR2RGB)
        # image = image_deal(image_path)
        image = transform(image_rgb)
        image = image.unsqueeze(0)
        scores,boxes = model.forward(to_numpy(image))
        print("scores.shape=",scores.shape) #scores.shape = (1, 3000, 2)
        print("boxes.shape=",boxes.shape) #boxes.shape = (1, 3000, 4)
        st = time.time()
        post_dealing = Post(scores,boxes)
        boxes, labels, prob = post_dealing.get_final_result()
        print("post dealing time {} ms".format(1000 *(time.time() - st)))
        # print("boxes,labels,prob=",boxes,labels,prob)
        boxes = boxes.numpy()
        if boxes.shape[0] == 0:
            continue
        for i in range(boxes.shape[0]):
            box = boxes[i, :]
            cv2.rectangle(image_bgr, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)
            # label = f"{class_names[labels[i]]}: {prob[i]:.2f}"
            cv2.imshow("image",image_bgr)
            cv2.waitKey(1000)
            # cv2.imwrite("eval_results" + "/" +"{}".format(name),image_bgr)