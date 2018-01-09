#!/usr/bin/env python
# coding=utf-8

import os
import math
import random

import numpy as np
import cv2

def get_image_lists(image_dir):
    image_list = os.listdir(image_dir)
    image_list = [os.path.join(image_dir, image) for image in image_list]
    image_dict = dict()

    random.seed(0)
    random.shuffle(image_list)
    idx1 = int(len(image_list) * 0.8)
    idx2 = int(len(image_list) * 0.9)
    image_dict['train'] = image_list[:idx1]
    image_dict['valid'] = image_list[idx1:idx2]
    image_dict['infer'] = image_list[idx2:]
    return image_dict

def iter_mini_batches_for_dcgan(input_image_shape, image_list, batch_size=4, shuffle=True):
    while True:
        if shuffle:
            random.seed()
            random.shuffle(image_list)

        num_batchs = int(math.ceil(len(image_list) / float(batch_size)))
        for batch in range(num_batchs):
            sidx = max(0, batch * batch_size)
            eidx = min(len(image_list), (batch + 1) * batch_size)
            num_cur_batch = eidx - sidx
            image_batch = np.zeros(tuple([batch_size] + list(input_image_shape)))
            # label_batch = np.zeros(tuple([batch_size] + [num_classes]))
            for idx in range(num_cur_batch):
                image = cv2.resize(cv2.imread(image_list[sidx + idx]), input_image_shape[:2])
                # TODO: tanh | sigmoid | relu
                image_batch[idx] = (image - (255.0/2.0)) / (255.0/2.0)
                # label_batch[idx][image_list[sidx + idx][1]] = 1
                # print(image_batch[idx].shape, label_batch[idx])
            yield batch, image_batch

def iter_mini_batches_for_deconvnet(input_image_shape, image_list, batch_size=4, shuffle=True):
    while True:
        if shuffle:
            random.seed()
            random.shuffle(image_list)

        num_batchs = int(math.ceil(len(image_list) / float(batch_size)))
        for batch in range(num_batchs):
            sidx = max(0, batch * batch_size)
            eidx = min(len(image_list), (batch + 1) * batch_size)
            num_cur_batch = eidx - sidx
            image_batch = np.zeros(tuple([batch_size] + list(input_image_shape)))
            # label_batch = np.zeros(tuple([batch_size] + [num_classes]))
            for idx in range(num_cur_batch):
                image_batch[idx] = cv2.resize(cv2.imread(image_list[sidx + idx]), input_image_shape[:2])
                # label_batch[idx][image_list[sidx + idx][1]] = 1
                # print(image_batch[idx].shape, label_batch[idx])
            yield batch, image_batch, image_batch