import os 
import sys
import re
import math
import random 
import argparse

import numpy as np
import cv2
import tensorflow as tf
# import tensorflow.gfile as gfile

parser = argparse.ArgumentParser()
parser.add_argument('--classes_path', default='../data/hmdb51_classes.txt', type=str)
parser.add_argument('--split_dir', default='../testTrainMulti_7030_splits/')
parser.add_argument('--split_round', default='3', type=str)
parser.add_argument('--video_dir', default='../hmdb51_org/', type=str)
parser.add_argument('--image_dir', default='../hmdb51_org_images/', type=str)
args, _ = parser.parse_known_args()

def get_classes(classes_path):
    with open(classes_path, 'r') as f:
        lines = f.readlines()
    classes = []
    for line in lines:
        line = line.strip()
        classes.append(line)
    return classes
# classes = get_classes(args.classes_path)

def create_video_dicts(video_dir, split_dir, sround=1):
    if not tf.gfile.Exists(video_dir):
        tf.logging.error("Video directory '" + video_dir + "' not found. ")
        return None
    result = {}
    sub_dirs = [x[0] for x in tf.gfile.Walk(video_dir)]
    # print(sub_dirs)
    is_root_dir = True
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue
        extensions = ['avi']
        file_list = []
        dir_name = os.path.basename(sub_dir)
        if dir_name == video_dir:
            continue
        # tf.logging.info("Looking for images in '" + dir_name + "'")
        for extension in extensions:
            file_glob = os.path.join(video_dir, dir_name, '*.' + extension)
            file_list.extend(tf.gfile.Glob(file_glob))
        if not file_list:
            tf.logging.warning('No files found')
            continue
        # if len(file_list) < 100:
        #     tf.logging.warning(
        #         'WARNING: Folder has less than 100 videos, which may cause issues.')
        # elif len(file_list) > 100:
        #     tf.logging.warning(
        #         'WARNING: Folder {} has more than {} images. Some images will '
        #         'never be selected.'.format(dir_name, 100))
        label_name = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower())

        training_videos = []
        testing_videos = []
        validation_videos = []

        # print(split_dir, dir_name)
        split_file_name = os.path.join(split_dir, dir_name + '_test_split%s.txt'%(sround))
        rf = open(split_file_name)
        temp = rf.readlines()
        rf.close()
        temp = [line.strip().split() for line in temp]
        mask = {item[0]:item[1] for item in temp}

        for file_name in file_list:
            base_name = os.path.basename(file_name)
            if mask[base_name] == '1': # training set
                training_videos.append(base_name)
            elif mask[base_name] == '2': # test set
                testing_videos.append(base_name)
            elif mask[base_name] == '0': # not used | valiation set
                validation_videos.append(base_name)
            else:
                raise ValueError("the mask of video must be 0, 1 or 2! ")
        result[label_name] = {
            'dir': dir_name,
            'training': training_videos,
            'testing': testing_videos,
            'validation': validation_videos,
        }
    return result

def create_video_list(video_dir, video_dicts, category, classes):
    video_list = []
    video_dist = [0] * len(video_dicts.keys())
    for class_name, class_dict in video_dicts.items():
        dir = class_dict['dir']
        for video in class_dict[category]:
            path = os.path.join(video_dir, dir, video)
            video_list.append((path, classes.index(dir)))
            video_dist[classes.index(dir)] += 1
    return len(video_list), video_dist, video_list

def create_image_list(image_dir, video_dicts, category, classes):
    image_list = []
    image_dist = [0] * len(video_dicts.keys())
    for class_name, class_dict in video_dicts.items():
        dir = class_dict['dir']
        for video in class_dict[category]:
            path = os.path.join(image_dir, dir, video)
            images = os.listdir(path)
            images = [(os.path.join(path, image), classes.index(dir)) for image in images]
            image_list.extend(images)
            image_dist[classes.index(dir)] += len(images)
    return len(image_list), image_dist, image_list

def iter_mini_batches_for_attention(args, category, classes, batch_size=1, shuffle=True):
    assert batch_size == 1

    # TODO: shuffle, resize, crop, flip, distort, blur, ...
    video_dicts = create_video_dicts(args.video_dir, args.split_dir, sround=args.split_round)
    # _, _, image_list = create_image_list(args.image_dir, video_dicts, category)
    _,_, video_list = create_video_list(args.video_dir, video_dicts, category, classes)

    while True:
        if shuffle:
            random.seed()
            random.shuffle(video_list)
        num_batchs = int(math.ceil(len(video_list) / float(batch_size)))
        for batch in range(num_batchs):
            sidx = max(0, batch * batch_size)
            eidx = min(len(video_list), (batch + 1) * batch_size)
            num_cur_batch = eidx - sidx

            video_batch = []
            label_batch = []
            for idx in range(num_cur_batch):
                video_path = video_list[sidx + idx][0]
                image_dir = video_path.replace('hmdb51_org', 'hmdb51_org_images')
                images = [cv2.resize(cv2.imread(os.path.join(image_dir, image)), args.input_shape[:2]) for image in os.listdir(image_dir)]
                label = [0] * 51
                label[video_list[sidx + idx][0]] = 1
                video_batch.append(images)
                label_batch.append(label)
            video_batch = np.array(video_batch)
            label_batch = np.array(label_batch)
            print(video_batch.shape, label_batch.shape)
            yield video_batch, label_batch

def iter_mini_batches(args, category, num_classes, batch_size=4, shuffle=True):
    '''
    data generator for classification problems.
    :param args:
    :param category:
    :param num_classes:
    :param batch_size:
    :param shuffle:
    :return:
    '''
    # TODO: shuffle, resize, crop, flip, distort, blur, ...
    video_dicts = create_video_dicts(args.video_dir, args.split_dir, sround=args.split_round)
    _, _, image_list = create_image_list(args.image_dir, video_dicts, category)

    while True:
        if shuffle:
            random.seed()
            random.shuffle(image_list)

        num_batchs = int(math.ceil(len(image_list)/float(batch_size)))
        for batch in range(num_batchs):
            sidx = max(0, batch * batch_size)
            eidx = min(len(image_list), (batch + 1) * batch_size)
            num_cur_batch = eidx - sidx
            image_batch = np.zeros(tuple([batch_size] + list(args.input_shape)))
            label_batch = np.zeros(tuple([batch_size] + [num_classes]))
            for idx in range(num_cur_batch):
                image_batch[idx] = cv2.resize(cv2.imread(image_list[sidx + idx][0]), args.input_shape[:2])
                label_batch[idx][image_list[sidx + idx][1]] = 1
                # print(image_batch[idx].shape, label_batch[idx])
            yield image_batch, label_batch

def iter_mini_batches_for_deconvnet(args, category, num_classes, batch_size=4, shuffle=True):
    '''
    data generator for deconvet and dcgan.
    :param args:
    :param category:
    :param num_classes:
    :param batch_size:
    :param shuffle:
    :return:
    '''
    # TODO: shuffle, resize, crop, flip, distort, blur, ...
    video_dicts = create_video_dicts(args.video_dir, args.split_dir, sround=args.split_round)
    _, _, image_list = create_image_list(args.image_dir, video_dicts, category)

    while True:
        if shuffle:
            random.seed()
            random.shuffle(image_list)

        num_batchs = int(math.ceil(len(image_list)/float(batch_size)))
        for batch in range(num_batchs):
            sidx = max(0, batch * batch_size)
            eidx = min(len(image_list), (batch + 1) * batch_size)
            num_cur_batch = eidx - sidx
            image_batch = np.zeros(tuple([batch_size] + list(args.input_shape)))
            label_batch = np.zeros(tuple([batch_size] + [num_classes]))
            for idx in range(num_cur_batch):
                image_batch[idx] = cv2.resize(cv2.imread(image_list[sidx + idx][0]), args.input_shape[:2])
                label_batch[idx][image_list[sidx + idx][1]] = 1
                # print(image_batch[idx].shape, label_batch[idx])
            yield image_batch, image_batch

if __name__ == "__main__":
    pass