import os 
import sys
import re

import numpy as np
import cv2
import tensorflow as tf
# import tensorflow.gfile as gfile

def create_video_lists(video_dir, split_dir, sround=1):
    if not tf.gfile.Exists(video_dir):
        tf.logging.error("Video directory '" + video_dir + "' not found. ")
        return None
    result = {}
    sub_dirs = [x[0] for x in tf.gfile.Walk(video_dir)]
    print(sub_dirs)
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
        tf.logging.info("Looking for images in '" + dir_name + "'")
        for extension in extensions:
            file_glob = os.path.join(video_dir, dir_name, '*.' + extension)
            file_list.extend(tf.gfile.Glob(file_glob))
        if not file_list:
            tf.logging.warning('No files found')
            continue
        if len(file_list) < 100:
            tf.logging.warning(
                'WARNING: Folder has less than 100 videos, which may cause issues.')
        elif len(file_list) > 100:
            tf.logging.warning(
                'WARNING: Folder {} has more than {} images. Some images will '
                'never be selected.'.format(dir_name, 100))
        label_name = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower())

        training_videos = []
        testing_videos = []
        validation_videos = []

        # print(split_dir, dir_name)
        split_file_name = os.path.join(split_dir, dir_name + '_test_split%d.txt'%(sround))
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

# split images of training video lists
def create_image_list(video_dir, image_dir, video_lists, sround=1):
    image_list = []
    label_list = []
    for class_name, video_list in video_lists.items():
        for video_path in video_list:
            sub_image_dir = video_path.replace(os.path.basename(video_dir), os.path.basename(image_dir))
            images = os.listdir(sub_image_dir)
            images = [os.path.join(sub_image_dir, image) for image in images]
            image_list.extend(images)
            labels = [classes.index(class_name)] * len(images)
            label_list.extend(labels)
    assert len(image_list) == len(label_list)

    def func(image_list, label_list, filename):
        assert len(image_list) == len(label_list)
        with open(filename, 'w') as f:
            for image, label in zip(image_list, label_list):
                line = '%s %s\n'%(image, label)
                f.write(line)

    import random
    for i in range(3):
        idx = [i for i in range(len(image_list))]
        idx = random.shuffle(idx)

        # training set | 1
        end1 = int(len(image_list) * 0.8)
        train_idx = idx[:end1]
        train_images = image_list[train_idx]
        train_labels = label_list[train_idx]

        # validation set | not used | 0
        end2 = int(len(image_list) * 0.9)
        valid_idx = idx[end1:end2]
        valid_images = image_list[valid_idx]
        valid_labels = label_list[valid_idx]

        # test set
        test_idx = idx[end2:]
        test_images = image_list[test_idx]
        test_labels = label_list[test_idx]


if __name__ == "__main__":
    pass