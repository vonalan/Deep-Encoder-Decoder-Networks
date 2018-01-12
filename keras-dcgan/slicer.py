import os 
import sys
import time

import numpy as np
import cv2


def create_video_lists(video_dir):
    video_list = []
    for item in os.walk(video_dir):
        for file_name in item[2]:
            if file_name.split('.')[-1] not in ['avi']:
                continue
            video_path = os.path.join(item[0], file_name)
            video_list.append(video_path)
            # print(video_path)
    print(len(video_list))
    return video_list

def segment(segment_list, image_dir, kernel=14, stride=1):
    # TODO: temporal relevant!!!
    images_list = os.listdir(image_dir)
    images_list = sorted(images_list, key=lambda x: int(x.split('.')[0]), reverse=False)
    images_list = [os.path.join(image_dir, image) for image in images_list]

    result = []

    # TODO: how to choose sizes of kernerl and stride???
    stride = int((len(images_list) - 18) * (1/float(13))) + 1
    stride = min(int(kernel/2), stride)

    flag = 1
    sidx, eidx = 0, 0
    while flag:
        eidx = sidx + kernel
        if eidx > len(images_list):
            gap = eidx - len(images_list)
            sidx -= gap
            eidx -= gap
            flag = 0
        assert eidx - sidx == kernel

        temp = images_list[sidx:eidx]
        result.append(temp)
        sidx += stride
    segment_list.extend(result)
    print('%4d %4d %4d %4d'%(len(images_list), stride, len(result), len(result[-1])))

if __name__ == '__main__':
    video_dir = '../hmdb51_org'
    image_dir = '../hmdb51_org_images'

    video_list = create_video_lists(video_dir)

    segment_list = []
    for video_path in video_list:
        sub_image_dir = video_path.replace(os.path.basename(video_dir), os.path.basename(image_dir))
        # print(sub_image_dir)
        segment(segment_list, sub_image_dir)
        # print(len(segment_list))