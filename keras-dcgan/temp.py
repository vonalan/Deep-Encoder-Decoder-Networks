import os 
import sys
import re
import math
import random 
import argparse

import numpy as np
import cv2

def embed_image(bg_img, fg_img):
    bg_h, bg_w = bg_img.shape[:2]
    fg_h, fg_w = fg_img.shape[:2]

    cv2.rectangle(fg_img, (0,0), (fg_h-1, fg_w-1), (255,0,255), thickness=1)

    # rotate && resize
    offset_y = random.randint(0, bg_h)
    offset_x = random.randint(0, bg_w)
    theta = random.randint(0, 360)

    # offset_y = bg_h / 2
    # offset_x = bg_w / 2
    # theta = 45

    alpha = ((theta + 45) % 360) / 180.0 * math.pi
    fg_diag = (fg_h ** 2 + fg_w ** 2) ** 0.5 * 0.5

    # y_test = fg_diag * math.sin(45/180.0*math.pi)
    # x_test = fg_diag * math.cos(45/180.0*math.pi)
    #
    # left_y = bg_h - offset_y
    # left_x = bg_w - offset_x
    # proj_y = fg_diag * math.sin(alpha)
    # proj_x = fg_diag * math.cos(alpha)

    # TODO: devided by zero when ((theta + 45) % 90 == 0)
    scale_x = min(offset_y, bg_h - offset_y) / (fg_diag * abs(math.sin(alpha)))
    scale_y = min(offset_x, bg_w - offset_x) / (fg_diag * abs(math.cos(alpha)))
    scale = min(min(scale_y, scale_x), 1)

    print(offset_y, offset_x, theta, scale)

    M = cv2.getRotationMatrix2D((fg_w/2,fg_h/2), theta, scale=scale)
    fg_img = cv2.warpAffine(fg_img, M, (fg_h,fg_w))
    cv2.imwrite('xxx.jpg', fg_img)

if __name__ == '__main__':
    bg_image_path = '../images/1.jpg'
    fg_image_path = '../images/3.jpg'
    bg_image = cv2.imread(bg_image_path)
    fg_image = cv2.imread(fg_image_path)
    embed_image(bg_image, fg_image)