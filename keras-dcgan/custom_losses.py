#!/usr/bin/env python
# coding=utf-8

import numpy as np 
import tensorflow as tf

def alpha_loss(y_true, y_pred):
    '''
    Deep Image Matting, alpha loss
    :param y_true:
    :param y_pred:
    :return:
    '''
    diff = y_true - y_pred
    loss = tf.sqrt(tf.square(diff) + 1e-12)
    return loss

def composition_loss(y_true, y_pred):
    '''
    Deep Image Matting, composition loss
    :param y_true:
    :param y_pred:
    :return:
    '''
    diff = y_true - y_pred
    loss = tf.sqrt(tf.square(diff) + 1e-12) / 255.0
    return loss 

