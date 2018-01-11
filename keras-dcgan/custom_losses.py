#!/usr/bin/env python
# coding=utf-8

import numpy as np 
import tensorflow as tf
from keras import backend as keras

def alpha_loss_tf(y_true, y_pred):
    '''
    Deep Image Matting, alpha loss
    :param y_true:
    :param y_pred:
    :return:
    '''
    diff = y_true - y_pred
    loss = tf.sqrt(tf.square(diff) + 1e-12)
    return loss

def composition_loss_tf(y_true, y_pred):
    '''
    Deep Image Matting, composition loss
    :param y_true:
    :param y_pred:
    :return:
    '''
    diff = y_true - y_pred
    loss = tf.sqrt(tf.square(diff) + 1e-12) / 255.0
    return loss 

def mean_composition_error(y_true, y_pred):
    diff = y_true - y_pred
    loss = keras.mean(keras.sqrt(keras.square(diff)) / 255.0, axis=-1)
    return loss