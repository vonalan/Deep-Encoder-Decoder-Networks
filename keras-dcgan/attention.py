#!/usr/bin/env python
# coding=utf-8

import numpy as np
import tensorflow as tf
from keras.layers import Dense,Input
from keras.models import Model
from keras import backend as keras

def single_attention_block(inputs):
    x = inputs

    # keras like
    x = Dense(1, use_bias=False)(x)
    x = x / keras.sum(x)

    # # tensorflow like
    # w = tf.get_variable('ab_0_w', shape=[1024, 1])
    # x = tf.matmul(x, w)
    # x = tf.div(x, tf.reduce_sum(x))

    return x

def cascaded_attention_block(inputs):
    # keras like
    x = inputs
    x = Dense(1024, activation='tanh', use_bias=True)(x)

    # # tensorflow like
    # w = tf.get_variable('ab_1_w', shape=[1024,1024])
    # b = tf.get_variable('ab_1_b', shape=[1024])
    # x = tf.add(tf.matmul(x, w), b)
    # x = tf.nn.tanh(x)

    return x

if __name__ == '__main__':
    inputs = Input(shape=(1024))
    alpha = single_attention_block(inputs)
    temp = alpha * inputs
    outputs = cascaded_attention_block(temp)
    model = Model(inputs, outputs)
    model.summary()