#!/usr/bin/env python
# coding=utf-8

import numpy as np
# import tensorflow as tf
from keras.layers import Dense,Input, Lambda
from keras.models import Model
from keras import backend as keras
from keras.activations import softmax, tanh
from keras.engine.topology import Layer

def single_attention_block(inputs):
    x = inputs

    # keras like
    x = Dense(1, use_bias=False)(x)
    # x = keras.exp(x)
    # x = x / keras.sum(x)
    x = softmax(x)

    # # tensorflow like
    # w = tf.get_variable('ab_0_w', shape=[1024, 1])
    # x = tf.matmul(x, w)
    # # x = tf.exp(x)
    # # x = tf.div(x, tf.reduce_sum(x))
    # x = tf.nn.softmax(x)

    return x

def cascaded_attention_block(inputs):
    # keras like
    x = inputs
    x = Dense(1024, activation=tanh, use_bias=True)(x)

    # # tensorflow like
    # w = tf.get_variable('ab_1_w', shape=[1024,1024])
    # b = tf.get_variable('ab_1_b', shape=[1024])
    # x = tf.add(tf.matmul(x, w), b)
    # x = tf.nn.tanh(x)

    return x

if __name__ == '__main__':
    inputs = Input(shape=(1024,))

    # alpha = single_attention_block(inputs)
    alpha = Lambda(single_attention_block)(inputs)

    # temp = keras.transpose(alpha) * inputs
    # temp = Lambda(lambda x,y: keras.dot(keras.transpose(x), y))((alpha, inputs))
    temp = keras.dot(keras.transpose(alpha), inputs)
    temp = Lambda(lambda x: x)(temp)

    # outputs = cascaded_attention_block(temp)
    outputs = Lambda(cascaded_attention_block)(temp)

    model = Model(inputs, outputs)
    model.summary()