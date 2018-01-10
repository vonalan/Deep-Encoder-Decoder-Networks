#!/usr/bin/env python
# coding=utf-8

import numpy as np
import tensorflow as tf
from keras.layers import Dense, Input, Lambda
from keras.models import Model
from keras import backend as keras
from keras.activations import softmax, tanh
from keras.engine.topology import Layer


def single_attention_block(inputs):
    x = inputs

    # tensorflow like
    w = tf.get_variable('ab_0_w', shape=[1024, 1])
    x = tf.matmul(x, w)
    # x = tf.exp(x)
    # x = tf.div(x, tf.reduce_sum(x))
    x = tf.nn.softmax(x)

    x = tf.matmul(tf.transpose(x), inputs)

    return x


def cascaded_attention_block(inputs):
    x = inputs

    # tensorflow like
    w = tf.get_variable('ab_1_w', shape=[1024, 1024])
    b = tf.get_variable('ab_1_b', shape=[1024])
    x = tf.add(tf.matmul(x, w), b)
    x = tf.nn.tanh(x)

    return x


class SingleAttentionBlock(Layer):
    def __init__(self, output_dim, **kwargs):
        assert output_dim == 1
        self.output_dim = output_dim
        super(SingleAttentionBlock, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(SingleAttentionBlock, self).build(input_shape)

    def call(self, x):
        inputs = x
        x = keras.dot(x, self.kernel)
        # x = softmax(x)
        x = keras.exp(x)
        x = x / keras.sum(x)
        x = keras.dot(keras.transpose(x), inputs)
        return x

    def compute_output_shape(self, input_shape):
        return (self.output_dim, input_shape[1])


class CascadedAttentionBlock(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(CascadedAttentionBlock, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        self.bias = self.add_weight(name='kernel',
                                    shape=(self.output_dim,),
                                    initializer='zero',
                                    trainable=True)
        super(CascadedAttentionBlock, self).build(input_shape)

    def call(self, x):
        x = keras.bias_add(keras.dot(x, self.kernel), self.bias)
        x = tanh(x)
        return x

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

def attention_test_tf():
    inputs = tf.placeholder(tf.float32, shape=[None,1024])
    x = inputs

    x = single_attention_block(x)
    x = cascaded_attention_block(x)

    outputs = x

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    for i in range(10):
        num_samples = np.random.randint(1,10,(1,))[0]
        x_batch = np.random.random((num_samples, 1024))
        y_batch = sess.run(outputs, feed_dict={inputs: x_batch})
        print(x_batch.shape, y_batch.shape)

    sess.close()


def attention_test_keras():
    inputs = Input(shape=(1024,))
    x = inputs

    x = SingleAttentionBlock(1)(x)
    x = CascadedAttentionBlock(1024)(x)

    outputs = x
    model = Model(inputs, outputs)
    model.summary()

    '''Keras is disgusting!!!'''
    # TODO: model.output.shape != sess.run(model.output).shape ???
    for i in range(1):
        num_samples = np.random.randint(1,10,(1,))[0]
        x_batch = np.random.random((num_samples, 1024))
        y_batch = model.predict(x_batch)
        print(x_batch.shape, y_batch.shape)
        print(y_batch)
        print(y_batch.mean(axis=0))

if __name__ == '__main__':
    # attention_test_tf()
    attention_test_keras()