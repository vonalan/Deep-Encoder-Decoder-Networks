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
        '''

        :param input_shape: [1, None, 1024]
        :return:
        '''
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[2], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(SingleAttentionBlock, self).build(input_shape)

    def call(self, x):
        '''

        :param x:[1,None, 1024]
        :return: [1,1,1024]
        '''
        x = keras.squeeze(x, axis=0) # [None,1024]
        inputs = x
        x = keras.dot(x, self.kernel)
        # x = softmax(x)
        x = keras.exp(x)
        x = x / keras.sum(x)
        x = keras.dot(keras.transpose(x), inputs)
        x = keras.expand_dims(x, axis=0)
        return x

    # TODO:
    def compute_output_shape(self, input_shape):
        # return (input_shape[0], 1, input_shape[2])
        return (1, 1, 1024)

class CascadedAttentionBlock(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(CascadedAttentionBlock, self).__init__(**kwargs)

    def build(self, input_shape):
        '''

        :param input_shape:[1,1,1024]
        :return:
        '''
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[2], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        self.bias = self.add_weight(name='kernel',
                                    shape=(self.output_dim,),
                                    initializer='zero',
                                    trainable=True)
        super(CascadedAttentionBlock, self).build(input_shape)

    def call(self, x):
        '''

        :param x:[1,1,1024]
        :return: [1,1,1024]
        '''
        x = keras.squeeze(x, axis=0) # [1,1024]
        x = keras.bias_add(keras.dot(x, self.kernel), self.bias)
        x = tanh(x)
        x = keras.expand_dims(x, axis=0)
        return x

    def compute_output_shape(self, input_shape):
        # return (input_shape[0], self.output_dim)
        return (1,1,1024)

def attention_test_tf():
    inputs = tf.placeholder(tf.float32, shape=[None, 1024])
    labels = tf.placeholder(tf.float32, shape=[None, 1024])

    x = inputs
    x = single_attention_block(x)
    x = cascaded_attention_block(x)
    outputs = x

    loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=outputs)
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-2).minimize(loss)
    accuracy = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(labels, axis=1), tf.argmax(outputs, axis=1)), tf.int32))

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    for i in range(1000000):
        num_samples = np.random.randint(1,10,(1,))[0]
        x_batch = np.random.random((num_samples, 1024))
        t_batch = np.zeros((1,1024))
        t_batch[0,np.random.randint(1,10,(1,))[0]] = 1
        [y_batch, cost, _, acc] = sess.run([outputs, loss, optimizer, accuracy], feed_dict={inputs: x_batch, labels: t_batch})
        print(np.argmax(y_batch, axis=1), cost, acc)

    sess.close()


def attention_test_keras():
    inputs = Input(batch_shape=(1,None,1024))
    x = inputs

    x = SingleAttentionBlock(1)(x)
    x = CascadedAttentionBlock(1024)(x)

    outputs = x
    model = Model(inputs, outputs)
    model.summary()

    '''Keras is disgusting!!!'''
    # TODO: model.output.shape != sess.run(model.output).shape ???
    for i in range(10):
        num_samples = np.random.randint(1,10,(1,))[0]
        x_batch = np.random.random((1, num_samples, 1024))
        y_batch = model.predict(x_batch)
        print(x_batch.shape, y_batch.shape)
        print(y_batch)

if __name__ == '__main__':
    # attention_test_tf()
    attention_test_keras()