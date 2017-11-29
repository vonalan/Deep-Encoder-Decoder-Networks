# main

import numpy as np
import tensorflow as tf

def pool(x, kernel_size=[1, 2, 2, 1], strides=[1,2,2,1], name='pool'):
    # TODO: how to implemente tf.nn.max_pool_with_argmax with cpu?
    with tf.variable_scope(name) as scope:
        with tf.device('/gpu:0'):
            return tf.nn.max_pool_with_argmax(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                              padding='SAME', name=scope)

def unpool_2x2(x, pool_argmax, output_shape):
    pass


def conv2d(x, kernel_sizes, bias_sizes, stride=[1,1,1,1],
           activation=None,
           use_batch_normalization=False,
           name='conv'):
    with tf.variable_scope(name) as scope:
        kernel = tf.Variable(tf.truncated_normal(kernel_sizes, dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(x, kernel, stride, padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=bias_sizes, dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        out = tf.nn.relu(out, name=name)
        return [kernel, biases], out

def conv2d_transpose(x, kernel_sizes, bias_sizes, stride=[1,1,1,1],
                     output_shape=None,
                     activation=None,
                     use_batch_normalization=False,
                     name='deconv'):
    with tf.variable_scope(name) as scope:
        kernel = tf.Variable(tf.truncated_normal(kernel_sizes, dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        x_shape = tf.shape(x)
        output_shape = tf.stack([x_shape[0], 2 * x_shape[1], 2 * x_shape[2], kernel_sizes[2]])
        conv = tf.nn.conv2d_transpose(x, kernel, output_shape, stride, padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=bias_sizes, dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        out = tf.nn.relu(out, name=name)
        return [kernel, biases], out

def build_discriminator():
    x = tf.placeholder(tf.float32, [64, 320, 320, 3])
    _, y = conv2d(x, [5,5,3,64], [64], [1,2,2,1], name='conv2d')
    _, z = conv2d_transpose(y, [5,5,3,64], [3], [1,2,2,1], [64,320,320,3], name='deconv2d')
    return y, z

if __name__ == '__main__':
    y, z = build_discriminator()
    z_shape = tf.shape(z)

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    print(sess.run(z_shape))