import numpy as np
import tensorflow as tf

def conv_2d(x, ksize, bsize, stride=[1,1,1,1],
           activation=None,
           use_batch_normalization=False,
           name=''):
    with tf.variable_scope(name) as scope:
        kernel = tf.Variable(tf.truncated_normal(ksize, dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(x, kernel, stride, padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=bsize, dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        out = tf.nn.relu(out, name=name)
        return [kernel, biases], out

def pool_2x2(x, ksize=[1,2,2,1], strides=[1,2,2,1], name=''):
    with tf.device('/gpu:0'):
        return tf.nn.max_pool_with_argmax(x,
                                          ksize=[1, 2, 2, 1],
                                          strides=[1, 2, 2, 1],
                                          padding='SAME',
                                          name=name)

def build_vgg16_graph(x):
    conv_1_1, conv_1_1_params = conv_2d(x, [3, 3, 3, 64], [64], name='conv_1_1')
    conv_1_2, conv_1_2_params = conv_2d(conv_1_1, [3, 3, 64, 64], [64], name='conv_1_2')
    pool_1, pool_1_argmax = pool_2x2(conv_1_2, name='pool1')
    # en_parameters.extend(conv_1_1_params, conv_1_2_params)
    # pool_parameters.extend(pool_1_argmax)

    conv_2_1, conv_2_1_params = conv_2d(pool_1, [3, 3, 64, 128], [128], name='conv_1_1')
    conv_2_2, conv_2_2_params = conv_2d(conv_2_1, [3, 3, 128, 128], [128], name='conv_1_2')
    pool_2, pool_2_argmax = pool_2x2(conv_2_2, name='pool2')

    conv_3_1, conv_3_1_params = conv_2d(pool_2, [3, 3, 128, 256], [256], name='conv_3_1')
    conv_3_2, conv_3_2_params = conv_2d(conv_3_1, [3, 3, 256, 256], [256], name='conv_3_2')
    conv_3_3, conv_3_3_params = conv_2d(conv_3_2, [3, 3, 256, 256], [256], name='conv_3_3')
    pool_3, pool_3_argmax = pool_2x2(conv_3_3, name='pool3')

    conv4_1, conv4_1_params = conv_2d(pool_3, [3, 3, 256, 512], [512], name='conv4_1')
    conv4_2, conv4_2_params = conv_2d(conv4_1, [3, 3, 512, 512], [512], name='conv4_2')
    conv4_3, conv4_3_params = conv_2d(conv4_2, [3, 3, 512, 512], [512], name='conv4_3')
    pool_4, pool_4_argmax = pool_2x2(conv4_3, name='pool4')

    conv5_1, conv5_1_params = conv_2d(pool_4, [3, 3, 512, 512], [512], name='conv5_1')
    conv5_2, conv5_2_params = conv_2d(conv5_1, [3, 3, 512, 512], [512], name='conv5_2')
    conv5_3, conv5_3_params = conv_2d(conv5_2, [3, 3, 512, 512], [512], name='conv5_3')
    pool_5, pool_5_argmax = pool_2x2(conv5_3, name='pool5')