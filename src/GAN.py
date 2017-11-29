# main

import numpy as np
import tensorflow as tf

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

def build_convnet_graph(inputs):
    _, conv1 = conv2d(inputs, [5, 5, 3, 64], [64], [1, 2, 2, 1], name='conv1')
    _, conv2 = conv2d(conv1, [5, 5, 64, 128], [128], [1, 2, 2, 1], name='conv2')
    _, conv3 = conv2d(conv2, [5, 5, 128, 256], [256], [1, 2, 2, 1], name='conv3')
    _, conv4 = conv2d(conv3, [5, 5, 256, 512], [512], [1, 2, 2, 1], name='conv4')
    _, conv5 = conv2d(conv4, [5, 5, 512, 1024], [1024], [1, 2, 2, 1], name='conv5')
    _, deconv5 = conv2d_transpose(conv5, [5, 5, 512, 1024], [512], [1, 2, 2, 1], [10, 5, 5, 512], name='deconv5')
    _, deconv4 = conv2d_transpose(deconv5, [5, 5, 256, 512], [256], [1, 2, 2, 1], [10, 10, 10, 512], name='deconv4')
    _, deconv3 = conv2d_transpose(deconv4, [5, 5, 128, 256], [128], [1, 2, 2, 1], [10, 20, 20, 512], name='deconv3')
    _, deconv2 = conv2d_transpose(deconv3, [5, 5, 64, 128], [64], [1, 2, 2, 1], [10, 40, 40, 512], name='deconv2')
    _, deconv1 = conv2d_transpose(deconv2, [5, 5, 3, 64], [3], [1, 2, 2, 1], [10, 40, 40, 512], name='deconv1')
    return conv5, deconv1

def load_images():
    import os
    from PIL import Image
    image_dir = '../images'

    RGBs = []
    for i, image in enumerate(os.listdir(image_dir)):
        image_path = os.path.join(image_dir, image)
        RGBs.append(np.array(Image.open(image_path)).astype(np.int))
    return np.array(RGBs)

def main():
    import itertools
    import random
    raw_rgbs = tf.placeholder(tf.float32, [10, 96, 96, 3], name='raw_rgbs')
    _, pred_rgbs = build_convnet_graph(raw_rgbs)
    c_diff = tf.sqrt(tf.square(pred_rgbs - raw_rgbs) + 1e-12)/255.0
    loss = tf.reduce_sum(c_diff)
    tf.summary.image('raw_RGBs', raw_rgbs, max_outputs=2)
    tf.summary.image('pred_RGBs', pred_rgbs, max_outputs=2)
    tf.summary.scalar('loss', loss)
    optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)
    merger = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter('../temp/logs/train', tf.get_default_graph())

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    for i in itertools.count():
        images = load_images()
        print(images.shape)
        for i in itertools.count():
            random.shuffle(images)
            _, summary, rgbs = sess.run([optimizer, merger, pred_rgbs], feed_dict={raw_rgbs: images[:10,:,:,:]})
            summary_writer.add_summary(summary, i)
            print('epoch: %d', i)
            print(rgbs)

if __name__ == '__main__':
    main()