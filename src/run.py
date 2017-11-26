# -*- coding: utf-8 -*-

import os
import itertools

import numpy as np
import tensorflow as tf

import reduced_vgg16 as vgg16

train_batch_size = 2

def build_train_graph(train_batch_size=1, image_size=320):
    raw_RGBs = tf.placeholder(tf.float32, shape=[train_batch_size, image_size, image_size, 3])
    training = tf.placeholder(tf.bool)
    _, _, pred_RGBs = vgg16.build_reduced_vgg16_graph(raw_RGBs, training)

    c_diff = tf.sqrt(tf.square(pred_RGBs - raw_RGBs) + 1e-12) / 255.0
    loss = tf.reduce_sum(c_diff)
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-2).minimize(loss) # learning_rate=1e-5
    return raw_RGBs, pred_RGBs, loss, optimizer, training

def load_images():
    from PIL import Image
    image_dir = '../images'

    RGBs = []
    for i, image in enumerate(os.listdir(image_dir)):
        if i >= train_batch_size: break
        image_path = os.path.join(image_dir, image)
        RGBs.append(np.array(Image.open(image_path).resize((320, 320))).astype(np.int))
    return RGBs

def main():
    raw_RGBs, pred_RGBs, loss, optimizer, training = build_train_graph(train_batch_size, image_size=320)
    tf.summary.image('raw_RGBs', raw_RGBs, max_outputs=5)
    tf.summary.image('pred_RGBs', pred_RGBs, max_outputs=5)
    tf.summary.scalar('loss', loss)
    merger = tf.summary.merge_all()
    init = tf.global_variables_initializer()
    summary_writer = tf.summary.FileWriter('../temp/logs/train', tf.get_default_graph())

    config = tf.ConfigProto(device_count={"CPU": 24, "GPU": 0})
    # sess = tf.Session()
    sess = tf.Session()
    sess.run(init)

    images = load_images()
    for i in itertools.count():
        # images = random.shuffle(images)
        _, summary = sess.run([optimizer, merger], feed_dict={raw_RGBs: images, training: False})
        summary_writer.add_summary(summary, i)
        print('epoch: %d', i)

if __name__ == '__main__':
    main()