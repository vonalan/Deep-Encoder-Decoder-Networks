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
        return out, [kernel, biases]

def pool_2x2(x, ksize=[1,2,2,1], strides=[1,2,2,1], name=''):
    with tf.device('/gpu:0'):
        return tf.nn.max_pool_with_argmax(x,
                                          ksize=[1, 2, 2, 1],
                                          strides=[1, 2, 2, 1],
                                          padding='SAME',
                                          name=name)

def deconv_2d(x, ksize, bsize, stride=[1,1,1,1], activation=None, use_batch_normalization=False, name=''):
    with tf.variable_scope(name) as scope:
        kernel = tf.Variable(tf.truncated_normal(ksize, dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(x, kernel, stride, padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=bsize, dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        if use_batch_normalization:
            out = tf.layers.batch_normalization(out)
        out = tf.nn.relu(out, name=name)
        return out, [kernel, biases]

def unpool(pool, ind, ksize=[1, 2, 2, 1], scope='unpool'):
    # the train_batch_size must be appointed.
    with tf.variable_scope(scope):
        input_shape = pool.get_shape().as_list()
        output_shape = (input_shape[0], input_shape[1] * ksize[1], input_shape[2] * ksize[2], input_shape[3])

        flat_input_size = np.prod(input_shape)
        flat_output_shape = [output_shape[0], output_shape[1] * output_shape[2] * output_shape[3]]

        pool_ = tf.reshape(pool, [flat_input_size])
        batch_range = tf.reshape(tf.range(output_shape[0], dtype=ind.dtype), shape=[input_shape[0], 1, 1, 1])
        b = tf.ones_like(ind) * batch_range
        b = tf.reshape(b, [flat_input_size, 1])
        ind_ = tf.reshape(ind, [flat_input_size, 1])
        ind_ = tf.concat([b, ind_], 1)

        ret = tf.scatter_nd(ind_, pool_, shape=flat_output_shape)
        ret = tf.reshape(ret, output_shape)
    return ret, None

def build_vgg16_graph(x, _):
    en_parameters = []
    pool_parameters = []

    conv_1_1, conv_1_1_params = conv_2d(x, [3, 3, 3, 64], [64], name='conv_1_1')
    conv_1_2, conv_1_2_params = conv_2d(conv_1_1, [3, 3, 64, 64], [64], name='conv_1_2')
    pool_1, pool_1_argmax = pool_2x2(conv_1_2, name='pool1')
    en_parameters.extend([conv_1_1_params, conv_1_2_params])
    pool_parameters.extend([pool_1_argmax])

    conv_2_1, conv_2_1_params = conv_2d(pool_1, [3, 3, 64, 128], [128], name='conv_2_1')
    conv_2_2, conv_2_2_params = conv_2d(conv_2_1, [3, 3, 128, 128], [128], name='conv_2_2')
    pool_2, pool_2_argmax = pool_2x2(conv_2_2, name='pool2')
    en_parameters.extend([conv_2_1_params, conv_2_2_params])
    pool_parameters.extend([pool_2_argmax])

    conv_3_1, conv_3_1_params = conv_2d(pool_2, [3, 3, 128, 256], [256], name='conv_3_1')
    conv_3_2, conv_3_2_params = conv_2d(conv_3_1, [3, 3, 256, 256], [256], name='conv_3_2')
    conv_3_3, conv_3_3_params = conv_2d(conv_3_2, [3, 3, 256, 256], [256], name='conv_3_3')
    pool_3, pool_3_argmax = pool_2x2(conv_3_3, name='pool3')
    en_parameters.extend([conv_3_1_params, conv_3_2_params, conv_3_3_params])
    pool_parameters.extend([pool_3_argmax])

    conv_4_1, conv_4_1_params = conv_2d(pool_3, [3, 3, 256, 512], [512], name='conv_4_1')
    conv_4_2, conv_4_2_params = conv_2d(conv_4_1, [3, 3, 512, 512], [512], name='conv_4_2')
    conv_4_3, conv_4_3_params = conv_2d(conv_4_2, [3, 3, 512, 512], [512], name='conv_4_3')
    pool_4, pool_4_argmax = pool_2x2(conv_4_3, name='pool4')
    en_parameters.extend([conv_4_1_params, conv_4_2_params, conv_4_3_params])
    pool_parameters.extend([pool_4_argmax])

    conv_5_1, conv_5_1_params = conv_2d(pool_4, [3, 3, 512, 512], [512], name='conv_5_1')
    conv_5_2, conv_5_2_params = conv_2d(conv_5_1, [3, 3, 512, 512], [512], name='conv_5_2')
    conv_5_3, conv_5_3_params = conv_2d(conv_5_2, [3, 3, 512, 512], [512], name='conv_5_3')
    pool_5, pool_5_argmax = pool_2x2(conv_5_3, name='pool5')
    en_parameters.extend([conv_5_1_params, conv_5_2_params, conv_5_3_params])
    pool_parameters.extend([pool_5_argmax])

    conv_6_1, conv_6_params = conv_2d(pool_5, [7,7,512,4096], [4096], name='conv_6_1')

    deconv_6, deconv_6_params = deconv_2d(conv_6_1, [1,1,4096, 512], [512], name='deconv_6')

    deconv_5_1, _ = unpool(deconv_6, pool_parameters[-1])
    deconv_5_2, _ = deconv_2d(deconv_5_1, [5,5,512,512], [512], name='deconv_5_2')

    deconv_4_1, _ = unpool(deconv_5_2, pool_parameters[-2])
    deconv_4_2, _ = deconv_2d(deconv_4_1, [5, 5, 512, 256], [256], name='deconv_4_2')

    deconv_3_1, _ = unpool(deconv_4_2, pool_parameters[-3])
    deconv_3_2, _ = deconv_2d(deconv_3_1, [5, 5, 256, 128], [128], name='deconv_3_2')

    deconv_2_1, _ = unpool(deconv_3_2, pool_parameters[-4])
    deconv_2_2, _ = deconv_2d(deconv_2_1, [5, 5, 128, 64], [64], name='deconv_2_2')

    deconv_1_1, _ = unpool(deconv_2_2, pool_parameters[-5])
    deconv_1_2, _ = deconv_2d(deconv_1_1, [5, 5, 64, 64], [64], name='deconv_1_2')

    out, _ = deconv_2d(deconv_1_2, [5,5,64,3], [3], name='out')
    return en_parameters, None, out