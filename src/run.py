import numpy as np
import tensorflow as tf

train_batch_size,image_size = 1, 10
# b_trimap = tf.zeros([train_batch_size,image_size,image_size, 1])
b_trimap = tf.expand_dims(tf.eye(image_size) * 128, axis=0)
b_trimap = tf.expand_dims(b_trimap, axis=3)

wl = tf.where(tf.equal(b_trimap,128),
              tf.fill([train_batch_size,image_size,image_size,1],10.),
              tf.fill([train_batch_size,image_size,image_size,1],-1.))
unknown_region_size = tf.reduce_sum(wl)

sess = tf.Session()
res = sess.run(wl)
print(res.shape)
print(np.reshape(res, (-1, 1, image_size, image_size)))