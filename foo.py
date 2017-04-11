import tensorflow as tf
import numpy as np


NUM_DIMS = 100
NUM_HID = 10
BATCH_SIZE = 20


x = tf.placeholder(tf.float32, [None, NUM_DIMS])

W = tf.Variable(tf.random_normal([NUM_DIMS, NUM_HID]), name='W')
b = tf.Variable(tf.zeros(NUM_HID))
poshidprobs_op = tf.sigmoid(tf.matmul(x, W) - b)
posprods_op = tf.matmul(tf.transpose(x), poshidprobs_op)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    batch = np.random.rand(BATCH_SIZE, NUM_DIMS)
    out = sess.run(posprods_op, feed_dict={x: batch})
    print(out)
