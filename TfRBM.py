import tensorflow as tf
import numpy as np
from PIL import Image
from util import tile_raster_images


def RBM(batch_gen, numbatches, batch_size, numdims, numhid, params, write_fn):
    activation = params['activation']
    noise = True if 'noise' in params else False
    epsilonw = params['epsilonw']
    epsilonvb = params['epsilonvb']
    epsilonhb = params['epsilonhb']

    weightcost = params['weightcost']

    initialmomentum = params['initialmomentum']
    finalmomentum = params['finalmomentum']
    maxepoch = params['maxepoch']

    momentum = initialmomentum

    def sample_prob(probs):
        return tf.nn.relu(tf.sign(probs - tf.random_uniform(tf.shape(probs))))


    X = tf.placeholder("float", [None, numdims], name='Input')

    vishid = tf.placeholder("float", [numdims, numhid], name='Weights')
    visbiases = tf.placeholder("float", [numdims], name='Vis_biases')
    hidbiases = tf.placeholder("float", [numhid], name='Hid_biases')

    vishid_inc = tf.placeholder("float", [numdims, numhid])
    visbiases_inc = tf.placeholder("float", [numdims])
    hidbiases_inc = tf.placeholder("float", [numhid])

    poshidstates = sample_prob(tf.nn.sigmoid(tf.matmul(X, vishid) + hidbiases))
    negdata = sample_prob(tf.nn.sigmoid(tf.matmul(poshidstates, tf.transpose(vishid)) + visbiases))
    neghidprobs = tf.nn.sigmoid(tf.matmul(negdata, vishid) + hidbiases)

    w_positive_grad = tf.matmul(tf.transpose(X), poshidstates)
    w_negative_grad = tf.matmul(tf.transpose(negdata), neghidprobs)

    vishid_inc = epsilonw * (w_positive_grad - w_negative_grad) / tf.to_float(tf.shape(X)[0]) - weightcost * vishid
    visbiases_inc = epsilonvb * tf.reduce_mean(X - negdata, 0)
    hidbiases_inc = epsilonhb * tf.reduce_mean(poshidstates - neghidprobs, 0)

    update_w = vishid + vishid_inc
    update_vb = visbiases + visbiases_inc
    update_hb = hidbiases + hidbiases_inc

    h_sample = sample_prob(tf.nn.sigmoid(tf.matmul(X, vishid) + hidbiases))
    v_sample = sample_prob(tf.nn.sigmoid(tf.matmul(h_sample, tf.transpose(vishid)) + visbiases))
    err = X - v_sample
    err_op = tf.reduce_sum(err * err)

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    n_w = np.zeros([numdims, numhid], np.float32)
    n_vb = np.zeros([numdims], np.float32)
    n_hb = np.zeros([numhid], np.float32)
    o_w = np.zeros([numdims, numhid], np.float32)
    o_vb = np.zeros([numdims], np.float32)
    o_hb = np.zeros([numhid], np.float32)

    # for start, end in zip(range(0, len(trX), batchsize), range(batchsize, len(trX), batchsize)):
    for epoch in range(maxepoch):
        errsum = 0
        if epoch > 5:
            momentum = finalmomentum

        for i in range(numbatches):
            # batch = next(batch_gen)
            batch = batch_gen[i]
            n_w = sess.run(update_w, feed_dict={X: batch, vishid: o_w, visbiases: o_vb, hidbiases: o_hb})
            n_vb = sess.run(update_vb, feed_dict={X: batch, vishid: o_w, visbiases: o_vb, hidbiases: o_hb})
            n_hb = sess.run(update_hb, feed_dict={X: batch, vishid: o_w, visbiases: o_vb, hidbiases: o_hb})
            errsum = errsum + sess.run(err_op, feed_dict={X: batch, vishid: o_w, visbiases: o_vb, hidbiases: o_hb})
            o_w = n_w
            o_vb = n_vb
            o_hb = n_hb
        # print(sess.run(err_op, feed_dict={X: trX, vishid: n_w, visbiases: n_vb, hidbiases: n_hb}))
        print(errsum)
        image = Image.fromarray(
            tile_raster_images(
                X=n_w.T,
                img_shape=(28, 28),
                tile_shape=(25, 20),
                tile_spacing=(1, 1)
            )
        )
        image.save("mnist/output/rbm_%d.png" % (i / 10000))

    return None
