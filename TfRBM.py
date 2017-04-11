import tensorflow as tf
import time


def RBM(batch_gen, numbatches, numdims, numhid, params, write_fn, callbacks=[]):
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

    tf.reset_default_graph()

    def sample_prob(probs):
        return tf.nn.relu(tf.sign(probs - tf.random_uniform(tf.shape(probs))))

    X = tf.placeholder("float", [None, numdims], name='X')
    p = tf.placeholder("float", [1], name='Momentum')

    vishid = tf.get_variable('W', shape=[numdims, numhid], initializer=tf.contrib.layers.xavier_initializer())
    visbiases = tf.get_variable('VB', shape=[numdims], initializer=tf.contrib.layers.xavier_initializer())
    hidbiases = tf.get_variable('HB', shape=[numhid], initializer=tf.contrib.layers.xavier_initializer())

    vishid_inc = tf.get_variable('W_inc', shape=[numdims, numhid], initializer=tf.zeros_initializer())
    visbiases_inc = tf.get_variable('VB_inc', shape=[numdims], initializer=tf.zeros_initializer())
    hidbiases_inc = tf.get_variable('HB_inc', shape=[numhid], initializer=tf.zeros_initializer())

    pos_logits = tf.matmul(X, vishid) + hidbiases
    if activation == 'sigmoid':
        poshidprobs = tf.nn.sigmoid(pos_logits)
    elif activation == 'linear':
        poshidprobs = pos_logits
    else:
        poshidprobs = pos_logits

    if not noise:
        poshidstates = sample_prob(poshidprobs)
    else:
        poshidstates = poshidprobs + tf.random_uniform(tf.shape(poshidprobs))

    negdata = tf.nn.sigmoid(tf.matmul(poshidstates, tf.transpose(vishid)) + visbiases)

    neg_logits = tf.matmul(negdata, vishid) + hidbiases
    if activation == 'sigmoid':
        neghidprobs = tf.nn.sigmoid(neg_logits)
    elif activation == 'linear':
        neghidprobs = neg_logits
    else:
        neghidprobs = neg_logits

    w_positive_grad = tf.matmul(tf.transpose(X), poshidstates)
    w_negative_grad = tf.matmul(tf.transpose(negdata), neghidprobs)

    vishid_inc = vishid_inc.assign(p * vishid_inc + epsilonw * (w_positive_grad - w_negative_grad) / tf.to_float(tf.shape(X)[0]) - weightcost * vishid)
    visbiases_inc = visbiases_inc.assign(p * visbiases_inc + epsilonvb * tf.reduce_mean(X - negdata, 0))
    hidbiases_inc = hidbiases_inc.assign(p * hidbiases_inc + epsilonhb * tf.reduce_mean(poshidstates - neghidprobs, 0))

    update_w = vishid.assign(vishid + vishid_inc)
    update_vb = visbiases.assign(visbiases + visbiases_inc)
    update_hb = hidbiases.assign(hidbiases + hidbiases_inc)

    # h_sample = sample_prob(tf.nn.sigmoid(tf.matmul(X, vishid) + hidbiases))
    # v_sample = sample_prob(tf.nn.sigmoid(tf.matmul(h_sample, tf.transpose(vishid)) + visbiases))
    err = X - negdata
    err_op = tf.reduce_sum(err * err)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        for epoch in range(maxepoch):
            errsum = 0
            if epoch > 5:
                momentum = finalmomentum

            epoch_time = 0
            n_w = None
            for i in range(numbatches):
                b_start = time.time()
                batch = next(batch_gen)
                n_w, err = sess.run([update_w, err_op], feed_dict={X: batch, p: [momentum]})
                n_vb = sess.run(update_vb, feed_dict={X: batch, p: [momentum]})
                n_hb = sess.run(update_hb, feed_dict={X: batch, p: [momentum]})
                if epoch == maxepoch - 1:
                    write_fn(sess.run(poshidprobs, feed_dict={X: batch}))
                errsum = errsum + err
                b_end = time.time()
                epoch_time += (b_end - b_start)
            print('Epoch:{}: {:.2f} seconds, error: {:.2f}'.format(epoch, epoch_time, errsum))
            if len(callbacks):
                for callback in callbacks:
                    callback(epoch, n_w)

        h_vishid = sess.run(vishid)
        h_visbiases = sess.run(visbiases)
        h_hidbiases = sess.run(hidbiases)
        print('Model shapes:')
        print(h_vishid.shape)
        print(h_visbiases.shape)
        print(h_hidbiases.shape)
        model = {
            'vishid': h_vishid,
            'visbiases': h_visbiases.reshape(1, numdims),
            'hidbiases': h_hidbiases.reshape(1, numhid)
        }

    return model
