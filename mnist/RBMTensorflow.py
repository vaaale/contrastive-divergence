import tensorflow as tf


def RBM(batchdata, numhid, params):
    type = params['type']
    noise = True if 'noise' in params else False
    epsilonw = params['epsilonw']
    epsilonvb = params['epsilonvb']
    epsilonhb = params['epsilonhb']

    weightcost = params['weightcost']

    initialmomentum = params['initialmomentum']
    finalmomentum = params['finalmomentum']
    maxepoch = params['maxepoch']

    numbatches = len(batchdata)
    numcases, numdims = batchdata[0].shape



    # tf Graph input
    x = tf.placeholder("float", [None, numdims])
    y = tf.placeholder("float", [None, n_classes])


    # Create model
    def multilayer_perceptron(x, weights, biases):
        # Hidden layer with RELU activation
        poshidprobs = tf.add(tf.matmul(x, weights['vishid']), biases['hidbiases'])
        poshidprobs = tf.nn.sigmoid(poshidprobs)



        negdata = tf.nn.sigmoid(tf.add(tf.matmul(poshidprobs, tf.transpose(weights['vishid'])), biases['visbiases']))
        negdata = tf.nn.sigmoid(tf.add(tf.matmul(poshidprobs, tf.transpose(weights['vishid'])), biases['visbiases']))

        neghidprobs = tf.add(tf.matmul(negdata, weights['vishid']), biases['hidbiases'])

        return out_layer

    # Store layers weight & bias
    weights = {
        'vishid': tf.Variable(tf.random_normal([numdims, numhid])),
    }
    biases = {
        'hidbiases': tf.Variable(tf.random_normal([numhid])),
        'visbiases': tf.Variable(tf.random_normal([numdims])),
    }



    # Construct model
    pred = multilayer_perceptron(x, weights, biases)


    # Initializing the variables
    init = tf.initialize_all_variables()

    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)

        # Training cycle
        for epoch in range(maxepoch):
            avg_cost = 0.
            # Loop over all batches
            for i in range(numbatches):
                batch_x = batchdata[i]
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                              y: batch_y})
                # Compute average loss
                avg_cost += c / total_batch
            # Display logs per epoch step
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch+1), "cost=", \
                    "{:.9f}".format(avg_cost))
        print("Optimization Finished!")

        # Test model
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
