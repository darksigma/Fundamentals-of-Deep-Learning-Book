import time
import uuid
import os
import numpy as np
import tensorflow as tf
from tensorflow.python.ops.rnn import dynamic_rnn
from tensorflow.python.ops.rnn_cell import MultiRNNCell
from lstm import LSTMCell, BNLSTMCell, orthogonal_initializer
import read_tweet_data as data
from sklearn.metrics import confusion_matrix

batch_size = 256
hidden_size = 16

with tf.device('/gpu:0'):
    x_inp = tf.placeholder(tf.float32, [None, 200, 155])
    training = tf.placeholder(tf.bool)

    lstm = BNLSTMCell(hidden_size, training)

    #c, h
    initialState = (
        tf.random_normal([batch_size, hidden_size], stddev=0.1),
        tf.random_normal([batch_size, hidden_size], stddev=0.1))

    outputs, state = dynamic_rnn(lstm, x_inp, initial_state=initialState)

    _, final_hidden = state

    W = tf.get_variable('W', [hidden_size, 2], initializer=orthogonal_initializer())
    b = tf.get_variable('b', [2])

    y = tf.nn.softmax(tf.matmul(final_hidden, W) + b)

    y_ = tf.placeholder(tf.float32, [None, 2])

    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

    optimizer = tf.train.AdamOptimizer()
    gvs = optimizer.compute_gradients(cross_entropy)
    capped_gvs = [(None if grad is None else tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
    train_step = optimizer.apply_gradients(capped_gvs)

    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Summaries
    a_summary = tf.scalar_summary("accuracy", accuracy)
    xe_summary = tf.scalar_summary("xe_loss", cross_entropy)
    for (grad, var), (capped_grad, _) in zip(gvs, capped_gvs):
        if grad is not None:
            tf.histogram_summary('grad/{}'.format(var.name), capped_grad)
            tf.histogram_summary('capped_fraction/{}'.format(var.name),
                tf.nn.zero_fraction(grad - capped_grad))
            tf.histogram_summary('weight/{}'.format(var.name), var)

    init = tf.initialize_all_variables()

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
    sess.run(init)

    logdir = 'airline_logs/'
    print('logging to ' + logdir)
    writer = tf.train.SummaryWriter(logdir, sess.graph)

    current_time = time.time()
    print("Using population statistics (training: False) at test time gives worse results than batch statistics")

    for i in range(100000):
        batch_xs, batch_ys = data.train.minibatch()
        loss, xe_str, _ = sess.run([cross_entropy, xe_summary, train_step], feed_dict={x_inp: batch_xs, y_: batch_ys, training: True})
        step_time = time.time() - current_time
        writer.add_summary(xe_str, i)
        current_time = time.time()
        if i % 100 == 0:
            batch_xs, batch_ys = data.val.minibatch()
            a_str, preds = sess.run([a_summary, y], feed_dict={x_inp: batch_xs, y_: batch_ys, training: False})

            print np.max(preds, axis=1)[:10], np.argmax(batch_ys, axis=1)[:10]

            cnf_matrix = confusion_matrix(np.argmax(preds, axis=1), np.argmax(batch_ys, axis=1))
            print "Confusion Matrix:", cnf_matrix.tolist()
            writer.add_summary(a_str, i)
        print(loss, step_time)
