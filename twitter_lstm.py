import time
import uuid
import os
import numpy as np
import tensorflow as tf
from tensorflow.python.ops.rnn import dynamic_rnn
from lstm import LSTMCell, BNLSTMCell, orthogonal_initializer
import read_tweet_data as data

batch_size = 100
hidden_size = 256

with tf.device('/gpu:0'):
    x_inp = tf.placeholder(tf.float32, [None, 399, 101])
    training = tf.placeholder(tf.bool)

    lstm = LSTMCell(hidden_size)

    #c, h
    initialState = (
        tf.random_normal([batch_size, hidden_size], stddev=0.1),
        tf.random_normal([batch_size, hidden_size], stddev=0.1))

    outputs, state = dynamic_rnn(lstm, x_inp, initial_state=initialState)

    _, final_hidden = state

    W = tf.get_variable('W', [hidden_size, 3], initializer=orthogonal_initializer())
    b = tf.get_variable('b', [3])

    y = tf.nn.softmax(tf.matmul(final_hidden, W) + b)

    y_ = tf.placeholder(tf.float32, [None, 3])

    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

    optimizer = tf.train.AdamOptimizer()
    gvs = optimizer.compute_gradients(cross_entropy)
    capped_gvs = [(None if grad is None else tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
    train_step = optimizer.apply_gradients(capped_gvs)

    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Summaries
    tf.scalar_summary("accuracy", accuracy)
    tf.scalar_summary("xe_loss", cross_entropy)
    for (grad, var), (capped_grad, _) in zip(gvs, capped_gvs):
        if grad is not None:
            tf.histogram_summary('grad/{}'.format(var.name), capped_grad)
            tf.histogram_summary('capped_fraction/{}'.format(var.name),
                tf.nn.zero_fraction(grad - capped_grad))
            tf.histogram_summary('weight/{}'.format(var.name), var)

    merged = tf.merge_all_summaries()

    init = tf.initialize_all_variables()

    sess = tf.Session(config=tf.ConfigProto(
              allow_soft_placement=True, log_device_placement=True))
    sess.run(init)

    logdir = 'lstm_bn_logs/' + str(uuid.uuid4())
    os.makedirs(logdir)
    print('logging to ' + logdir)
    writer = tf.train.SummaryWriter(logdir, sess.graph)

    current_time = time.time()
    print("Using population statistics (training: False) at test time gives worse results than batch statistics")

    for i in range(100000):
        batch_xs, batch_ys = data.train.minibatch(batch_size)
        loss, _ = sess.run([cross_entropy, train_step], feed_dict={x_inp: batch_xs, y_: batch_ys, training: True})
        step_time = time.time() - current_time
        current_time = time.time()
        if i % 100 == 0:
            batch_xs, batch_ys = data.val.minibatch(batch_size)
            summary_str = sess.run(merged, feed_dict={x_inp: batch_xs, y_: batch_ys, training: False})
            writer.add_summary(summary_str, i)
        print(loss, step_time)
