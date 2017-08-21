import time
import uuid
import os
import numpy as np
import tensorflow as tf
from tensorflow.python import control_flow_ops
from tensorflow.python.ops.rnn import dynamic_rnn
from tensorflow.python.ops.rnn_cell import MultiRNNCell
from lstm import LSTMCell, BNLSTMCell, orthogonal_initializer
import read_tweet_data as data
from sklearn.metrics import confusion_matrix

batch_size = 256
hidden_size = 32

def layer_batch_norm(x, n_out, phase_train):
    beta_init = tf.constant_initializer(value=0.0, dtype=tf.float32)
    gamma_init = tf.constant_initializer(value=1.0, dtype=tf.float32)

    beta = tf.get_variable("beta", [n_out], initializer=beta_init)
    gamma = tf.get_variable("gamma", [n_out], initializer=gamma_init)

    batch_mean, batch_var = tf.nn.moments(x, [0], name='moments')
    ema = tf.train.ExponentialMovingAverage(decay=0.9)
    ema_apply_op = ema.apply([batch_mean, batch_var])
    ema_mean, ema_var = ema.average(batch_mean), ema.average(batch_var)
    def mean_var_with_update():
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(batch_mean), tf.identity(batch_var)
    mean, var = control_flow_ops.cond(phase_train,
        mean_var_with_update,
        lambda: (ema_mean, ema_var))

    reshaped_x = tf.reshape(x, [-1, 1, 1, n_out])
    normed = tf.nn.batch_norm_with_global_normalization(reshaped_x, mean, var,
        beta, gamma, 1e-3, True)
    return tf.reshape(normed, [-1, n_out])

def layer(input, weight_shape, bias_shape, phase_train):
    weight_init = tf.random_normal_initializer(stddev=(1.0/weight_shape[0])**0.5)
    bias_init = tf.constant_initializer(value=0)
    W = tf.get_variable("W", weight_shape,
                        initializer=weight_init)
    b = tf.get_variable("b", bias_shape,
                        initializer=bias_init)
    logits = tf.matmul(input, W) + b
    return tf.nn.sigmoid(layer_batch_norm(logits, weight_shape[1], phase_train))

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

    intermediary = layer(final_hidden, [hidden_size, 2], [2], training)

    y = tf.nn.softmax(intermediary)

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
    tr_acc = tf.scalar_summary("train_accuracy", accuracy)
    xe_summary = tf.scalar_summary("xe_loss", cross_entropy)
    val_summary_op = tf.scalar_summary("val_loss", cross_entropy)
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
        t_batch_xs, t_batch_ys = data.train.minibatch()
        loss, xe_str, _, train_preds, train_acc = sess.run([cross_entropy, xe_summary, train_step, y, tr_acc], feed_dict={x_inp: t_batch_xs, y_: t_batch_ys, training: True})
        step_time = time.time() - current_time
        writer.add_summary(xe_str, i)
        writer.add_summary(train_acc, i)
        current_time = time.time()
        if i % 100 == 0:
            batch_xs, batch_ys = data.val.minibatch()
            a_str, val_summary, preds = sess.run([a_summary, val_summary_op, y], feed_dict={x_inp: batch_xs, y_: batch_ys, training: False})


            print train_preds[:10], t_batch_ys[:10]


            cnf_matrix = confusion_matrix(np.argmax(train_preds, axis=1), np.argmax(t_batch_ys, axis=1))
            print "Traning Confusion Matrix:", cnf_matrix.tolist()


            print preds[:10], batch_ys[:10]


            cnf_matrix = confusion_matrix(np.argmax(preds, axis=1), np.argmax(batch_ys, axis=1))
            print "Validation Confusion Matrix:", cnf_matrix.tolist()
            writer.add_summary(a_str, i)
            writer.add_summary(val_summary, i)
        print(loss, step_time)
