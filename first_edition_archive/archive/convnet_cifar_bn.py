import cifar10_input
cifar10_input.maybe_download_and_extract()

import tensorflow as tf
from tensorflow.python import control_flow_ops
import numpy as np
import time, os

# Architecture
n_hidden_1 = 256
n_hidden_2 = 256

# Parameters
learning_rate = 0.01
training_epochs = 1000
batch_size = 128
display_step = 1

def inputs(eval_data=True):
  data_dir = os.path.join('data/cifar10_data', 'cifar-10-batches-bin')
  return cifar10_input.inputs(eval_data=eval_data, data_dir=data_dir,
                              batch_size=batch_size)

def distorted_inputs():
  data_dir = os.path.join('data/cifar10_data', 'cifar-10-batches-bin')
  return cifar10_input.distorted_inputs(data_dir=data_dir,
                                        batch_size=batch_size)

def conv_batch_norm(x, n_out, phase_train):
    beta_init = tf.constant_initializer(value=0.0, dtype=tf.float32)
    gamma_init = tf.constant_initializer(value=1.0, dtype=tf.float32)

    beta = tf.get_variable("beta", [n_out], initializer=beta_init)
    gamma = tf.get_variable("gamma", [n_out], initializer=gamma_init)

    batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
    ema = tf.train.ExponentialMovingAverage(decay=0.9)
    ema_apply_op = ema.apply([batch_mean, batch_var])
    ema_mean, ema_var = ema.average(batch_mean), ema.average(batch_var)
    def mean_var_with_update():
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(batch_mean), tf.identity(batch_var)
    mean, var = control_flow_ops.cond(phase_train,
        mean_var_with_update,
        lambda: (ema_mean, ema_var))

    normed = tf.nn.batch_norm_with_global_normalization(x, mean, var,
        beta, gamma, 1e-3, True)
    return normed

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

def filter_summary(V, weight_shape):
    ix = weight_shape[0]
    iy = weight_shape[1]
    cx, cy = 8, 8
    V_T = tf.transpose(V, (3, 0, 1, 2))
    tf.image_summary("filters", V_T, max_images=64) 

def conv2d(input, weight_shape, bias_shape, phase_train, visualize=False):
    incoming = weight_shape[0] * weight_shape[1] * weight_shape[2]
    weight_init = tf.random_normal_initializer(stddev=(2.0/incoming)**0.5)
    W = tf.get_variable("W", weight_shape, initializer=weight_init)
    if visualize:
        filter_summary(W, weight_shape)
    bias_init = tf.constant_initializer(value=0)
    b = tf.get_variable("b", bias_shape, initializer=bias_init)
    logits = tf.nn.bias_add(tf.nn.conv2d(input, W, strides=[1, 1, 1, 1], padding='SAME'), b)
    return tf.nn.relu(conv_batch_norm(logits, weight_shape[3], phase_train))

def max_pool(input, k=2):
    return tf.nn.max_pool(input, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

def layer(input, weight_shape, bias_shape, phase_train):
    weight_init = tf.random_normal_initializer(stddev=(2.0/weight_shape[0])**0.5)
    bias_init = tf.constant_initializer(value=0)
    W = tf.get_variable("W", weight_shape,
                        initializer=weight_init)
    b = tf.get_variable("b", bias_shape,
                        initializer=bias_init)
    logits = tf.matmul(input, W) + b
    return tf.nn.relu(layer_batch_norm(logits, weight_shape[1], phase_train))

def inference(x, keep_prob, phase_train):

    with tf.variable_scope("conv_1"):
        conv_1 = conv2d(x, [5, 5, 3, 64], [64], phase_train, visualize=True)
        pool_1 = max_pool(conv_1)

    with tf.variable_scope("conv_2"):
        conv_2 = conv2d(pool_1, [5, 5, 64, 64], [64], phase_train)
        pool_2 = max_pool(conv_2)

    with tf.variable_scope("fc_1"):

        dim = 1
        for d in pool_2.get_shape()[1:].as_list():
            dim *= d

        pool_2_flat = tf.reshape(pool_2, [-1, dim])
        fc_1 = layer(pool_2_flat, [dim, 384], [384], phase_train)
        
        # apply dropout
        fc_1_drop = tf.nn.dropout(fc_1, keep_prob)

    with tf.variable_scope("fc_2"):

        fc_2 = layer(fc_1_drop, [384, 192], [192], phase_train)
        
        # apply dropout
        fc_2_drop = tf.nn.dropout(fc_2, keep_prob)

    with tf.variable_scope("output"):
        output = layer(fc_2_drop, [192, 10], [10], phase_train)

    return output


def loss(output, y):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(output, tf.cast(y, tf.int64))    
    loss = tf.reduce_mean(xentropy)
    return loss

def training(cost, global_step):
    tf.scalar_summary("cost", cost)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(cost, global_step=global_step)
    return train_op

def evaluate(output, y):
    correct_prediction = tf.equal(tf.cast(tf.argmax(output, 1), dtype=tf.int32), y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.scalar_summary("validation error", (1.0 - accuracy))
    return accuracy

if __name__ == '__main__':



    with tf.device("/gpu:0"):

        with tf.Graph().as_default():

            with tf.variable_scope("cifar_conv_bn_model"):

                x = tf.placeholder("float", [None, 24, 24, 3])
                y = tf.placeholder("int32", [None])
                keep_prob = tf.placeholder(tf.float32) # dropout probability
                phase_train = tf.placeholder(tf.bool) # training or testing

                distorted_images, distorted_labels = distorted_inputs()
                val_images, val_labels = inputs()

                output = inference(x, keep_prob, phase_train)

                cost = loss(output, y)

                global_step = tf.Variable(0, name='global_step', trainable=False)

                train_op = training(cost, global_step)

                eval_op = evaluate(output, y)

                summary_op = tf.merge_all_summaries()

                saver = tf.train.Saver()

                sess = tf.Session()

                summary_writer = tf.train.SummaryWriter("conv_cifar_bn_logs/",
                                                        graph_def=sess.graph_def)

                
                init_op = tf.initialize_all_variables()

                sess.run(init_op)

                tf.train.start_queue_runners(sess=sess)

                # Training cycle
                for epoch in range(training_epochs):

                    avg_cost = 0.
                    total_batch = int(cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN/batch_size)
                    # Loop over all batches
                    for i in range(total_batch):
                        # Fit training using batch data

                        train_x, train_y = sess.run([distorted_images, distorted_labels])

                        _, new_cost = sess.run([train_op, cost], feed_dict={x: train_x, y: train_y, keep_prob: 1, phase_train: True})
                        # Compute average loss
                        avg_cost += new_cost/total_batch
                        print "Epoch %d, minibatch %d of %d. Cost = %0.4f." %(epoch, i, total_batch, new_cost)
                    
                    # Display logs per epoch step
                    if epoch % display_step == 0:
                        print "Epoch:", '%04d' % (epoch+1), "cost =", "{:.9f}".format(avg_cost)

                        val_x, val_y = sess.run([val_images, val_labels])

                        accuracy = sess.run(eval_op, feed_dict={x: val_x, y: val_y, keep_prob: 1, phase_train: False})

                        print "Validation Error:", (1 - accuracy)

                        summary_str = sess.run(summary_op, feed_dict={x: train_x, y: train_y, keep_prob: 1, phase_train: False})
                        summary_writer.add_summary(summary_str, sess.run(global_step))

                        saver.save(sess, "conv_cifar_bn_logs/model-checkpoint", global_step=global_step)


                print "Optimization Finished!"

                val_x, val_y = sess.run([val_images, val_labels])
                accuracy = sess.run(eval_op, feed_dict={x: val_x, y: val_y, keep_prob: 1, phase_train: False})

                print "Test Accuracy:", accuracy

