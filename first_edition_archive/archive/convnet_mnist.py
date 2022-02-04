import input_data
mnist = input_data.read_data_sets("data/", one_hot=True)

import tensorflow as tf
import time

# Architecture
n_hidden_1 = 256
n_hidden_2 = 256

# Parameters
learning_rate = 0.0001
training_epochs = 1000
batch_size = 100
display_step = 1

def conv2d(input, weight_shape, bias_shape):
    incoming = weight_shape[0] * weight_shape[1] * weight_shape[2]
    weight_init = tf.random_normal_initializer(stddev=(2.0/incoming)**0.5)
    W = tf.get_variable("W", weight_shape, initializer=weight_init)
    bias_init = tf.constant_initializer(value=0)
    b = tf.get_variable("b", bias_shape, initializer=bias_init)
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(input, W, strides=[1, 1, 1, 1], padding='SAME'), b))

def max_pool(input, k=2):
    return tf.nn.max_pool(input, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

def layer(input, weight_shape, bias_shape):
    weight_init = tf.random_normal_initializer(stddev=(2.0/weight_shape[0])**0.5)
    bias_init = tf.constant_initializer(value=0)
    W = tf.get_variable("W", weight_shape,
                        initializer=weight_init)
    b = tf.get_variable("b", bias_shape,
                        initializer=bias_init)
    return tf.nn.relu(tf.matmul(input, W) + b)


def inference(x, keep_prob):

    x = tf.reshape(x, shape=[-1, 28, 28, 1])
    with tf.variable_scope("conv_1"):
        conv_1 = conv2d(x, [5, 5, 1, 32], [32])
        pool_1 = max_pool(conv_1)

    with tf.variable_scope("conv_2"):
        conv_2 = conv2d(pool_1, [5, 5, 32, 64], [64])
        pool_2 = max_pool(conv_2)

    with tf.variable_scope("fc"):
        pool_2_flat = tf.reshape(pool_2, [-1, 7 * 7 * 64])
        fc_1 = layer(pool_2_flat, [7*7*64, 1024], [1024])
        
        # apply dropout
        fc_1_drop = tf.nn.dropout(fc_1, keep_prob)

    with tf.variable_scope("output"):
        output = layer(fc_1_drop, [1024, 10], [10])

    return output


def loss(output, y):
    xentropy = tf.nn.softmax_cross_entropy_with_logits(output, y)    
    loss = tf.reduce_mean(xentropy)
    return loss

def training(cost, global_step):
    tf.scalar_summary("cost", cost)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(cost, global_step=global_step)
    return train_op


def evaluate(output, y):
    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.scalar_summary("validation error", (1.0 - accuracy))
    return accuracy

if __name__ == '__main__':

    with tf.device("/gpu:0"):

        with tf.Graph().as_default():

            with tf.variable_scope("mnist_conv_model"):

                x = tf.placeholder("float", [None, 784]) # mnist data image of shape 28*28=784
                y = tf.placeholder("float", [None, 10]) # 0-9 digits recognition => 10 classes
                keep_prob = tf.placeholder(tf.float32) # dropout probability

                output = inference(x, keep_prob)

                cost = loss(output, y)

                global_step = tf.Variable(0, name='global_step', trainable=False)

                train_op = training(cost, global_step)

                eval_op = evaluate(output, y)

                summary_op = tf.merge_all_summaries()

                saver = tf.train.Saver()

                sess = tf.Session()

                summary_writer = tf.train.SummaryWriter("conv_mnist_logs/",
                                                    graph_def=sess.graph_def)

                
                init_op = tf.initialize_all_variables()

                sess.run(init_op)


                # Training cycle
                for epoch in range(training_epochs):

                    avg_cost = 0.
                    total_batch = int(mnist.train.num_examples/batch_size)
                    # Loop over all batches
                    for i in range(total_batch):
                        minibatch_x, minibatch_y = mnist.train.next_batch(batch_size)
                        # Fit training using batch data
                        sess.run(train_op, feed_dict={x: minibatch_x, y: minibatch_y, keep_prob: 0.5})
                        # Compute average loss
                        avg_cost += sess.run(cost, feed_dict={x: minibatch_x, y: minibatch_y, keep_prob: 0.5})/total_batch
                    # Display logs per epoch step
                    if epoch % display_step == 0:
                        print "Epoch:", '%04d' % (epoch+1), "cost =", "{:.9f}".format(avg_cost)

                        accuracy = sess.run(eval_op, feed_dict={x: mnist.validation.images, y: mnist.validation.labels, keep_prob: 1})

                        print "Validation Error:", (1 - accuracy)

                        summary_str = sess.run(summary_op, feed_dict={x: minibatch_x, y: minibatch_y, keep_prob: 0.5})
                        summary_writer.add_summary(summary_str, sess.run(global_step))

                        saver.save(sess, "conv_mnist_logs/model-checkpoint", global_step=global_step)


                print "Optimization Finished!"


                accuracy = sess.run(eval_op, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1})

                print "Test Accuracy:", accuracy
