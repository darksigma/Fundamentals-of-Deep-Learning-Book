import read_pos_data as data
import tensorflow as tf
import numpy as np
from tensorflow.python import control_flow_ops
import time, argparse

# Architecture
n_gram = 3
embedding_size = 300
n_hidden_1 = 512
n_hidden_2 = 256
n_output = len(data.tags_to_index.keys())

# Parameters
training_epochs = 1000
batch_size = 100
display_step = 1

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

def inference(x, phase_train):
    with tf.variable_scope("inference"):
        with tf.variable_scope("hidden_1"):
            hidden_1 = layer(x, [n_gram * embedding_size, n_hidden_1], [n_hidden_1], phase_train)

        with tf.variable_scope("hidden_2"):
            hidden_2 = layer(hidden_1, [n_hidden_1, n_hidden_2], [n_hidden_2], phase_train)

        with tf.variable_scope("output"):
            output = layer(hidden_2, [n_hidden_2, n_output], [n_output], phase_train)

    return output

def loss(output, y):
    with tf.variable_scope("training"):
        xentropy = tf.nn.softmax_cross_entropy_with_logits(output, y)
        loss = tf.reduce_mean(xentropy)
        train_summary_op = tf.scalar_summary("train_cost", loss)
        return loss, train_summary_op

def training(cost, global_step):
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08,
        use_locking=False, name='Adam')
    train_op = optimizer.minimize(cost, global_step=global_step)
    return train_op

def evaluate(output, y):
    with tf.variable_scope("validation"):
        correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        val_summary_op = tf.scalar_summary("validation error", (1.0 - accuracy))
        return accuracy, val_summary_op


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Test various #-gram')
    parser.add_argument('n_gram', nargs=1, type=int, default=3)
    args = parser.parse_args()
    n_gram = args.n_gram[0]

    print "Using a %d-gram model" % n_gram

    data.train.prepare_n_gram(n_gram)
    data.test.prepare_n_gram(n_gram)

    with tf.Graph().as_default():

        with tf.variable_scope("autoencoder_model"):

            x = tf.placeholder("float", [None, n_gram * embedding_size]) # mnist data image of shape 28*28=784
            y = y = tf.placeholder("float", [None, n_output])
            phase_train = tf.placeholder(tf.bool)

            output = inference(x, phase_train)

            cost, train_summary_op = loss(output, y)

            global_step = tf.Variable(0, name='global_step', trainable=False)

            train_op = training(cost, global_step)

            eval_op, val_summary_op = evaluate(output, y)

            summary_op = tf.merge_all_summaries()

            saver = tf.train.Saver(max_to_keep=50)

            sess = tf.Session()

            train_writer = tf.train.SummaryWriter("pos_tagger=" + str(n_gram) + "-gram_logs/",
                                                graph=sess.graph)

            val_writer = tf.train.SummaryWriter("pos_tagger=" + str(n_gram) + "-gram_logs/",
                                                graph=sess.graph)

            init_op = tf.initialize_all_variables()

            sess.run(init_op)

            prefix = ["She", "decided", "that", "it", "was", "time", "to", "leave", "home", "."]
            sentence = ["Then", "the", "woman", ",", "after", "grabbing", "her", "umbrella", ",", "went", "to", "the", "bank", "to", "deposit", "her", "cash", "."]
            test_vectors = []
            if n_gram > 1:
                for token in prefix[1-n_gram:]:
                    test_vectors.append(np.fromstring(data.db.Get(token), dtype=np.float32))
            for token in sentence:
                test_vectors.append(np.fromstring(data.db.Get(token), dtype=np.float32))

            test_vectors = np.array(test_vectors, dtype=np.float32)

            counter = 0
            test_input = []
            while counter < len(test_vectors) - n_gram + 1:
                test_input.append(test_vectors[counter:counter+n_gram].flatten())
                counter += 1

            test_input = np.array(test_input, dtype=np.float32)


            # Training cycle
            for epoch in range(training_epochs):

                avg_cost = 0.
                total_batch = int(len(data.train.inputs)/batch_size)
                # Loop over all batches
                for i in range(total_batch):
                    minibatch_x, minibatch_y = data.train.minibatch(batch_size)
                    # Fit training using batch data
                    _, new_cost, train_summary = sess.run([train_op, cost, train_summary_op], feed_dict={x: minibatch_x, y: minibatch_y, phase_train: True})
                    train_writer.add_summary(train_summary, sess.run(global_step))
                    # Compute average loss
                    avg_cost += new_cost/total_batch
                # Display logs per epoch step
                if epoch % display_step == 0:
                    print "Epoch:", '%04d' % (epoch+1), "cost =", "{:.9f}".format(avg_cost)

                    val_x, val_y = data.test.minibatch(0)

                    train_writer.add_summary(train_summary, sess.run(global_step))

                    accuracy, val_summary = sess.run([eval_op, val_summary_op], feed_dict={x: val_x, y: val_y, phase_train: False})
                    val_writer.add_summary(val_summary, sess.run(global_step))
                    print "Validation Error:", (1 - accuracy)


                    test_output = sess.run(output, feed_dict={x: test_input, phase_train: False})
                    tags = []
                    for tag_vector in test_output:
                        index = np.argmax(tag_vector)
                        tags.append(data.index_to_tags[index])

                    counter = 0
                    while counter < len(sentence):
                        print "%s\t\t%s" % (sentence[counter], tags[counter])
                        counter += 1



                    saver.save(sess, "pos_tagger=" + str(n_gram) + "-gram_logs//model-checkpoint-" + '%04d' % (epoch+1), global_step=global_step)


            print "Optimization Finished!"
