import input_data
import tensorflow as tf
from tensorflow.python import control_flow_ops
import time, argparse

# Architecture
n_encoder_hidden_1 = 1000
n_encoder_hidden_2 = 500
n_encoder_hidden_3 = 250
n_decoder_hidden_1 = 250
n_decoder_hidden_2 = 500
n_decoder_hidden_3 = 1000

# Parameters
learning_rate = 0.01
training_epochs = 1000
batch_size = 100
display_step = 1

def corrupt_input(x):
    corrupting_matrix = tf.random_uniform(shape=tf.shape(x), minval=0,maxval=2,dtype=tf.int32)
    return x * tf.cast(corrupting_matrix, tf.float32)

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

def encoder(x, n_code, phase_train):
    with tf.variable_scope("encoder"):
        with tf.variable_scope("hidden_1"):
            hidden_1 = layer(x, [784, n_encoder_hidden_1], [n_encoder_hidden_1], phase_train)

        with tf.variable_scope("hidden_2"):
            hidden_2 = layer(hidden_1, [n_encoder_hidden_1, n_encoder_hidden_2], [n_encoder_hidden_2], phase_train)

        with tf.variable_scope("hidden_3"):
            hidden_3 = layer(hidden_2, [n_encoder_hidden_2, n_encoder_hidden_3], [n_encoder_hidden_3], phase_train)

        with tf.variable_scope("code"):
            code = layer(hidden_3, [n_encoder_hidden_3, n_code], [n_code], phase_train)

    return code

def decoder(code, n_code, phase_train):
    with tf.variable_scope("decoder"):
        with tf.variable_scope("hidden_1"):
            hidden_1 = layer(code, [n_code, n_decoder_hidden_1], [n_decoder_hidden_1], phase_train)

        with tf.variable_scope("hidden_2"):
            hidden_2 = layer(hidden_1, [n_decoder_hidden_1, n_decoder_hidden_2], [n_decoder_hidden_2], phase_train)

        with tf.variable_scope("hidden_3"):
            hidden_3 = layer(hidden_2, [n_decoder_hidden_2, n_decoder_hidden_3], [n_decoder_hidden_3], phase_train)

        with tf.variable_scope("output"):
            output = layer(hidden_3, [n_decoder_hidden_3, 784], [784], phase_train)

    return output

def loss(output, x):
    with tf.variable_scope("training"):
        l2 = tf.sqrt(tf.reduce_sum(tf.square(tf.sub(output, x)), 1))
        train_loss = tf.reduce_mean(l2)
        train_summary_op = tf.scalar_summary("train_cost", train_loss)
        return train_loss, train_summary_op

def training(cost, global_step):
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08,
        use_locking=False, name='Adam')
    train_op = optimizer.minimize(cost, global_step=global_step)
    return train_op

def image_summary(label, tensor):
    tensor_reshaped = tf.reshape(tensor, [-1, 28, 28, 1])
    return tf.image_summary(label, tensor_reshaped)

def evaluate(output, x, c_x):
    with tf.variable_scope("validation"):
        in_im_op = image_summary("input_image", c_x)
        out_im_op = image_summary("output_image", output)
        l2 = tf.sqrt(tf.reduce_sum(tf.square(tf.sub(output, x, name="val_diff")), 1))
        val_loss = tf.reduce_mean(l2)
        val_summary_op = tf.scalar_summary("val_cost", val_loss)
        return val_loss, in_im_op, out_im_op, val_summary_op


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Test various optimization strategies')
    parser.add_argument('n_code', nargs=1, type=str)
    args = parser.parse_args()
    n_code = args.n_code[0]

    mnist = input_data.read_data_sets("data/", one_hot=True)

    with tf.Graph().as_default():

        with tf.variable_scope("autoencoder_model"):

            x = tf.placeholder("float", [None, 784]) # mnist data image of shape 28*28=784
            corrupt = tf.placeholder(tf.float32)
            phase_train = tf.placeholder(tf.bool)

            c_x = (corrupt_input(x) * corrupt) + (x * (1 - corrupt))

            code = encoder(c_x, int(n_code), phase_train)

            output = decoder(code, int(n_code), phase_train)

            cost, train_summary_op = loss(output, x)

            global_step = tf.Variable(0, name='global_step', trainable=False)

            train_op = training(cost, global_step)

            eval_op, in_im_op, out_im_op, val_summary_op = evaluate(output, x, c_x)

            summary_op = tf.merge_all_summaries()

            saver = tf.train.Saver(max_to_keep=200)

            sess = tf.Session()

            train_writer = tf.train.SummaryWriter("dn_mnist_autoencoder_hidden=" + n_code + "_logs/",
                                                graph=sess.graph)

            val_writer = tf.train.SummaryWriter("dn_mnist_autoencoder_hidden=" + n_code + "_logs/",
                                                graph=sess.graph)

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
                    _, new_cost, train_summary = sess.run([train_op, cost, train_summary_op], feed_dict={x: minibatch_x, phase_train: True, corrupt: 1})
                    train_writer.add_summary(train_summary, sess.run(global_step))
                    # Compute average loss
                    avg_cost += new_cost/total_batch
                # Display logs per epoch step
                if epoch % display_step == 0:
                    print "Epoch:", '%04d' % (epoch+1), "cost =", "{:.9f}".format(avg_cost)

                    train_writer.add_summary(train_summary, sess.run(global_step))

                    validation_loss, in_im, out_im, val_summary = sess.run([eval_op, in_im_op, out_im_op, val_summary_op], feed_dict={x: mnist.validation.images, phase_train: False, corrupt: 1})
                    val_writer.add_summary(in_im, sess.run(global_step))
                    val_writer.add_summary(out_im, sess.run(global_step))
                    val_writer.add_summary(val_summary, sess.run(global_step))
                    print "Validation Loss:", validation_loss

                    saver.save(sess, "dn_mnist_autoencoder_hidden=" + n_code + "_logs/model-checkpoint-" + '%04d' % (epoch+1), global_step=global_step)


            print "Optimization Finished!"


            test_loss = sess.run(eval_op, feed_dict={x: mnist.test.images, phase_train: False, corrupt: 0})

            print "Test Loss:", loss
