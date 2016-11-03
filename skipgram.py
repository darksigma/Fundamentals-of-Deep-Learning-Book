import input_word_data as data
import numpy as np
import tensorflow as tf
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# TRAINING PARAMETERS
batch_size = 32                                             # Number of training examples per batch
embedding_size = 128                                        # Dimension of embedding vectors
skip_window = 5                                             # Window size for context to the left and right of target
num_skips = 4                                               # How many times to reuse target to generate a label for context.
batches_per_epoch = data.data_size*num_skips/batch_size     # Number of batches per epoch of training
training_epochs = 5                                         # Number of epochs to utilize for training
neg_size = 64                                               # Number of negative samples to use for NCE
display_step = 2000                                         # Frequency with which to print statistics
val_step = 10000                                            # Frequency with which to perform validation
learning_rate = 0.1                                         # Learning rate for SGD

print "Epochs: %d, Batches per epoch: %d, Examples per batch: %d" % (training_epochs, batches_per_epoch, batch_size)

# NEAREST NEIGHBORS VALIDATION PARAMETERS
val_size = 20
val_dist_span = 500
val_examples = np.random.choice(val_dist_span, val_size, replace=False)
top_match = 8
plot_num = 500


def embedding_layer(x, embedding_shape):
    with tf.variable_scope("embedding"):
        embedding_init = tf.random_uniform(embedding_shape, -1.0, 1.0)
        embedding_matrix = tf.get_variable("E", initializer=embedding_init)
        return tf.nn.embedding_lookup(embedding_matrix, x), embedding_matrix

def noise_contrastive_loss(embedding_lookup, weight_shape, bias_shape, y):
    with tf.variable_scope("nce"):
        nce_weight_init = tf.truncated_normal(weight_shape, stddev=1.0/(weight_shape[1])**0.5)
        nce_bias_init = tf.zeros(bias_shape)
        nce_W = tf.get_variable("W", initializer=nce_weight_init)
        nce_b = tf.get_variable("b", initializer=nce_bias_init)

        total_loss = tf.nn.nce_loss(nce_W, nce_b, embedding_lookup, y, neg_size, data.vocabulary_size)
        return tf.reduce_mean(total_loss)

def training(cost, global_step):
    with tf.variable_scope("training"):
        summary_op = tf.scalar_summary("cost", cost)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train_op = optimizer.minimize(cost, global_step=global_step)
        return train_op, summary_op

def validation(embedding_matrix, x_val):
    norm = tf.reduce_sum(embedding_matrix**2, 1, keep_dims=True)**0.5
    normalized = embedding_matrix/norm
    val_embeddings = tf.nn.embedding_lookup(normalized, x_val)
    cosine_similarity = tf.matmul(val_embeddings, normalized, transpose_b=True)
    return normalized, cosine_similarity

if __name__ == '__main__':

    with tf.Graph().as_default():

        with tf.variable_scope("skipgram_model"):

            x = tf.placeholder(tf.int32, shape=[batch_size])
            y = tf.placeholder(tf.int32, [batch_size, 1])
            val = tf.constant(val_examples, dtype=tf.int32)
            global_step = tf.Variable(0, name='global_step', trainable=False)

            e_lookup, e_matrix = embedding_layer(x, [data.vocabulary_size, embedding_size])

            cost = noise_contrastive_loss(e_lookup, [data.vocabulary_size, embedding_size], [data.vocabulary_size], y)

            train_op, summary_op = training(cost, global_step)

            val_op = validation(e_matrix, val)

            sess = tf.Session()

            train_writer = tf.train.SummaryWriter("skipgram_logs/", graph=sess.graph)

            init_op = tf.initialize_all_variables()

            sess.run(init_op)

            step = 0
            avg_cost = 0

            for epoch in xrange(training_epochs):
                for minibatch in xrange(batches_per_epoch):

                    step +=1

                    minibatch_x, minibatch_y = data.generate_batch(batch_size, num_skips, skip_window)
                    feed_dict = {x : minibatch_x, y : minibatch_y}

                    _, new_cost, train_summary = sess.run([train_op, cost, summary_op], feed_dict=feed_dict)
                    train_writer.add_summary(train_summary, sess.run(global_step))
                    # Compute average loss
                    avg_cost += new_cost/display_step

                    if step % display_step == 0:
                        print "Elapsed:", str(step), "batches. Cost =", "{:.9f}".format(avg_cost)
                        avg_cost = 0

                    if step % val_step == 0:
                        _, similarity = sess.run(val_op)
                        for i in xrange(val_size):
                            val_word = data.reverse_dictionary[val_examples[i]]
                            neighbors = (-similarity[i, :]).argsort()[1:top_match+1]
                            print_str = "Nearest neighbor of %s:" % val_word
                            for k in xrange(top_match):
                                print_str += " %s," % data.reverse_dictionary[neighbors[k]]
                            print print_str[:-1]

            final_embeddings, _ = sess.run(val_op)


    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    plot_embeddings = np.asfarray(final_embeddings[:plot_num,:], dtype='float')
    low_dim_embs = tsne.fit_transform(plot_embeddings)
    labels = [reverse_dictionary[i] for i in xrange(plot_only)]
    data.plot_with_labels(low_dim_embs, labels)
