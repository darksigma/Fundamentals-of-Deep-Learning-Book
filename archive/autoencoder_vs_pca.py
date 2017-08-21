from sklearn import decomposition
from matplotlib import pyplot as plt
import tensorflow as tf
import autoencoder_mnist as ae
import argparse, input_data
import numpy as np
# model-checkpoint-0349-191950

def scatter(codes, labels):
    colors = [
        ('#27ae60', 'o'),
        ('#2980b9', 'o'),
        ('#8e44ad', 'o'),
        ('#f39c12', 'o'),
        ('#c0392b', 'o'),
        ('#27ae60', 'x'),
        ('#2980b9', 'x'),
        ('#8e44ad', 'x'),
        ('#c0392b', 'x'),
        ('#f39c12', 'x'),
    ]
    for num in xrange(10):
        plt.scatter([codes[:,0][i] for i in xrange(len(labels)) if labels[i] == num],
        [codes[:,1][i] for i in xrange(len(labels)) if labels[i] == num], 7,
        label=str(num), color = colors[num][0], marker=colors[num][1])
    plt.legend()
    plt.show()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Test various optimization strategies')
    parser.add_argument('savepath', nargs=1, type=str)
    args = parser.parse_args()

    print "\nPULLING UP MNIST DATA"
    mnist = input_data.read_data_sets("data/", one_hot=False)
    print mnist.test.labels

    # print "\nSTARTING PCA"
    # pca = decomposition.PCA(n_components=2)
    # pca.fit(mnist.train.images)
    #
    # print "\nGENERATING PCA CODES AND RECONSTRUCTION"
    # pca_codes = pca.transform(mnist.test.images)
    # print pca_codes
    #
    # scatter(pca_codes, mnist.test.labels)

    with tf.Graph().as_default():

        with tf.variable_scope("autoencoder_model"):

            x = tf.placeholder("float", [None, 784]) # mnist data image of shape 28*28=784
            phase_train = tf.placeholder(tf.bool)

            code = ae.encoder(x, 2, phase_train)

            output = ae.decoder(code, 2, phase_train)

            cost, train_summary_op = ae.loss(output, x)

            global_step = tf.Variable(0, name='global_step', trainable=False)

            train_op = ae.training(cost, global_step)

            eval_op, in_im_op, out_im_op, val_summary_op = ae.evaluate(output, x)

            saver = tf.train.Saver()

            sess = tf.Session()


            print "\nSTARTING AUTOENCODER\n", args.savepath[0]
            sess = tf.Session()
            saver = tf.train.Saver()
            saver.restore(sess, args.savepath[0])

            print "\nGENERATING AE CODES AND RECONSTRUCTION"
            ae_codes, ae_reconstruction = sess.run([code, output], feed_dict={x: mnist.test.images * np.random.randint(2, size=(784)), phase_train: True})

            scatter(ae_codes, mnist.test.labels)

            plt.imshow(ae_reconstruction[0].reshape((28,28)), cmap=plt.cm.gray)
            plt.show()
