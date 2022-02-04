"""
We will use this class to represent a simple logistic regression
classifier. We'll represent this in Theano as a neural network 
with no hidden layers. This is our first attempt at building a 
neural network model to solve interesting problems. Here, we'll
use this class to crack the MNIST handwritten digit dataset problem,
but this class has been constructed so that it can be reappropriated
to any use!

References:
    - textbooks: "Pattern Recognition and Machine Learning", Christopher M. Bishop, section 4.3.2
    - websites: http://deeplearning.net/tutorial, Lisa Lab
"""

import numpy as np
import theano.tensor as T 
import theano

class LogisticNetwork(object):
    """
    The logistic regression class is described by two parameters (which
    we will want to learn). The first is a weight matrix. We'll refer to
    this weight matrix as W. The second is a bias vector b. Refer to the 
    text if you want to learn more about how this network works. Let's get
    started!
    """

    def __init__(self, input, input_dim, output_dim):
        """
        We first initialize the logistic network object with some important
        information.

        PARAM input : theano.tensor.TensorType
        A symbolic variable that we'll use to represent one minibatch of our
        dataset

        PARAM input_dim : int
        This will represent the number of input neurons in our model

        PARAM ouptut_dim : int 
        This will represent the number of neurons in the output layer (i.e. 
        the number of possible classifications for the input)
        """

        # We initialize the weight matrix W of size (input_dim, output_dim)
        self.W = theano.shared(
                value=np.zeros((input_dim, output_dim)),
                name='W',
                borrow=True
            )

        # We initialize a bias vector for the neurons of the output layer
        self.b = theano.shared(
                value=np.zeros(output_dim),
                name='b',
                borrow='True'
            )

        # Symbolic description of how to compute class membership probabilities
        self.output = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # Symbolic description of the final prediction
        self.predicted = T.argmax(self.output, axis=1)

    def logistic_network_cost(self, y, lambda_l2=0):
        """
        Here we express the cost incurred by an example given the correct
        distribution

        PARAM y : theano.tensor.TensorType
        These are the correct answers, and we compute the cost with 
        respect to this ground truth (over the entire minibatch). This 
        means that y is of size (minibatch_size, output_dim)

        PARAM lambda : float
        This is the L2 regularization parameter that we use to penalize large
        values for components of W, thus discouraging potential overfitting
        """
        # Calculate the log probabilities of the softmax output
        log_probabilities = T.log(self.output)

        # We use these log probabilities to compute the negative log likelihood
        negative_log_likelihood = -T.mean(log_probabilities[T.arange(y.shape[0]), y])
        
        # Compute the L2 regularization component of the cost function
        l2_regularization = lambda_l2 * (self.W ** 2).sum()
        
        # Return a symbolic description of the cost function
        return negative_log_likelihood + l2_regularization

    def error_rate(self, y):
        """
        Here we return the error rate of the model over a set of given labels
        (perhaps in a minibatch, in the validation set, or the test set)

        PARAM y : theano.tensor.TensorType
        These are the correct answers, and we compute the cost with 
        respect to this ground truth (over the entire minibatch). This 
        means that y is of size (minibatch_size, output_dim)
        """

        # Make sure y is of the correct dimension 
        assert y.ndim == self.predicted.ndim

        # Make sure that y contains values of the correct data type (ints)
        assert y.dtype.startswith('int')

        # Return the error rate on the data 
        return T.mean(T.neq(self.predicted, y))

        



