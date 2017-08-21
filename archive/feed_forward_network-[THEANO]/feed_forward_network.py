"""
We will use this class to represent a simple feedforward neural 
network. Here, we'll use this class to crack the MNIST handwritten 
digit dataset problem, but this class has been constructed so 
that it can be reappropriated to any use!

References:
    - textbooks: "Pattern Recognition and Machine Learning", Christopher M. Bishop, section 4.3.2
    - websites: http://deeplearning.net/tutorial, Lisa Lab
"""

import numpy as np
import theano.tensor as T 
import theano

from hidden_layer import HiddenLayer as HL
from softmax_layer import SoftmaxLayer as SL



class FeedForwardNetwork(object):
    """
    The feed forward neural network is described mostly input
    data in the form of a minibatch, a list of hidden layers,
    and a softmax layer to make predictions. 
    """

    def __init__ (self, random_gen, input, input_dim, output_dim, hidden_layer_sizes):
        """
        We first initialize the feed forward network with some important
        information.

        PARAM random_gen : numpy.random.RandomState
        A random number generator used to properly initialize the weights
        of this neural network

        PARAM input : theano.tensor.TensorType
        A symbolic variable that we'll use a minibatch of data 

        PARAM input_dim : int
        This will represent the number of input neurons in our model (size
        of a single training example's input vector)

        PARAM ouptut_dim : int 
        This will represent the number of neurons in the output layer (i.e. 
        the number of possible classifications for the input) 

        Param hidden_layers : List[int]
        This will represent an ordered list of number of neurons in each
        hidden layer of our network. The first element corresponds to the
        first hidden layer and the last element corresponds to the last. 
        This list cannot be empty
        """
        
        # We'll keep track of these sizes internally in case we need them later
        self.hidden_layer_sizes = hidden_layer_sizes

        # Now we'll build all of our hidden layers
        self.hidden_layers = []
        for i in xrange(len(hidden_layer_sizes)):
            if i == 0:
                hidden_layer = HL(
                        input=input,
                        input_dim=input_dim,
                        output_dim=hidden_layer_sizes[i],
                        random_gen=random_gen,
                    )
                self.hidden_layers.append(hidden_layer)
            else:
                hidden_layer = HL(
                        input=self.hidden_layers[i - 1].output,
                        input_dim=hidden_layer_sizes[i - 1],
                        output_dim=hidden_layer_sizes[i],
                        random_gen=random_gen,
                    )

        self.softmax_layer = SL(
                input=self.hidden_layers[-1].output,
                input_dim=hidden_layer_sizes[-1],
                output_dim=output_dim
            )

        # Let's grab the output of the softmax layer and use that as our output
        self.output = self.softmax_layer.output

        # Now let's look at what our final prediction should be
        self.predicted = T.argmax(self.output, axis=1)

    def feed_forward_network_cost(self, y, lambda_l2=0):
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
        hl_squared_sum = (self.hidden_layers[0].W ** 2).sum()
        for hidden_layer in self.hidden_layers[1:]:
            hl_squared_sum += (hidden_layer.W ** 2).sum()

        sl_squared_sum = (self.softmax_layer.W ** 2).sum()

        l2_regularization = lambda_l2 * (hl_squared_sum + sl_squared_sum)
        
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




