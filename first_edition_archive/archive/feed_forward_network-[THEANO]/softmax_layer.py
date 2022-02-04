"""
We will use this class to represent a simple softmax layer. 
This will be a building block for a simplefeedforward neural 
network.

References:
    - textbooks: "Pattern Recognition and Machine Learning", Christopher M. Bishop, section 4.3.2
    - websites: http://deeplearning.net/tutorial, Lisa Lab
"""

import numpy as np
import theano.tensor as T 
import theano

class SoftmaxLayer(object):
    """
    The softmax layer class is described by two parameters (which
    we will want to learn). The first is a incoming weight matrix. 
    We'll refer to this weight matrix as W. The second is a bias 
    vector b. Refer to the text if you want to learn more about how 
    this layer works. Let's get started!
    """

    def __init__(self, input, input_dim, output_dim):
        """
        We first initialize the softmax layer object with some important
        information.

        PARAM input : theano.tensor.TensorType
        A symbolic variable that we'll use to describe incoming data from
        the previous layer

        PARAM input_dim : int
        This will represent the number of neurons in the previous layer

        PARAM ouptut_dim : int 
        This will represent the number of neurons in the softmax layer (i.e. 
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


        



