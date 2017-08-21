"""
We'll now use the LogisticNetwork object we built in feed_forward_network.py 
in order to tackle the MNIST dataset challenge. We will use minibatch gradient
descent to train this simplistic network model. 

References:
    - textbooks: "Pattern Recognition and Machine Learning", Christopher M. Bishop, section 4.3.2
    - websites: http://deeplearning.net/tutorial, Lisa Lab
"""

__docformat__ = 'restructedtext en'

import cPickle
import gzip
import os
import time
import urllib
from theano import function, shared, config
import theano.tensor as T 
import numpy as np
import feed_forward_network


# Let's start off by defining some constants
# EXPERIMENT!!! Play around the the learning rate!
LEARNING_RATE = 0.01
N_EPOCHS = 1000
DATASET = 'mnist.pkl.gz'
BATCH_SIZE = 20

# Time to check if we have the data and if we don't, let's download it 
print "... LOADING DATA ..." 

data_path = os.path.join(
        os.path.split(__file__)[0],
        "..",
        "data",
        DATASET
    )

if (not os.path.isfile(data_path)):
    import urllib
    origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
    print 'Downloading data from %s' % origin
    urllib.urlretrieve(origin, data_path)

# Time to build our models
print "... BUILDING MODEL ..."

# Load the dataset
data_file = gzip.open(data_path, 'rb')
training_set, validation_set, test_set = cPickle.load(data_file)
data_file.close()

# Define a quick function to established a shared dataset (for efficiency)

def shared_dataset(data_xy):
    """
    We store the data in a shared variable because it allows Theano to copy it
    into GPU memory (if GPU utilization is enabled). By default, if a variable is
    not shared, it is moved to GPU at every use. This results in a big performance
    hit because that means the data will be copied one minibatch at a time. Instead,
    if we use shared variables, we don't have to worry about copying data 
    repeatedly.
    """

    data_x, data_y = data_xy
    shared_x = shared(np.asarray(data_x, dtype=config.floatX), borrow=True)
    shared_y = shared(np.asarray(data_y, dtype='int32'), borrow=True)
    return shared_x, shared_y

# We now instantiate the shared datasets
training_set_x , training_set_y = shared_dataset(training_set)
validation_set_x, validation_set_y = shared_dataset(validation_set)
test_set_x, test_set_y = shared_dataset(test_set) 

# Lets compute the number of minibatches for training, validation, and testing
n_training_batches = training_set_x.get_value(borrow=True).shape[0] / BATCH_SIZE
n_validation_batches = validation_set_x.get_value(borrow=True).shape[0] / BATCH_SIZE
n_test_batches = test_set_x.get_value(borrow=True).shape[0] / BATCH_SIZE

# Now it's time for us to build the model! 
#Let's start of with an index to the minibatch we're using
index = T.lscalar() 

# Generate symbolic variables for the input (a minibatch)
x = T.dmatrix('x')
y = T.ivector('y')

# Create a random number generator for seeding weight initialization
random_gen = np.random.RandomState(1234)

# Construct the logistic network model
# Keep in mind MNIST image is of size (28, 28)
# Also number of output class is is 10 (digits 0, 1, ..., 9)
model = feed_forward_network.FeedForwardNetwork(
        random_gen=random_gen,
        input=x, 
        input_dim=28*28, 
        output_dim=10,
        hidden_layer_sizes=[500]
    )

# Obtain a symbolic expression for the objective function
# EXPERIMENT!!! Play around with L2 regression parameter!
objective = model.feed_forward_network_cost(y, lambda_l2=0.0001)

# Obtain a symbolic expression for the error incurred
error = model.error_rate(y)

# Compute symbolic gradients of objective with respect to model parameters
updates = []
for hidden_layer in model.hidden_layers:
    dW = T.grad(objective, hidden_layer.W)
    db = T.grad(objective, hidden_layer.b) 
    updates.append((hidden_layer.W, hidden_layer.W - LEARNING_RATE * dW))
    updates.append((hidden_layer.b, hidden_layer.b - LEARNING_RATE * db))

dW = T.grad(objective, model.softmax_layer.W)
db = T.grad(objective, model.softmax_layer.b)
updates.append((model.softmax_layer.W, model.softmax_layer.W - LEARNING_RATE * dW))
updates.append((model.softmax_layer.b, model.softmax_layer.b - LEARNING_RATE * db))

# Compile theano function for training with a minibatch
train_model = function(
        inputs=[index],
        outputs=objective, 
        updates=updates,
        givens={
            x : training_set_x[index * BATCH_SIZE : (index + 1) * BATCH_SIZE],
            y : training_set_y[index * BATCH_SIZE : (index + 1) * BATCH_SIZE]
        }
    )

# Compile theano functions for validation and testing
validate_model = function(
        inputs=[index],
        outputs=error,
        givens={
            x : validation_set_x[index * BATCH_SIZE : (index + 1) * BATCH_SIZE],
            y : validation_set_y[index * BATCH_SIZE : (index + 1) * BATCH_SIZE]
        }
    )

test_model = function(
        inputs=[index],
        outputs=error,
        givens={
            x : test_set_x[index * BATCH_SIZE : (index + 1) * BATCH_SIZE],
            y : test_set_y[index * BATCH_SIZE : (index + 1) * BATCH_SIZE]
        }
    )


# Let's set up the early stopping parameters (based on the validation set)

# Must look at this many examples no matter what
patience = 10000

# Wait this much longer if a new best is found                     
patience_increase = 2

# This is when an improvement is significant
improvement_threshold = 0.995

# We go through this number of minbatches before we check on the validation set
validation_freq = min(n_training_batches, patience / 2)

# We keep of the best loss on the validation set here
best_loss = np.inf

# We also keep track of the epoch we are in
epoch = 0

# A boolean flag that propagates when patience has been exceeded
exceeded_patience = False

# Now we're ready to start training the model
print "... TRAINING MODEL ..."
start_time = time.clock()
while (epoch < N_EPOCHS) and not exceeded_patience:
    epoch = epoch + 1
    for minibatch_index in xrange(n_training_batches):
        minibatch_objective = train_model(minibatch_index)
        iteration = (epoch - 1) * n_training_batches + minibatch_index

        if (iteration + 1) % validation_freq == 0:
            # Compute loss on validation set
            validation_losses = [validate_model(i) for i in xrange(n_validation_batches)]
            validation_loss = np.mean(validation_losses)

            print 'epoch %i, minibatch %i/%i, validation error: %f %%' % (
                    epoch,
                    minibatch_index + 1,
                    n_training_batches,
                    validation_loss * 100
                )

            if validation_loss < best_loss:
                if validation_loss < best_loss * improvement_threshold:
                    patience = max(patience, iteration * patience_increase)
                best_loss = validation_loss

        if patience <= iteration:
            exceeded_patience = True
            break
end_time = time.clock()

# Let's compute how well we do on the test set
test_losses = [test_model(i) for i in xrange(n_test_batches)]
test_loss = np.mean(test_losses)

# Print out the results!
print '\n'
print 'Optimization complete with best validation score of %f %%' % (best_loss * 100)
print 'And with a test score of %f %%' % (test_loss * 100)
print '\n'
print 'The code ran for %d epochs and for a total time of %.1f seconds' % (epoch, end_time - start_time)
print '\n'

