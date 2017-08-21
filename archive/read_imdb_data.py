import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
from tflearn.datasets import imdb
import numpy as np

# IMDB Dataset loading
train, test, _ = imdb.load_data(path='data/imdb.pkl', n_words=30000,
                                valid_portion=0.1)
trainX, trainY = train
testX, testY = test

# Data preprocessing
# Sequence padding
trainX = pad_sequences(trainX, maxlen=500, value=0.)
testX = pad_sequences(testX, maxlen=500, value=0.)
# Converting labels to binary vectors
trainY = to_categorical(trainY, nb_classes=2)
testY = to_categorical(testY, nb_classes=2)



class IMDBDataset():
    def __init__(self, X, Y):
        self.num_examples = len(X)
        self.inputs = X
        self.tags = Y
        self.ptr = 0


    def minibatch(self, size):
        ret = None
        if self.ptr + size < len(self.inputs):
            ret =  self.inputs[self.ptr:self.ptr+size], self.tags[self.ptr:self.ptr+size]
        else:
            ret = np.concatenate((self.inputs[self.ptr:], self.inputs[:size-len(self.inputs[self.ptr:])])), np.concatenate((self.tags[self.ptr:], self.tags[:size-len(self.tags[self.ptr:])]))
        self.ptr = (self.ptr + size) % len(self.inputs)

        return ret
        # return np.eye(10000)[ret[0]], ret[1]


train = IMDBDataset(trainX, trainY)
val = IMDBDataset(testX, testY)
