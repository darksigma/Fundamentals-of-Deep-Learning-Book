import dbm, os
import cPickle as pickle
from gensim.models import Word2Vec
import numpy as np

def save_model(model, directory):
    model.init_sims() # making sure syn0norm is initialised
    if not os.path.exists(directory):
        os.makedirs(directory)
    # Saving indexes as DBM'ed dictionary
    word_to_index = dbm.open(os.path.join(directory, 'word_to_index'), 'n')
    index_to_word = dbm.open(os.path.join(directory, 'index_to_word'), 'n')
    for key in model.vocab.keys():
        word_to_index[key.encode('utf8')] = pickle.dumps(model.vocab[key])
        index_to_word[str(model.vocab[key].index)] = key.encode('utf8')
    word_to_index.close()
    index_to_word.close()
    # Memory-mapping normalised word vectors
    syn0norm_m = np.memmap(os.path.join(directory, 'syn0norm.dat'), dtype='float32', mode='w+', shape=model.syn0norm.shape)
    syn0norm_m[:] = model.syn0norm[:]
    syn0norm_m.flush()
    # And pickling model object, witout data
    vocab, syn0norm, syn0, index2word = model.vocab, model.syn0norm, model.syn0, model.index2word
    model.vocab, model.syn0norm, model.syn0, model.index2word = None, None, None, None
    model_f = open(os.path.join(directory, 'model.pickle'), 'w')
    pickle.dump(model, model_f)
    model_f.close()
    model.vocab, model.syn0norm, model.syn0, model.index2word = vocab, syn0norm, syn0, index2word

def load_model(directory):
    model = pickle.load(open(os.path.join(directory, 'model.pickle')))
    model.vocab = DBMPickledDict(os.path.join(directory, 'word_to_index'))
    model.index2word = DBMPickledDict(os.path.join(directory, 'index_to_word'))
    model.syn0norm = np.memmap(os.path.join(directory, 'syn0norm.dat'), dtype='float32', mode='r', shape=(len(model.vocab.keys()), model.layer1_size))
    model.syn0 = model.syn0norm
    return model


class DBMPickledDict(dict):
    def __init__(self, dbm_file):
        self._dbm = dbm.open(dbm_file, 'r')
    def __setitem__(self, key, value):
        raise Exception("Read-only vocabulary")
    def __delitem__(self, key):
        raise Exception("Read-only vocabulary")
    def __iter__(self):
        return iter(self._dbm.keys())
    def __len__(self):
        return len(self._dbm)
    def __contains__(self, key):
        if isinstance(key, int):
            key = str(key)
        return key in self._dbm
    def __getitem__(self, key):
        if isinstance(key, int):
            key = str(key)
            return self._dbm[key]
        else:
            return pickle.loads(self._dbm[key])
    def keys(self):
        return self._dbm.keys()
    def values(self):
        return [self._dbm[key] for key in self._dbm.keys()]
    def itervalues(self):
        return (self._dbm[key] for key in self._dbm.keys())
