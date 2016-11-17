import numpy as np
import gensim, leveldb, os, re

db = None

tags_to_index = {}
index_to_tags = {}
train_dataset_raw = {}
train_dataset = []
test_dataset_raw = {}
test_dataset = []
dataset_vocab = {}

print "LOADING PRETRAINED WORD2VEC MODEL... "
if not os.path.isdir("data/word2vecdb"):
    model = gensim.models.Word2Vec.load_word2vec_format('/Users/nikhilbuduma/Downloads/GoogleNews-vectors-negative300.bin', binary=True)
    db = leveldb.LevelDB("data/word2vecdb")

    try:
        os.remove("data/pos_data/pos.train.processed.txt")
    except OSError:
        pass

    try:
        os.remove("data/pos_data/pos.test.processed.txt")
    except OSError:
        pass

    with open("data/pos_data/pos.train.txt") as f:
        train_dataset_raw = f.readlines()
        train_dataset_raw = [element.split() for element in train_dataset_raw if len(element.split()) > 0]

    counter = 0
    while counter < len(train_dataset_raw):
        pair = train_dataset_raw[counter]
        if counter < len(train_dataset_raw) - 1:
            next_pair = train_dataset_raw[counter + 1]
            if (pair[0] + "_" + next_pair[0] in model) and (pair[1] == next_pair[1]):
                train_dataset.append([pair[0] + "_" + next_pair[0], pair[1]])
                counter += 2
                continue

        word = re.sub("\d", "#", pair[0])
        word = re.sub("-", "_", word)

        if word in model:
            train_dataset.append([word, pair[1]])
            counter += 1
            continue

        if "_" in word:
            subwords = word.split("_")
            for subword in subwords:
                if not (subword.isspace() or len(subword) == 0):
                    train_dataset.append([subword, pair[1]])
            counter += 1
            continue

        train_dataset.append([word, pair[1]])
        counter += 1

    with open('data/pos_data/pos.train.processed.txt', 'w') as train_file:
        for item in train_dataset:
            train_file.write("%s\n" % (item[0] + " " + item[1]))


    with open("data/pos_data/pos.test.txt") as f:
        test_dataset_raw = f.readlines()
        test_dataset_raw = [element.split() for element in test_dataset_raw if len(element.split()) > 0]

    counter = 0
    while counter < len(test_dataset_raw):
        pair = test_dataset_raw[counter]
        if counter < len(test_dataset_raw) - 1:
            next_pair = test_dataset_raw[counter + 1]
            if (pair[0] + "_" + next_pair[0] in model) and (pair[1] == next_pair[1]):
                test_dataset.append([pair[0] + "_" + next_pair[0], pair[1]])
                counter += 2
                continue

        word = re.sub("\d", "#", pair[0])
        word = re.sub("-", "_", word)

        if word in model:
            test_dataset.append([word, pair[1]])
            counter += 1
            continue

        if "_" in word:
            subwords = word.split("_")
            for subword in subwords:
                if not (subword.isspace() or len(subword) == 0):
                    test_dataset.append([subword, pair[1]])
            counter += 1
            continue

        test_dataset.append([word, pair[1]])
        counter += 1

    with open('data/pos_data/pos.test.processed.txt', 'w') as test_file:
        for item in test_dataset:
            test_file.write("%s\n" % (item[0] + " " + item[1]))

    counter = 0
    for pair in train_dataset + test_dataset:
        dataset_vocab[pair[0]] = 1
        if pair[1] not in tags_to_index:
            tags_to_index[pair[1]] = counter
            index_to_tags[counter] = pair[1]
            counter += 1

    nonmodel_cache = {}

    counter = 1
    total = len(dataset_vocab.keys())
    for word in dataset_vocab:
        if counter % 100 == 0:
            print "Inserted %d words out of %d total" % (counter, total)
        if word in model:
            db.Put(word, model[word])
        elif word in nonmodel_cache:
            db.Put(word, nonmodel_cache[word])
        else:
            print word
            nonmodel_cache[word] = np.random.uniform(-0.25, 0.25, 300).astype(np.float32)
            db.Put(word, nonmodel_cache[word])
        counter += 1
else:
    db = leveldb.LevelDB("data/word2vecdb")

    with open("data/pos_data/pos.train.processed.txt") as f:
        train_dataset = f.readlines()
        train_dataset = [element.split() for element in train_dataset if len(element.split()) > 0]

    with open("data/pos_data/pos.test.processed.txt") as f:
        test_dataset = f.readlines()
        test_dataset = [element.split() for element in test_dataset if len(element.split()) > 0]

    counter = 0
    for pair in train_dataset + test_dataset:
        dataset_vocab[pair[0]] = 1
        if pair[1] not in tags_to_index:
            tags_to_index[pair[1]] = counter
            index_to_tags[counter] = pair[1]
            counter += 1




class POSDataset():
    def __init__(self, db, dataset, tags_to_index, get_all=False):
        self.db = db
        self.inputs = []
        self.tags = []
        self.ptr = 0
        self.n = 0
        self.get_all = get_all

        for pair in dataset:
            self.inputs.append(np.fromstring(db.Get(pair[0]), dtype=np.float32))
            self.tags.append(tags_to_index[pair[1]])

        self.inputs = np.array(self.inputs, dtype=np.float32)
        self.tags = np.eye(len(tags_to_index.keys()))[self.tags]

    def prepare_n_gram(self, n):
        self.n = n

    def minibatch(self, size):
        batch_inputs = []
        batch_tags = []
        if self.get_all:
            counter = 0
            while counter < len(self.inputs) - self.n + 1:
                batch_inputs.append(self.inputs[counter:counter+self.n].flatten())
                batch_tags.append(self.tags[counter + self.n - 1])
                counter += 1
        elif self.ptr + size < len(self.inputs) - self.n:
            counter = self.ptr
            while counter < self.ptr + size:
                batch_inputs.append(self.inputs[counter:counter+self.n].flatten())
                batch_tags.append(self.tags[counter + self.n - 1])
                counter += 1
        else:
            counter = self.ptr
            while counter < len(self.inputs) - self.n + 1:
                batch_inputs.append(self.inputs[counter:counter+self.n].flatten())
                batch_tags.append(self.tags[counter + self.n - 1])
                counter += 1

            counter2 = 0
            while counter2 < size - counter + self.ptr:
                batch_inputs.append(self.inputs[counter2:counter2+self.n].flatten())
                batch_tags.append(self.tags[counter2 + self.n - 1])
                counter2 += 1

        self.ptr = (self.ptr + size) % (len(self.inputs) - self.n)
        return np.array(batch_inputs, dtype=np.float32), np.array(batch_tags)



train = POSDataset(db, train_dataset, tags_to_index)
test = POSDataset(db, test_dataset, tags_to_index, get_all=True)
