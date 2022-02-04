import numpy as np
import cPickle as pickle
import csv, leveldb, os

char_2_index = {}
index_2_char = {}

sentiment_2_index = {}
index_2_sentiment = {}

POSITIVE = 'positive'
NEGATIVE = 'negative'

train_dataset_raw = []
val_dataset_raw = []

db = None
train_minibatches = 0

if not os.path.isdir("data/twitter/tweetdb"):

    db = leveldb.LevelDB("data/twitter/tweetdb")


    with open('data/twitter/training.1600000.processed.noemoticon.csv', 'rb') as f:
        reader = csv.reader(f)
        train_dataset_raw = list(reader)
        np.random.shuffle(train_dataset_raw)

    with open('data/twitter/testdata.manual.2009.06.14.csv', 'rb') as f:
        reader = csv.reader(f)
        val_dataset_raw = list(reader)
        np.random.shuffle(val_dataset_raw)

    counter_c = 0
    counter_s = 0
    max_row = 0
    for row in train_dataset_raw + val_dataset_raw:
        # print row[0], row[5]
        max_row = max(len(row[5]), max_row)
        if row[0] not in sentiment_2_index:
            sentiment_2_index[row[0]] = counter_s
            index_2_sentiment[counter_s] = row[0]
            counter_s += 1
        for char in row[5]:
            if char not in char_2_index:
                char_2_index[char] = counter_c
                index_2_char[counter_c] = char
                counter_c += 1

    print "Dataset has max length %d" % max_row

    print index_2_char

    print index_2_sentiment

    train_minibatches = 0
    inputs = []
    tags = []
    for row in train_dataset_raw:
        if len(row[5]) > 200:
            continue
        # print row[1]
        if row[0] == '4':
            tags.append(1)
        elif row[0] == '0':
            tags.append(0)
        else:
            print "ERROR ON:", row
            continue

        print row[0], row[5]

        cur_input = []
        for char in row[5]:
            cur_input.append(char_2_index[char])
        cur_input = np.eye(len(char_2_index.keys()))[cur_input]
        init_len = len(cur_input)
        if 200 - init_len > 0:
            zero = np.zeros((200 - init_len, len(char_2_index.keys())))
            cur_input = np.concatenate((cur_input, zero))
        inputs.append(cur_input)

        print len(tags)

        if len(inputs) == 256:
            print "FINISH MINIBATCH %d, INSERT INTO DB" % train_minibatches
            inputs = np.array(inputs, dtype=np.float32)
            tags = np.eye(2, dtype=np.float32)[tags]
            db.Put("train_inputs_" + str(train_minibatches), inputs)
            db.Put("train_tags_" + str(train_minibatches), tags)
            train_minibatches += 1

            inputs = []
            tags = []

    db.Put("n_minibatches", pickle.dumps(train_minibatches))

    inputs = []
    tags = []

    for row in val_dataset_raw:
        if len(row[5]) > 200:
            continue
        # print row[1]
        if row[0] == '4':
            tags.append(1)
        elif row[0] == '0':
            tags.append(0)
        else:
            print "ERROR ON:", row
            continue
        cur_input = []
        for char in row[5]:
            cur_input.append(char_2_index[char])
        cur_input = np.eye(len(char_2_index.keys()))[cur_input]
        init_len = len(cur_input)
        if 200 - init_len > 0:
            zero = np.zeros((200 - init_len, len(char_2_index.keys())))
            cur_input = np.concatenate((cur_input, zero))
        inputs.append(cur_input)

        if len(inputs) == 256:
            inputs = np.array(inputs, dtype=np.float32)
            tags = np.eye(2, dtype=np.float32)[tags]
            db.Put("val_inputs_0", inputs)
            db.Put("val_tags_0", tags)
            break
else:
    db = leveldb.LevelDB("data/twitter/tweetdb")
    train_minibatches = pickle.loads(db.Get("n_minibatches"))





class TweetDataset:
    def __init__(self, db, max_minibatch, prefix):
        self.ptr = 0
        self.prefix = prefix
        self.max_minibatch = max_minibatch

    def minibatch(self):
        inputs, tags = np.fromstring(db.Get(self.prefix + "_inputs_" + str(self.ptr)), dtype=np.float32).reshape((-1, 200, 194)), np.fromstring(db.Get(self.prefix + "_tags_" + str(self.ptr)), dtype=np.float32).reshape((-1, 2))
        self.ptr = (self.ptr + 1) % self.max_minibatch
        return inputs, tags


print "Start train dataset loading"

train = TweetDataset(db, train_minibatches, "train")

print "Start val dataset loading"

val = TweetDataset(db, 1, "val")

print "Finish dataset loading"
