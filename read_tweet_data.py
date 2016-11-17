import numpy as np

char_2_index = {}
index_2_char = {}

sentiment_2_index = {}
index_2_sentiment = {}

POSITIVE = 'positive'
NEGATIVE = 'negative'

with open("data/twitter/twitter-train-text.tsv") as f:
    train_dataset_raw = f.readlines()
    train_dataset_raw = [(" ".join(x.split("\t")[3].split()), x.split("\t")[2]) for x in train_dataset_raw]

with open("data/twitter/twitter-dev-text.tsv") as f:
    val_dataset_raw = f.readlines()
    val_dataset_raw = [(" ".join(x.split("\t")[3].split()), x.split("\t")[2]) for x in val_dataset_raw]

counter_c = 0
counter_s = 0
max_row = 0
for row in train_dataset_raw + val_dataset_raw:
    max_row = max(len(row[0]), max_row)
    if row[1] not in sentiment_2_index:
        sentiment_2_index[row[1]] = counter_s
        index_2_sentiment[counter_s] = row[1]
        counter_s += 1
    for char in row[0]:
        if char not in char_2_index:
            char_2_index[char] = counter_c
            index_2_char[counter_c] = char
            counter_c += 1

print "Dataset has max length %d" % max_row

print index_2_char

print index_2_sentiment

class TweetDataset:
    def __init__(self, dataset, char_2_index, max_row, get_all=False):
        self.inputs = []
        self.tags = []
        self.ptr = 0
        self.get_all = get_all

        for row in dataset:
            # print row[1]
            if row[1] == 'positive':
                self.tags.append(2)
            elif row[1] == 'negative':
                self.tags.append(0)
            else:
                self.tags.append(1)
            cur_input = []
            for char in row[0]:
                cur_input.append(char_2_index[char])
            cur_input = np.eye(len(char_2_index.keys()))[cur_input]
            init_len = len(cur_input)
            # print init_len
            if max_row - init_len > 0:
                zero = np.zeros((max_row - init_len, len(char_2_index.keys())))
                cur_input = np.concatenate((cur_input, zero))
            self.inputs.append(cur_input)

        self.inputs = np.array(self.inputs, dtype=np.float32)
        self.tags = np.eye(3)[self.tags]

    def minibatch(self, size):
        batch_inputs = []
        batch_tags = []
        if self.get_all:
            return self.inputs, self.tags
        elif self.ptr + size < len(self.inputs):
            start = self.ptr
            self.ptr += size
            return self.inputs[start:self.ptr], self.tags[start:self.ptr]

        start = self.ptr
        self.ptr = (self.ptr + size) % len(self.inputs)
        return np.concatenate((self.inputs[start:], self.inputs[:self.ptr])), np.concatenate((self.tags[start:], self.tags[:self.ptr]))

print "Start train dataset loading"

train = TweetDataset(train_dataset_raw, char_2_index, max_row)

print "Start val dataset loading"

val = TweetDataset(val_dataset_raw, char_2_index, max_row)

print "Finish dataset loading"
