import numpy as np
import csv

char_2_index = {}
index_2_char = {}

sentiment_2_index = {}
index_2_sentiment = {}

POSITIVE = 'positive'
NEGATIVE = 'negative'

train_dataset_raw = []
val_dataset_raw = []


with open('data/twitter/training.1600000.processed.noemoticon.csv', 'rb') as f:
    reader = csv.reader(f)
    train_dataset_raw = list(reader)

with open('data/twitter/testdata.manual.2009.06.14.csv', 'rb') as f:
    reader = csv.reader(f)
    val_dataset_raw = list(reader)

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

class TweetDataset:
    def __init__(self, dataset, char_2_index, max_row, get_all=False):
        self.inputs = []
        self.tags = []
        self.ptr = 0
        self.get_all = get_all

        for row in dataset:
            if len(row[5]) > 200:
                continue
            # print row[1]
            if row[0] == '4':
                self.tags.append(1)
            elif row[0] == '0':
                self.tags.append(0)
            else:
                print "ERROR ON:", row
                continue
            cur_input = []
            for char in row[5]:
                cur_input.append(char_2_index[char])
            cur_input = np.eye(len(char_2_index.keys()))[cur_input]
            init_len = len(cur_input)
            # print init_len
            if 200 - init_len > 0:
                zero = np.zeros((max_row - init_len, len(char_2_index.keys())))
                cur_input = np.concatenate((cur_input, zero))
            self.inputs.append(cur_input)

        self.inputs = np.array(self.inputs, dtype=np.float32)
        self.tags = np.eye(2)[self.tags]

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
