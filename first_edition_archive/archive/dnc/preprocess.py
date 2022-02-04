import sys
import pickle
import getopt
import urllib2
import tarfile
import numpy as np
from shutil import rmtree
from os import listdir, mkdir
from os.path import join, isfile, isdir, dirname, basename, normpath, realpath, exists, getsize

def llprint(message):
    sys.stdout.write(message)
    sys.stdout.flush()


def create_dictionary(files_list):
    """
    creates a dictionary of unique lexicons in the dataset and their mapping to numbers

    Parameters:
    ----------
    files_list: list
        the list of files to scan through

    Returns: dict
        the constructed dictionary of lexicons
    """

    lexicons_dict = {}
    id_counter = 0

    llprint("Creating Dictionary ... 0/%d" % (len(files_list)))

    for indx, filename in enumerate(files_list):
        with open(filename, 'r') as fobj:
            for line in fobj:

                # first seperate . and ? away from words into seperate lexicons
                line = line.replace('.', ' .')
                line = line.replace('?', ' ?')
                line = line.replace(',', ' ')

                for word in line.split():
                    if not word.lower() in lexicons_dict and word.isalpha():
                        lexicons_dict[word.lower()] = id_counter
                        id_counter += 1

        llprint("\rCreating Dictionary ... %d/%d" % ((indx + 1), len(files_list)))

    print "\rCreating Dictionary ... Done!"
    return lexicons_dict


def encode_data(files_list, lexicons_dictionary, length_limit=None):
    """
    encodes the dataset into its numeric form given a constructed dictionary

    Parameters:
    ----------
    files_list: list
        the list of files to scan through
    lexicons_dictionary: dict
        the mappings of unique lexicons

    Returns: tuple (dict, int)
        the data in its numeric form, maximum story length
    """

    files = {}
    story_inputs = None
    story_outputs = None
    stories_lengths = []
    answers_flag = False  # a flag to specify when to put data into outputs list
    limit = length_limit if not length_limit is None else float("inf")

    llprint("Encoding Data ... 0/%d" % (len(files_list)))

    for indx, filename in enumerate(files_list):

        files[filename] = []

        with open(filename, 'r') as fobj:
            for line in fobj:

                # first seperate . and ? away from words into seperate lexicons
                line = line.replace('.', ' .')
                line = line.replace('?', ' ?')
                line = line.replace(',', ' ')

                answers_flag = False  # reset as answers end by end of line

                for i, word in enumerate(line.split()):

                    if word == '1' and i == 0:
                        # beginning of a new story
                        if not story_inputs is None:
                            stories_lengths.append(len(story_inputs))
                            if len(story_inputs) <= limit:
                                files[filename].append({
                                    'inputs':story_inputs,
                                    'outputs': story_outputs
                                })
                        story_inputs = []
                        story_outputs = []

                    if word.isalpha() or word == '?' or word == '.':
                        if not answers_flag:
                            story_inputs.append(lexicons_dictionary[word.lower()])
                        else:
                            story_inputs.append(lexicons_dictionary['-'])
                            story_outputs.append(lexicons_dictionary[word.lower()])

                        # set the answers_flags if a question mark is encountered
                        if not answers_flag:
                            answers_flag = (word == '?')

        llprint("\rEncoding Data ... %d/%d" % (indx + 1, len(files_list)))

    print "\rEncoding Data ... Done!"
    return files, stories_lengths


if __name__ == '__main__':
    task_dir = dirname(realpath(__file__))
    options,_ = getopt.getopt(sys.argv[1:], '', ['length_limit='])
    data_dir = join(task_dir, "../data/babi-en-10k/")
    joint_train = True
    length_limit = None
    files_list = []

    if not exists(join(task_dir, 'data')):
        mkdir(join(task_dir, 'data'))

    for opt in options:
        if opt[0] == '--length_limit':
            length_limit = int(opt[1])

    """if data_dir is None:
        raise ValueError("data_dir argument cannot be None")"""

    for entryname in listdir(data_dir):
        entry_path = join(data_dir, entryname)
        if isfile(entry_path):
            files_list.append(entry_path)

    lexicon_dictionary = create_dictionary(files_list)
    lexicon_count = len(lexicon_dictionary)

    # append used punctuation to dictionary
    lexicon_dictionary['?'] = lexicon_count
    lexicon_dictionary['.'] = lexicon_count + 1
    lexicon_dictionary['-'] = lexicon_count + 2

    encoded_files, stories_lengths = encode_data(files_list, lexicon_dictionary, length_limit)

    stories_lengths = np.array(stories_lengths)
    length_limit = np.max(stories_lengths) if length_limit is None else length_limit
    print "Total Number of stories: %d" % (len(stories_lengths))
    print "Number of stories with lengthes > %d: %d (%% %.2f) [discarded]" % (length_limit, np.sum(stories_lengths > length_limit), np.mean(stories_lengths > length_limit) * 100.0)
    print "Number of Remaining Stories: %d" % (len(stories_lengths[stories_lengths <= length_limit]))

    processed_data_dir = join(task_dir, 'data', basename(normpath(data_dir)))
    train_data_dir = join(processed_data_dir, 'train')
    test_data_dir = join(processed_data_dir, 'test')
    if exists(processed_data_dir) and isdir(processed_data_dir):
        rmtree(processed_data_dir)

    mkdir(processed_data_dir)
    mkdir(train_data_dir)
    mkdir(test_data_dir)

    llprint("Saving processed data to disk ... ")

    pickle.dump(lexicon_dictionary, open(join(processed_data_dir, 'lexicon-dict.pkl'), 'wb'))

    joint_train_data = []

    for filename in encoded_files:
        if filename.endswith("test.txt"):
            pickle.dump(encoded_files[filename], open(join(test_data_dir, basename(filename) + '.pkl'), 'wb'))
        elif filename.endswith("train.txt"):
            joint_train_data.extend(encoded_files[filename])

    pickle.dump(joint_train_data, open(join(train_data_dir, 'train.pkl'), 'wb'))

    llprint("Done!\n")
