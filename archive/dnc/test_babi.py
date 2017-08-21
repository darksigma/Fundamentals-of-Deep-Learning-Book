# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import getopt
import shutil
import pickle
import sys
import os
import re

from mem_ops import *

def llprint(message):
    sys.stdout.write(message)
    sys.stdout.flush()

def load(path):
    return pickle.load(open(path, 'rb'))

def onehot(index, size):
    vec = np.zeros(size, dtype=np.float32)
    index = int(index)
    vec[index] = 1.0
    return vec

def prepare_sample(sample, target_code, word_space_size):
    """
    prepares the input/output sequence of a sample story by encoding it
    into one-hot vectors and generates the necessary loss weights
    """
    input_vec = np.array(sample[0]['inputs'], dtype=np.float32)
    output_vec = np.array(sample[0]['inputs'], dtype=np.float32)
    seq_len = input_vec.shape[0]
    weights_vec = np.zeros(seq_len, dtype=np.float32)

    target_mask = (input_vec == target_code)
    output_vec[target_mask] = sample[0]['outputs']
    weights_vec[target_mask] = 1.0

    input_vec = np.array([onehot(code, word_space_size) for code in input_vec])
    output_vec = np.array([onehot(code, word_space_size) for code in output_vec])

    return (
        np.reshape(input_vec, (-1, word_space_size)),
        np.reshape(output_vec, (-1, word_space_size)),
        seq_len,
        np.reshape(weights_vec, (-1, 1))
    )

task_dir = os.path.dirname(os.path.realpath(__file__))
lexicon_dictionary = load(os.path.join(task_dir, "data/babi-en-10k/lexicon-dict.pkl"))
test_files = []

test_dir = os.path.join(task_dir, 'data/babi-en-10k/test/')
for entryname in os.listdir(test_dir):
    entry_path = os.path.join(test_dir, entryname)
    if os.path.isfile(entry_path):
        test_files.append(entry_path)

# the model parameters
N = 256; W = 64; R = 4   # memory parameters
X = Y = 159  # input/output size
NN = 256  # controller's network output size
zeta_size = R*W + 3*W + 5*R + 3

def network(step_input, state):
    """
    defines the recurrent neural network operation
    """
    global NN
    step_input = tf.expand_dims(step_input, 0)
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(NN)

    return lstm_cell(step_input, state)

# START: Computaional Graph
graph = tf.Graph()
with graph.as_default():

    # placeholders
    input_data = tf.placeholder(tf.float32, [None, X])
    sequence_length = tf.placeholder(tf.int32)

    initial_nn_state = tf.nn.rnn_cell.BasicLSTMCell(NN).zero_state(1, tf.float32)

    empty_unpacked_inputs = tf.TensorArray(tf.float32, sequence_length)
    unpacked_inputs = empty_unpacked_inputs.unpack(input_data)
    outputs_container = tf.TensorArray(tf.float32, sequence_length)  # accumelates the step outputs
    t = tf.constant(0, dtype=tf.int32)

    def step_op(time, memory_state, controller_state, inputs, outputs):
        """
        defines the operation of one step of the sequence
        """
        global N, W, R

        step_input = inputs.read(time)
        M, u, p, L, wr, ww, r = memory_state

        with tf.variable_scope('controller'):
            Xt = tf.concat(0, [step_input, tf.reshape(r, [-1])])
            nn_output, nn_state = network(Xt, controller_state)
            std = lambda input_size: np.min(0.01, np.sqrt(2. / input_size))
            W_y = tf.get_variable('W_y', [NN, Y], tf.float32, tf.truncated_normal_initializer(stddev=std(NN)))
            W_zeta = tf.get_variable('W_zeta', [NN, zeta_size], tf.float32, tf.truncated_normal_initializer(stddev=std(NN)))

            pre_output = tf.matmul(nn_output, W_y)
            zeta = tf.squeeze(tf.matmul(nn_output, W_zeta))
            kr, br, kw, bw, e, v, f, ga, gw, pi = parse_interface(zeta, N, W, R)

            # write head operations
            u_t = ut(u, f, wr, ww)
            a_t = at(u_t, N)
            cw_t = C(M, kw, bw)
            ww_t = wwt(cw_t, a_t, gw, ga)
            M_t = Mt(M, ww_t, e, v)
            L_t = Lt(L, ww_t, p, N)
            p_t = pt(ww_t, p)

            # read heads operations
            cr_t = C(M_t, kr, br)
            wr_t = wrt(wr, L_t, cr_t, pi)
            r_t = rt(M_t, wr_t)

            W_r = tf.get_variable('W_r', [W*R, Y], tf.float32, tf.truncated_normal_initializer(stddev=std(W*R)))
            flat_rt = tf.reshape(r_t, [-1])
            final_output = pre_output + tf.matmul(tf.expand_dims(flat_rt, 0), W_r)
            updated_outputs = outputs.write(time, tf.squeeze(final_output))

            return time + 1, (M_t, u_t, p_t, L_t, wr_t, ww_t, r_t), nn_state, inputs, updated_outputs

    _, _, _, _, final_outputs = tf.while_loop(
        cond = lambda time, *_: time < sequence_length,
        body = step_op,
        loop_vars=(t, init_memory(N,W,R), initial_nn_state, unpacked_inputs, outputs_container),
        parallel_iterations=32,
        swap_memory=True
    )

    # pack the individual steps outputs into a single (sequence_length x Y) tensor
    packed_output = final_outputs.pack()
    softmaxed = tf.nn.softmax(packed_output)

# END: Computational Graph

    with tf.Session(graph=graph) as session:

        tf.train.Saver().restore(session, os.path.join(task_dir, 'babi-model/model.ckpt'))

        tasks_results = {}
        tasks_names = {}
        for test_file in test_files:
            test_data = load(test_file)
            task_regexp = r'qa([0-9]{1,2})_([a-z\-]*)_test.txt.pkl'
            task_filename = os.path.basename(test_file)
            task_match_obj = re.match(task_regexp, task_filename)
            task_number = task_match_obj.group(1)
            task_name = task_match_obj.group(2).replace('-', ' ')
            tasks_names[task_number] = task_name
            counter = 0
            results = []

            llprint("%s ... %d/%d" % (task_name, counter, len(test_data)))

            for story in test_data:
                astory = np.array(story['inputs'])
                questions_indecies = np.argwhere(astory == lexicon_dictionary['?'])
                questions_indecies = np.reshape(questions_indecies, (-1,))
                target_mask = (astory == lexicon_dictionary['-'])

                desired_answers = np.array(story['outputs'])
                input_vec, _, seq_len, _ = prepare_sample([story], lexicon_dictionary['-'], len(lexicon_dictionary))
                softmax_output = session.run(softmaxed, feed_dict={
                        input_data: input_vec,
                        sequence_length: seq_len
                })

                softmax_output = np.squeeze(softmax_output)
                given_answers = np.argmax(softmax_output[target_mask], axis=1)

                answers_cursor = 0
                for question_indx in questions_indecies:
                    question_grade = []
                    targets_cursor = question_indx + 1
                    while targets_cursor < len(astory) and astory[targets_cursor] == lexicon_dictionary['-']:
                        question_grade.append(given_answers[answers_cursor] == desired_answers[answers_cursor])
                        answers_cursor += 1
                        targets_cursor += 1
                    results.append(np.prod(question_grade))

                counter += 1
                llprint("\r%s ... %d/%d" % (task_name, counter, len(test_data)))

            error_rate = 1. - np.mean(results)
            tasks_results[task_number] = error_rate
            llprint("\r%s ... %.3f%% Error Rate.\n" % (task_name, error_rate * 100))

        print "\n"
        print "%-27s%-27s%s" % ("Task", "Result", "Paper's Mean")
        print "-------------------------------------------------------------------"
        paper_means = {
            '1': '9.0±12.6%', '2': '39.2±20.5%', '3': '39.6±16.4%',
            '4': '0.4±0.7%', '5': '1.5±1.0%', '6': '6.9±7.5%', '7': '9.8±7.0%',
            '8': '5.5±5.9%', '9': '7.7±8.3%', '10': '9.6±11.4%', '11':'3.3±5.7%',
            '12': '5.0±6.3%', '13': '3.1±3.6%', '14': '11.0±7.5%', '15': '27.2±20.1%',
            '16': '53.6±1.9%', '17': '32.4±8.0%', '18': '4.2±1.8%', '19': '64.6±37.4%',
            '20': '0.0±0.1%', 'mean': '16.7±7.6%', 'fail': '11.2±5.4'
        }
        for k in range(20):
            task_id = str(k + 1)
            task_result = "%.2f%%" % (tasks_results[task_id] * 100)
            print "%-27s%-27s%s" % (tasks_names[task_id], task_result, paper_means[task_id])
        print "-------------------------------------------------------------------"
        all_tasks_results = [v for _,v in tasks_results.iteritems()]
        results_mean = "%.2f%%" % (np.mean(all_tasks_results) * 100)
        failed_count = "%d" % (np.sum(np.array(all_tasks_results) > 0.05))

        print "%-27s%-27s%s" % ("Mean Err.", results_mean, paper_means['mean'])
        print "%-27s%-27s%s" % ("Failed (err. > 5%)", failed_count, paper_means['fail'])
