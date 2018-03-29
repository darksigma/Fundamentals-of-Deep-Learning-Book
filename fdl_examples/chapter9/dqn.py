import tensorflow as tf
import numpy as np
import random
import gym
import tqdm
from scipy.misc import imresize
slim = tf.contrib.slim
import matplotlib.pyplot as plt
import time
import copy

def epsilon_greedy_action(action_distribution, epsilon=1e-5):
    if random.random() < epsilon:
        return np.argmax(np.random.random(
           action_distribution.shape))
    else:
        return np.argmax(action_distribution)

def epsilon_greedy_action_annealed(action_distribution,
                                   percentage, 
                                   epsilon_start=1.0, 
                                   epsilon_end=1e-8):
    annealed_epsilon = epsilon_start*(1.0-percentage) + epsilon_end*percentage
    if random.random() < annealed_epsilon:
        return np.argmax(np.random.random(
          action_distribution.shape))
    else:
        return np.argmax(action_distribution)

class EpisodeHistory(object):

    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.state_primes = []
        self.terminals = []

    def add_to_history(self, state, action, reward, 
      state_prime, terminal):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.state_primes.append(state_prime)
        self.terminals.append(terminal)

# DQNAgent

class DQNAgent(object):

    def __init__(self, session, num_actions,
                 learning_rate=1e-3, history_length=4,
                 screen_height=84, screen_width=84, 
                 gamma=0.99):
        self.session = session
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.history_length = history_length
        self.screen_height = screen_height
        self.screen_width = screen_width
        self.gamma = gamma

        self.build_prediction_network()
        self.build_target_network()
        self.build_training()

    def build_prediction_network(self):
        # print('build pred')
        with tf.variable_scope('pred_network'):
            self.s_t = tf.placeholder('float32', shape=[
                                      None, 
                                      self.history_length,
                                      self.screen_height,
                                      self.screen_width],
                                      name='state')
            self.conv_0 = slim.conv2d(self.s_t, 32, 8, 4, 
              scope='conv_0')
            self.conv_1 = slim.conv2d(self.conv_0, 64, 4, 2, 
              scope='conv_1')
            self.conv_2 = slim.conv2d(self.conv_1, 64, 3, 1, 
              scope='conv_2')

            shape = self.conv_2.get_shape().as_list()

            self.flattened = tf.reshape(
                self.conv_2, [-1, shape[1]*shape[2]*shape[3]])
            self.fc_0 = slim.fully_connected(self.flattened, 
               512, scope='fc_0')
            self.q_t = slim.fully_connected(
              self.fc_0, self.num_actions, activation_fn=None,
               scope='q_values')

            self.q_action = tf.argmax(self.q_t, dimension=1)

    def build_target_network(self):
        # print('build target')
        with tf.variable_scope('target_network'):
            self.target_s_t = tf.placeholder('float32', 
              shape=[None, self.history_length, 
                self.screen_height, self.screen_width], 
                  name='state')
            self.target_conv_0 = slim.conv2d(
                self.target_s_t, 32, 8, 4, scope='conv_0')
            self.target_conv_1 = slim.conv2d(
                self.target_conv_0, 64, 4, 2, scope='conv_1')
            self.target_conv_2 = slim.conv2d(
                self.target_conv_1, 64, 3, 1, scope='conv_2')

            shape = self.target_conv_2.get_shape().as_list()

            self.target_flattened = tf.reshape(
                self.target_conv_2, [-1, 
                  shape[1]*shape[2]*shape[3]])
            self.target_fc_0 = slim.fully_connected(
                self.target_flattened, 512, scope='fc_0')
            self.target_q = slim.fully_connected(
                self.target_fc_0, self.num_actions, 
                  activation_fn=None, scope='q_values')

            self.target_q_action = tf.argmax(self.target_q, dimension=1)

    def update_target_q_weights(self):
        # print('update target weights')
        target_vars = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope=
              'target_network')
        pred_vars = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope=
              'pred_network')
        for target_var, pred_var in zip(target_vars, pred_vars):
            weight_input = tf.placeholder('float32', 
              name='weight')
            target_var.assign(weight_input).eval(
                {weight_input: pred_var.eval()})

    def sample_and_train_pred(self, replay_table, batch_size):

        s_t, action, reward, s_t_plus_1, terminal = replay_table.sample_batch(
              batch_size)
        q_t_plus_1 = self.target_q.eval(
            {self.target_s_t: 
              s_t_plus_1})


        terminal = np.array(terminal) + 0.
        max_q_t_plus_1 = np.max(q_t_plus_1, 
          axis=1)
        target_q_t = (1. - terminal) * self.gamma * max_q_t_plus_1 + reward
        _, q_t, loss = self.session.run(
            [self.train_step, self.q_t,
              self.loss], {
            self.target_q_t: target_q_t,
            self.action: action,
            self.s_t: s_t
        })

        return q_t


    def build_training(self):
        self.target_q_t = tf.placeholder('float32', [None], 
          name='target_q_t')
        self.action = tf.placeholder('int64', [None], 
          name='action')

        action_one_hot = tf.one_hot(
            self.action, self.num_actions, 1.0, 0.0, 
              name='action_one_hot')
        q_of_action = tf.reduce_sum(
            self.q_t * action_one_hot, reduction_indices=1, 
              name='q_of_action')

        self.delta = (self.target_q_t - q_of_action)
        self.loss = tf.reduce_mean(self.clip_error(self.delta), name='loss')

        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate)
        # self.optimizer = tf.train.RMSPropOptimizer(
        #     learning_rate=self.learning_rate, momentum=0.95, epsilon=0.01)
        self.train_step = self.optimizer.minimize(self.loss)

    def sample_action_from_distribution(self, 
      action_distribution, epsilon_percentage):
        # Choose an action based on the action probability 
        # distribution
        action = epsilon_greedy_action_annealed(
            action_distribution, epsilon_percentage)
        return action

    def predict_action(self, state, epsilon_percentage):
        action_distribution = self.session.run(
            self.q_t, feed_dict={self.s_t: [state]})[0]
        action = self.sample_action_from_distribution(
            action_distribution, epsilon_percentage)
        return action

    def process_state_into_stacked_frames(self, frame, 
      past_frames, past_state=None):
        full_state = np.zeros(
            (self.history_length, self.screen_width, 
              self.screen_height))

        if past_state is not None:
            for i in range(len(past_state)-1):
                full_state[i, :, :] = past_state[i+1, :, :]
            full_state[-1, :, :] = self.preprocess_frame(frame, (self.screen_width, self.screen_height))
        else:
            all_frames = past_frames + [frame]
            for i, frame_f in enumerate(all_frames):
                full_state[i, :, :] = self.preprocess_frame(frame_f, (self.screen_width, self.screen_height))
        return full_state

    def to_grayscale(self, x):
        return np.dot(x[...,:3], [0.299, 0.587, 0.114])

    def clip_error(self, x):
      try:
        return tf.select(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)
      except:
        return tf.where(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)

    def preprocess_frame(self, im, shape):
        cropped = im[16:201,:]
        grayscaled = self.to_grayscale(cropped)
        resized = imresize(grayscaled, shape, 'nearest').astype('float32')
        mean, std = 40.45, 64.15
        frame = (resized-mean)/std
        return frame

class ExperienceReplayTable(object):

    def __init__(self, table_size=50000):
        self.states = []
        self.actions = []
        self.rewards = []
        self.state_primes = []
        self.terminals = []

        self.table_size = table_size

    def add_episode(self, episode):
        self.states += episode.states
        self.actions += episode.actions
        self.rewards += episode.rewards
        self.state_primes += episode.state_primes
        self.terminals += episode.terminals

        self.purge_old_experiences()

    def purge_old_experiences(self):
        while len(self.states) > self.table_size:
            self.states.pop(0)
            self.actions.pop(0)
            self.rewards.pop(0)
            self.state_primes.pop(0)

    def sample_batch(self, batch_size):
        s_t, action, reward, s_t_plus_1, terminal = [], [], [], [], []
        rands = np.arange(len(self.states))
        np.random.shuffle(rands)
        rands = rands[:batch_size]

        for r_i in rands:
            s_t.append(self.states[r_i])
            action.append(self.actions[r_i])
            reward.append(self.rewards[r_i])
            s_t_plus_1.append(self.state_primes[r_i])
            terminal.append(self.terminals[r_i])
        return np.array(s_t), np.array(action), np.array(reward), np.array(s_t_plus_1), np.array(terminal)


def main(argv):
    # Configure Settings
    learn_start = 15000
    # learn_start = 1
    scale = 30
    total_episodes = 500*scale
    epsilon_stop = 200*scale
    # epsilon_stop = 20
    train_frequency = 4
    target_frequency = 1000
    # batch_size = 64
    batch_size = 32
    max_episode_length = 100000
    render_start = 10
    should_render = False

    env = gym.make('Breakout-v4')
    num_actions = env.action_space.n

    solved = False
    with tf.Session() as session:
        agent = DQNAgent(session=session, 
          num_actions=num_actions, learning_rate=1e-4, history_length=4,
                 gamma=0.98)
        session.run(tf.global_variables_initializer())

        episode_rewards = []
        q_t_list = []
        batch_losses = []
        past_frames_last_time = None

        replay_table = ExperienceReplayTable()
        global_step_counter = 0
        for i in tqdm.tqdm(range(total_episodes)):
            frame = env.reset()
            past_frames = [copy.deepcopy(frame) for _ in range(agent.history_length-1)]
            state = agent.process_state_into_stacked_frames(
                frame, past_frames, past_state=None)
            episode_reward = 0.0
            episode_history = EpisodeHistory()
            epsilon_percentage = float(min(i/float(
              epsilon_stop), 1.0))
            for j in range(max_episode_length):
                action = agent.predict_action(state, 
                  epsilon_percentage)
                if global_step_counter < learn_start:
                    action = np.argmax(np.random.random((agent.num_actions)))

                reward = 0

                frame_prime, reward, terminal, _ = env.step(action)
                if terminal == True:
                    reward -= 1

                state_prime = agent.process_state_into_stacked_frames(frame_prime, past_frames, past_state=state)

                past_frames.append(frame_prime)
                past_frames = past_frames[len(past_frames)-agent.history_length:]

                past_frames_last_time = past_frames

                if (i > render_start) and should_render or (solved and should_render):
                    env.render()
                episode_history.add_to_history(
                    state, action, reward, state_prime, terminal)
                state = state_prime
                episode_reward += reward
                global_step_counter += 1

                if global_step_counter > learn_start:
                    if global_step_counter % train_frequency == 0:
                        q_t = agent.sample_and_train_pred(replay_table, batch_size)
                        q_t_list.append(q_t)

                        if global_step_counter % target_frequency == 0:
                            agent.update_target_q_weights()


                if j == (max_episode_length - 1):
                    terminal = True

                if terminal:
                    replay_table.add_episode(episode_history)
                    episode_rewards.append(episode_reward)
                    break

            if i % 50 == 0:
                ave_reward = np.mean(episode_rewards[-100:])
                ep_percent = float(min(i/float(epsilon_stop), 1.0))
                print("Reward Stats (min, max, median, mean): ", np.min(episode_rewards[-100:]), np.max(episode_rewards[-100:]), np.median(episode_rewards[-100:]), np.mean(episode_rewards[-100:]))
                print("Global Stats (ep_percent, global_step_counter): ", ep_percent, global_step_counter)
                if q_t_list:
                  print("Qt Stats (min, max, median, mean): ", np.min(q_t_list[-1000:]), np.max(q_t_list[-100:]), np.median(q_t_list[-100:]), np.mean(q_t_list[-100:]))
                if ave_reward > 50.0:
                    solved = True
                    print('solved')
                else:
                    solved = False

main('')
print('done')