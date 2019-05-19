import tensorflow as tf
import numpy as np
import gym
import matplotlib.pyplot as plt
import os
import coding_challenge
import matplotlib.animation
from collections import deque
import random
import math
from tqdm import tqdm

ACTIONS = 100
GAMMA = 0.99
REPLAY_MEMORY = 50000
OBSERVE = REPLAY_MEMORY // 96
EXPLORE = 100000
FINAL_EPSILON = 0.0001
INITIAL_EPSILON = 0.1
BATCH = 32
NR_EPOCHS = 1000000
LEARNING_RATE = 0.001


def animate(history):
    frames = len(history)
    print("Rendering {} frames...".format(frames))
    
    M = np.reshape(history[0], [10, 10])
    
    def render_frame(i):
        M = np.reshape(history[i], [10, 10])
        # Render grid
        matrice.set_array(M)
    
    fig, ax = plt.subplots()
    matrice = ax.matshow(M, vmin=0, vmax=1)
    plt.colorbar(matrice)
    anim = matplotlib.animation.FuncAnimation(
        fig, render_frame, frames=frames, interval=100, repeat=True
    )
    
    plt.show()


class Model:
    @staticmethod
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.01)
        return tf.Variable(initial)

    @staticmethod
    def bias_variable(shape):
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial)

    @staticmethod
    def conv2d(x, W, stride):
        return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="SAME")

    @staticmethod
    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    
    def __init__(self, board_side_length, nr_actions, batch_size, learning_rate=0.005):
        self.board_side_length = board_side_length
        self.nr_states = board_side_length * board_side_length
        self.nr_actions = nr_actions
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        self.W_conv1 = self.weight_variable([3, 3, 1, 32])
        self.b_conv1 = self.bias_variable([32])
        
        self.W_conv2 = self.weight_variable([3, 3, 32, 64])
        self.b_conv2 = self.bias_variable([64])
        
        self.W_conv3 = self.weight_variable([3, 3, 64, 64])
        self.b_conv3 = self.bias_variable([64])
        
        self.W_fc1 = self.weight_variable([576, 256])
        self.b_fc1 = self.bias_variable([256])
        
        self.W_fc2 = self.weight_variable([256, nr_actions])
        self.b_fc2 = self.bias_variable([nr_actions])
        
        self.define_model()
    
    def define_model(self):
        # input layer
        self.s = tf.placeholder("float", [None, self.board_side_length, self.board_side_length, 1])
        self.q_s_a = tf.placeholder("float", [None, self.nr_actions])
        
        # hidden layers
        self.h_conv1 = tf.nn.relu(self.conv2d(self.s, self.W_conv1, 2) + self.b_conv1)
        self.h_conv2 = tf.nn.relu(self.conv2d(self.h_conv1, self.W_conv2, 2) + self.b_conv2)
        self.h_conv3 = tf.nn.relu(self.conv2d(self.h_conv2, self.W_conv3, 1) + self.b_conv3)
        self.h_conv3_flat = tf.reshape(self.h_conv3, [-1, 576])
        self.h_fc1 = tf.nn.relu(tf.matmul(self.h_conv3_flat, self.W_fc1) + self.b_fc1)
        self.logits = tf.matmul(self.h_fc1, self.W_fc2) + self.b_fc2
        
        loss = tf.losses.mean_squared_error(self.q_s_a, self.logits)
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(loss)
        
        self.var_init = tf.global_variables_initializer()
    
    def predict_one(self, state, sess):
        return sess.run(self.logits, feed_dict={
            self.s: state.reshape([1, 10, 10, 1])
        })
    
    def predict_batch(self, states, sess):
        return sess.run(self.logits, feed_dict={
            self.s: states
        })
    
    def train_batch(self, sess, x_batch, y_batch):
        sess.run(self.optimizer, feed_dict={
            self.s: x_batch,
            self.q_s_a: y_batch
        })


class ReplayMemory:
    def __init__(self, max_memory):
        self.max_memory = max_memory
        self.samples = deque()
        
    def add_sample(self, sample):
        self.samples.append(sample)
        if len(self.samples) > self.max_memory:
            self.samples.popleft()
            
    def sample(self, nr_samples):
        if nr_samples > len(self.samples):
            return random.sample(self.samples, len(self.samples))
        else:
            return random.sample(self.samples, nr_samples)


class Game:
    @staticmethod
    def toAction(a):
        y = a % 10 / 10
        x = (a // 10) / 10
        return np.array([x, y])

    @staticmethod
    def fromAction(action):
        x = int(action[0] * 10)
        y = int(action[0] * 10)
        return 10 * x + y
    
    def __init__(self, sess, model, env, memory, max_eps, min_eps, lam, gamma):
        self.sess = sess
        self.model = model
        self.env = env
        self.memory = memory
        self.eps = max_eps
        self.max_eps = max_eps
        self.min_eps = min_eps
        self.lam = lam
        self.gamma = gamma
        self.nr_games = 0
        
    def run(self, training=True):
        steps = 0
        state = self.env.reset()
        terminal = False
        while not terminal:
            action = self.choose_action(state.reshape(-1))
            next_state, reward, terminal, info = self.env.step(self.toAction(action))
            
            if terminal:
                next_state = None
            
            self.memory.add_sample((state, action, reward, next_state))
            
            if training:
                self.replay()
            
            state = next_state
            steps += 1

            self.eps = self.min_eps + (self.max_eps - self.min_eps) * math.exp(-self.lam * self.nr_games)
            
        self.nr_games += 1
        
        return steps

    @staticmethod
    def randomAction(state):
        choices = list()
        count = 0
        for i in state:
            if i == 0:
                choices.append(count)
            count += 1
    
        if len(choices) == 0:
            print(state.reshape([10, 10]))
        return random.choice(choices)
        
    @staticmethod
    def is_legal(index, state):
        if state[index] == 0.0:
            return True
        else:
            return False
        
    def choose_action(self, state):
        if random.random() < self.eps:
            action = self.randomAction(state)
        else:
            all_actions = self.model.predict_one(state, self.sess).reshape([-1])
            legal_actions = [x if self.is_legal(index, state) else -np.inf for index, x in enumerate(all_actions)]
            action = np.argmax(legal_actions)
            
        return action
    
    def replay(self):
        batch = self.memory.sample(self.model.batch_size)
        states = np.array([val[0] for val in batch])
        next_states = np.array([(np.zeros((self.model.board_side_length, self.model.board_side_length, 1))
                                if val[3] is None else val[3]) for val in batch])
        
        q_s_a = self.model.predict_batch(states, self.sess)
        q_s_a_next = self.model.predict_batch(next_states, self.sess)
        
        x = np.zeros((len(batch), self.model.board_side_length, self.model.board_side_length, 1))
        y = np.zeros((len(batch), self.model.nr_actions))
        
        for i, b in enumerate(batch):
            state, action, reward, next_state = b
            current_q = q_s_a[i]
            if next_state is None:
                current_q[action] = reward
            else:
                current_q[action] = reward + self.gamma * np.amax(q_s_a_next[i])
                
            x[i] = state.reshape([self.model.board_side_length, self.model.board_side_length, 1])
            y[i] = current_q
            
        self.model.train_batch(self.sess, x, y)


def plot_game_lengths(average_over, game_lengths):
    running_average_length = [np.mean(game_lengths[i:i + average_over]) for i in
                              range(len(game_lengths) - average_over)]
    plt.plot(running_average_length)
    plt.show()


def playGame():
    environment = gym.make('Battleship-v0')
    
    board_size = 10
    nr_actions = 100
    
    model = Model(nr_actions=nr_actions, board_side_length=board_size, batch_size=96, learning_rate=0.00005)
    memory = ReplayMemory(50000)
    
    saver = tf.train.Saver(max_to_keep=0)
    
    with tf.Session() as sess:
        sess.run(model.var_init)
        
        game = Game(sess, model, environment, memory, max_eps=0.9, min_eps=0.01, lam=0.99995, gamma=0.99)
        num_epochs = 100000

        game_lengths = list()
        with tqdm(range(num_epochs), unit="games") as t:
            for i in t:
                game_lengths.append(game.run())
                
                if i != 0 and i % 1000 == 0:
                    saver.save(sess, 'saved_networks/' + 'dqn', global_step=i)
    
                    window_size = 500
                    running_average_length = [np.mean(game_lengths[i:i + window_size]) for i in
                                              range(len(game_lengths) - window_size)]
                    plt.plot(running_average_length)
                    plt.savefig('dqn-' + str(i) + '.png')
                    
                if i > 500 and i % 500 == 0:
                    plot_game_lengths(500, game_lengths)

                t.set_postfix(train_game_length=np.mean(game_lengths[-100:]))
        

if __name__ == "__main__":
    playGame()
