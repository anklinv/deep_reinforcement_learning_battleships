import argparse
import tensorflow as tf
import numpy as np
import gym
import matplotlib.pyplot as plt
import os
import coding_challenge
import matplotlib.animation
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm, trange
from mpl_toolkits.axes_grid1 import make_axes_locatable


class Model:
    def __init__(self, board_size, step_size):
        self.board_size = board_size
        self.step_size = step_size
        
        self.input_positions = tf.placeholder(tf.float32, shape=(1, self.board_size))
        self.labels = tf.placeholder(tf.int64)
        self.learning_rate = tf.placeholder(tf.float32, shape=[])
        self.W1 = tf.Variable(tf.truncated_normal([self.board_size, int(2 * self.board_size)],
                                                  stddev=0.1 / np.sqrt(float(self.board_size))))
        self.b1 = tf.Variable(tf.zeros([1, int(2 * self.board_size)]))
        self.W2 = tf.Variable(tf.truncated_normal([int(2 * self.board_size), self.board_size],
                                                  stddev=0.1 / np.sqrt(float(self.board_size))))
        self.b2 = tf.Variable(tf.zeros([1, self.board_size]))
        
        self.__define_model()
        self.saver = tf.train.Saver(max_to_keep=0)
    
    def __define_model(self):
        self.h1 = tf.tanh(tf.matmul(self.input_positions, self.W1) + self.b1)
        self.logits = tf.matmul(self.h1, self.W2) + self.b2
        self.probabilities = tf.nn.softmax(self.logits)
        self.cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self.logits, labels=self.labels, name='xentropy')
        self.train_step = tf.train.GradientDescentOptimizer(
            learning_rate=self.learning_rate).minimize(self.cross_entropy)
        self.init = tf.global_variables_initializer()
    
    def save(self, sess, epoch):
        self.saver.save(sess, "models/policy_gradient", global_step=epoch)
    
    def load(self, sess, path):
        self.saver.restore(sess, path)
    
    def get_probability(self, sess, state):
        return sess.run([self.probabilities], feed_dict={
            self.input_positions: state
        })[0][0]
    
    def train(self, sess, state, action, reward):
        sess.run([self.train_step], feed_dict={
            self.input_positions: state,
            self.labels: [action],
            self.learning_rate: self.step_size * reward
        })


class Game:
    def __init__(self, sess, model, env, nr_epochs=300000, save_frequency=1000, plot_frequency=100):
        self.sess = sess
        self.model = model
        self.env = env
        self.epoch = 0
        self.game_lengths = list()
        self.nr_epochs = nr_epochs
        self.save_frequency = save_frequency
        self.plot_frequency = plot_frequency

    @staticmethod
    def __toAction(a):
        y = a % 10 / 10
        x = (a // 10) / 10
        return np.array([x, y])
        
    def __choose_action(self, state, probability, training):
        # Filter out illegal actions
        probs = [p * (state[0][index] == 0) for index, p in enumerate(probability)]
        probs = np.array(probs)
        probs /= probs.sum()
        
        if training:
            bomb_index = np.random.choice(self.model.board_size, p=probs)
        else:
            bomb_index = np.argmax(probs)
            
        return bomb_index
    
    def plot_game_lengths(self, average_over):
        running_average_length = [np.mean(self.game_lengths[i:i + average_over]) for i in
                                  range(len(self.game_lengths) - average_over)]
        plt.plot(running_average_length)
        plt.savefig(os.path.join("models", "policy-gradient-" + str(self.epoch) + ".png"))
        
    def play_game(self, training):
        steps = 0
        board_position_log = list()
        action_log = list()
        reward_log = list()
        probabilities = list()
        state = self.env.reset()
        terminal = False
        while not terminal:
            current_board = state.reshape(1, -1)
            board_position_log.append(np.copy(current_board))
            
            probability = self.model.get_probability(self.sess, current_board)
            
            # Compute action to take
            action = self.__choose_action(current_board, probability, training=training)
            
            # Take action
            new_state, reward, terminal, info = self.env.step(self.__toAction(action))
            
            action_log.append(action)
            reward_log.append(reward)
            probabilities.append(probability)
            state = new_state
            steps += 1
            
        board_position_log.append(np.copy(state.reshape(1, -1)))
        
        # Train if necessary
        if training:
            for reward, state, action in zip(reward_log, board_position_log, action_log):
                self.model.train(self.sess, state=state, action=action, reward=reward)
            self.epoch += 1
        
        return steps, board_position_log, probabilities
    
    def train(self, load=False):
        if not load:
            self.sess.run(model.init)
        with tqdm(range(self.nr_epochs), unit="games") as t:
            for epoch in t:
                
                # Train one game
                length, _, _ = self.play_game(training=True)
                self.game_lengths.append(length)
                
                if epoch > 0 and epoch % self.save_frequency == 0:
                    self.model.save(sess=self.sess, epoch=self.epoch)
                    
                if epoch > 500 and epoch % self.plot_frequency == 0:
                    self.plot_game_lengths(500)
                    
                t.set_postfix(train_game_length=np.mean(self.game_lengths[-100:]))
           
    def load_latest(self):
        paths = [x.split(".")[0] for x in os.listdir("models") if ("policy_gradient" in x and x[-1] == "x")]
        paths.sort(key=lambda x: int(x.split("-")[-1]))
        self.model.load(self.sess, os.path.join("models", paths[-1]))
        self.epoch = int(paths[-1].split("-")[-1])
        
    def test(self, latest=True, modelname="", trials=1000):
        if latest:
            self.load_latest()
        else:
            self.model.load(self.sess, os.path.join("models", modelname))
        game_lengths = list()
        for _ in trange(trials):
            length, _, _ = self.play_game(training=False)
            game_lengths.append(length)
        game_lengths.sort()

        median = game_lengths[trials // 2]
        upper = game_lengths[(trials * 3) // 4]
        lower = game_lengths[trials // 4]
        
        print("Min   \t{}".format(np.min(game_lengths)))
        print("25%   \t{}".format(lower))
        print("Mean  \t{}".format(np.mean(game_lengths)))
        print("Median\t{}".format(median))
        print("75%   \t{}".format(upper))
        print("Max   \t{}".format(np.max(game_lengths)))
            
    def test_all_timesteps(self, trials):
        labels = list()
        
        # Get paths
        paths = [x.split(".")[0] for x in os.listdir("models") if ("policy_gradient" in x and x[-1] == "x")]
        paths.sort(key=lambda x: int(x.split("-")[-1]))

        all_data = None
        
        with tqdm(paths) as t:
            for path in t:
                self.model.load(self.sess, os.path.join("models", path))
                game_lengths = list()
                for _ in range(trials):
                    length, _, _ = self.play_game(training=False)
                    game_lengths.append(length)
                
                if all_data is None:
                    all_data = np.reshape(np.array(game_lengths), [1, -1])
                else:
                    all_data = np.append(np.reshape(np.array(game_lengths), [1, -1]), all_data, axis=0)

                labels.append(int(path.split("-")[1]))
                
        np.save("data/labels", np.array(labels))
        np.save("data/all_data", all_data)
        
    def animate(self, latest=True, modelname="", save=False, with_probs=False):
        if latest:
            self.load_latest()
        else:
            self.model.load(self.sess, os.path.join("models", modelname))
            
        frames, history, probabilities = self.play_game(training=False)
        print("Rendering {} frames...".format(frames))

        M = np.reshape(history[0], [10, 10])
        P = np.reshape(probabilities[0], [10, 10])

        def render_frame(i):
            M = np.reshape(history[i], [10, 10])
            if i < len(probabilities):
                P = np.reshape(probabilities[i], [10, 10])
            else:
                P = np.zeros((10, 10))
            # Render grid
            game_matrix.set_array(M)
            probs_matrix.set_array(P)

            ax_game.get_xaxis().set_ticks([])
            ax_game.get_yaxis().set_ticks([])
            ax_probs.get_xaxis().set_ticks([])
            ax_probs.get_yaxis().set_ticks([])

        fig, (ax_game, ax_probs) = plt.subplots(1, 2, figsize=(14, 7))
        
        ax_game.set_title("Gameplay")
        ax_probs.set_title("Action Probability")
        ax_game.get_xaxis().set_ticks([])
        ax_game.get_yaxis().set_ticks([])
        ax_probs.get_xaxis().set_ticks([])
        ax_probs.get_yaxis().set_ticks([])

        cmaplist = [(0.0, 0.0, 0.0, 1.0),
                    (42 / 255, 77 / 255, 110 / 255, 1.0),
                    (170 / 255, 124 / 255, 57 / 255, 1.0),
                    (170 / 255, 95 / 255, 57 / 255, 1.0)]
        cmap = LinearSegmentedColormap.from_list("battleships", cmaplist, N=4)
        
        game_matrix = ax_game.matshow(M, vmin=0, vmax=1, cmap=cmap)
        probs_matrix = ax_probs.matshow(P, vmin=0, vmax=1, cmap="afmhot")

        divider = make_axes_locatable(ax_game)
        cax1 = divider.append_axes("left", size="5%", pad=0.05)
        fig.colorbar(game_matrix, cax=cax1, ticks=[0.125, 0.375, 0.625, 0.875])
        cax1.set_yticklabels(["Unexplored", "Miss", "Ship Hit", "Ship Sunk"])
        cax1.yaxis.tick_left()

        divider = make_axes_locatable(ax_probs)
        cax2 = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(probs_matrix, cax=cax2)
        
        anim = matplotlib.animation.FuncAnimation(
            fig, render_frame, frames=frames+1, interval=100, repeat=True
        )
        
        if save:
            anim.save("animations/" + "battleships_" + str(frames) + ".gif", dpi=80, fps=2, writer="imagemagick")
        else:
            matplotlib.use("TkAgg")
            plt.show()
            
        plt.close()
  

def plot_with_confidence_intervals(middle, upper, lower, labels, name, plot_name, nr_trials):
    fig, ax = plt.subplots()
    ax.fill_between(labels, upper, lower, color='g', alpha=0.5)
    ax.plot(labels, middle, 'g')
    ax.set_title(plot_name + " Number of Rounds per Game over {} Trails".format(nr_trials))
    ax.set_xlabel("Number of games trained")
    ax.set_ylabel("Number of rounds needed")
    ax.ticklabel_format(axis="both", style="sci")
    ax.tick_params(axis='x', rotation=30)
    ax.xaxis.labelpad = 15
    plt.tight_layout()
    fig.savefig("plots/" + name + ".svg")
    plt.show()
    plt.close()


def plot_all_timesteps():
    if os.path.isfile("data/all_data.npy") and os.path.isfile("data/labels.npy"):
        data = np.load("data/all_data.npy")
        labels = np.load("data/labels.npy")
        
        # Fix ordering
        data = np.flip(data, axis=0)
        
        means = np.mean(data, axis=1)
        medians = np.median(data, axis=1)

        best_medians = list(enumerate(medians))
        best_medians.sort(key=lambda x: x[1])
        for i in range(10):
            print("{} best value is {} at epoch {}".format(i, best_medians[i][1], (best_medians[i][0] + 1) * 1000))

        best_means = list(enumerate(means))
        best_means.sort(key=lambda x: x[1])
        for i in range(10):
            print("{} best value is {} at epoch {}".format(i, best_means[i][1], (best_means[i][0] + 1) * 1000))
        
        percentile5 = np.percentile(data, 5, axis=1)
        percentile25 = np.percentile(data, 25, axis=1)
        percentile75 = np.percentile(data, 75, axis=1)
        percentile95 = np.percentile(data, 95, axis=1)
        
        plot_with_confidence_intervals(medians, percentile75, percentile25, labels,
                                       "median_with_25_and_75_percentiles", "Median", data.shape[1])
        plot_with_confidence_intervals(means, percentile75, percentile25, labels,
                                       "mean_with_25_and_75_percentiles", "Mean", data.shape[1])
        plot_with_confidence_intervals(medians, percentile95, percentile5, labels,
                                       "median_with_5_and_95_percentiles", "Median", data.shape[1])
        plot_with_confidence_intervals(means, percentile95, percentile5, labels,
                                       "mean_with_5_and_95_percentiles", "Mean", data.shape[1])
        
    else:
        print("no data file found. Run test_all_timesteps first")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    default_model_name = "policy_gradient-3299008"
    parser.add_argument("--train", action="store_const", dest="train", const=True, default=False,
                        help="Train model instead of loading the best pre-trained model")
    parser.add_argument("--load", action="store_const", dest="load", const=True, default=False,
                        help="Load specific model")
    parser.add_argument("--load-dir", nargs="?", type=str, dest="load_dir", default=default_model_name,
                        help="Relative path of the model to load. Default: {}".format(default_model_name))
    parser.add_argument("--plot", action="store_const", dest="plot", const=True, default=False,
                        help="Plot all timesteps and show in a seperate window")
    parser.add_argument("--only-plot", action="store_const", dest="only_plot", const=True, default=False,
                        help="Load all_data.npy and plot all timesteps")
    parser.add_argument("--test", action="store_const", dest="test", const=True, default=False,
                        help="Test the model and show the performance numbers")
    args = parser.parse_args()
    
    if args.only_plot:
        plot_all_timesteps()
    else:
        with tf.Session() as session:
            environment = gym.make('Battleship-v0')
            model = Model(board_size=100, step_size=0.003)
            game = Game(sess=session, model=model, env=environment, nr_epochs=500000,
                        save_frequency=1000, plot_frequency=1000)
            
            if args.train:
                if args.load:
                    game.load_latest()
                    game.train(load=True)
                else:
                    game.train(load=False)
                
            if args.plot:
                game.test_all_timesteps(250)
                plot_all_timesteps()
                    
            if args.test:
                if args.train:
                    game.test(latest=True, trials=10000)
                else:
                    game.test(latest=False, modelname=default_model_name, trials=10000)
            
            if args.train:
                game.test(latest=True, trials=10000)
                game.animate(latest=True)
            else:
                game.animate(latest=False, modelname=default_model_name, save=True)
