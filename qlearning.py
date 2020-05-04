import numpy as np
import matplotlib as plt
import time
from IPython.display import clear_output


class QLearning:

    # Enviroment Parameters
    env = None

    # Model Parameters
    q_table = None
    lr = 0.0
    dr = 0.0
    explr_decay = 0.001
    explr_rate = 1.0
    min_explr_rate = 0.001
    max_explr_rate = 1.0

    # Learning Parameters
    num_episodes = 10000
    max_steps = 100

    # Training Parameters
    episodes_rewards = None

    def __init__(self, env, lr=0.01, dr=0.99, explr_decay=0.001, min_explr_rate=0.001):
        self.env = env
        self.lr = lr
        self.dr = dr
        self.explr_decay = explr_decay
        self.min_explr_rate = min_explr_rate

        action_space_size = env.action_space.n
        state_space_size = env.observation_space.n
        self.q_table = np.zeros((state_space_size, action_space_size))

        self.episodes_rewards = []

    def learning_settings(self, num_episodes, max_steps):
        self.num_episodes = num_episodes
        self.max_steps = max_steps

    def train(self):
        for episode in range(self.num_episodes):
            state = self.env.reset()
            done = False
            episode_reward = 0

            for step in range(self.max_steps):
                action = self.__egreedy_policy(state)
                new_state, reward, done, info = self.env.step(action)
                self.__update_qtable(state, action, new_state, reward)

                state = new_state
                episode_reward += reward

                if done:
                    break

            self.__exp_decay(episode)
            self.episodes_rewards.append(episode_reward)

    def __update_qtable(self, state, action, new_state, reward):
        self.q_table[state, action] = self.q_table[state, action] * (1 - self.lr) + \
                                 self.lr * (reward + self.dr * np.max(self.q_table[new_state, :]))

    def __egreedy_policy(self, state):
        explr_rate_threshold = np.random.uniform(0, 1)
        if explr_rate_threshold > self.explr_rate:
            action = np.argmax(self.q_table[state, :])
        else:
            action = self.env.action_space.sample()
        return action

    def __exp_decay(self, episode):
        self.explr_rate = self.min_explr_rate + \
                          (self.max_explr_rate - self.min_explr_rate) * np.exp(-self.explr_decay*episode)

    def run(self, render=False):
        state = self.env.reset()
        done = False

        for step in range(self.max_steps):
            if render:
                clear_output(wait=True)
                self.env.render()
                time.sleep(0.3)

            action = np.argmax(self.q_table[state, :])
            new_state, reward, done, info = self.env.step(action)

            if done:
                if reward == 1:
                    #print('Goal Reached!')
                    return 1
                else:
                    #print('Fell through a hole!')
                    return 0

            state = new_state

        self.env.close()
