import gym
from gym import wrappers
import pandas as pd
import numpy as np
import random
import cPickle as pickle
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")
# https://gym.openai.com/envs/CartPole-v0
# Carlos Aguayo - carlos.aguayo@gmail.com


class QLearner(object):
    def __init__(self,
                 num_states=100,
                 num_actions=4,
                 alpha=0.2,
                 gamma=0.9,
                 random_action_rate=0.5,
                 random_action_decay_rate=0.99):
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.random_action_rate = random_action_rate
        self.random_action_decay_rate = random_action_decay_rate
        self.state = 0
        self.action = 0
        self.qtable = np.zeros((num_states, num_actions))
        # self.qtable = np.random.uniform(low=-1, high=1, size=(num_states, num_actions))

    def set_initial_state(self, state):
        """
        @summary: Sets the initial state and returns an action
        @param state: The initial state
        @returns: The selected action
        """
        self.state = state
        self.action = self.qtable[state].argsort()[-1]
        return self.action

    def move(self, state_prime, reward, robust=False, p=0.1, update=True):
        """
        @summary: Moves to the given state with given reward and returns action
        @param state_prime: The new state
        @param reward: The reward
        @returns: The selected action
        """
        alpha = self.alpha
        gamma = self.gamma
        state = self.state
        action = self.action
        qtable = self.qtable

        choose_random_action = (1 - self.random_action_rate) <= np.random.uniform(0, 1)

        if choose_random_action:
            action_prime = random.randint(0, self.num_actions - 1)
        else:
            action_prime = self.qtable[state_prime].argsort()[-1]

        self.random_action_rate *= self.random_action_decay_rate
        if update:
            noise = 0
            if robust:
                value = np.amax(qtable, axis=1)
                assert value.shape[0] == self.num_states
                noise = get_sigma(value, p)
                # print("noise = {}, p = {}".format(noise, p))
                # print "value = {}".format(value)
                # print "norm of value = {}".format(np.linalg.norm(value))
                # print "noise = {}".format(noise)

            qtable[state, action] = (1 - alpha) * qtable[state, action] +\
                alpha * (reward + gamma * noise + gamma * qtable[state_prime, action_prime])

        self.state = state_prime
        self.action = action_prime

        return self.action


def get_sigma(value, p):
    sigma = - p * np.linalg.norm(value)
    # print("value = {}".format(value))
    # print("sigma = {}".format(sigma))
    return sigma


def get_list(num_features, num_bins):
    if num_features == 1:
        l = []
        for b in xrange(num_bins):
            k = (b,)
            l.append(k)
        return l
    # Else recurse
    l = get_list(num_features - 1, num_bins)
    new_l = []
    for b in xrange(num_bins):
        for k in l:
            new_k = (b,) + k
            new_l.append(new_k)
    return new_l


def get_state_dict(num_features, num_bins):
    num_states = num_bins ** num_features
    l = get_list(num_features, num_bins)
    d = {}
    for i, k in enumerate(l):
        d[k] = i
    return d


def build_state(features):
    return int("".join(map(lambda feature: str(int(feature)), features)))


def to_bin(value, bins):
    return np.digitize(x=[value], bins=bins)[0]


def get_random_state_tuple(cart_position_bins, pole_angle_bins,
                        cart_velocity_bins, angle_rate_bins):
    cart_position = np.random.choice(len(cart_position_bins))
    pole_angle = np.random.choice(len(pole_angle_bins))
    cart_velocity_bins = np.random.choice(len(cart_velocity_bins))
    angle_rate_bins = np.random.choice(len(angle_rate_bins))
    features = (cart_position, pole_angle, cart_velocity_bins,
                angle_rate_bins)
    return features


def q_learning(environment, robust, p_env, p_est, qtable=None, num_episode=10000, update=True, fname=None):
    env = gym.make(environment)
    directory = environment
    env = wrappers.Monitor(env, directory, force=True)

    goal_average_steps = 195
    max_number_of_steps = 2000
    number_of_iterations_to_average = 100
    num_bins = 20
    number_of_features = env.observation_space.shape[0]
    state_dict = get_state_dict(number_of_features, num_bins)
    last_time_steps = np.ndarray(0)
    cart_position_bins = pd.cut([-2.4, 2.4], bins=num_bins, retbins=True)[1][1:-1]
    pole_angle_bins = pd.cut([-2, 2], bins=num_bins, retbins=True)[1][1:-1]
    cart_velocity_bins = pd.cut([-1, 1], bins=num_bins, retbins=True)[1][1:-1]
    angle_rate_bins = pd.cut([-3.5, 3.5], bins=num_bins, retbins=True)[1][1:-1]

    learner = QLearner(num_states=num_bins ** number_of_features,
                       num_actions=env.action_space.n,
                       alpha=0.2,
                       gamma=1,
                       random_action_rate=0,
                       random_action_decay_rate=0.99)

    if qtable is not None:
        learner.qtable = qtable
    history = []
    ave_cumu_r = None
    for episode in xrange(num_episode):
        observation = env.reset()
        cart_position, pole_angle, cart_velocity, angle_rate_of_change = observation
        state = state_dict[(to_bin(cart_position, cart_position_bins),
                            to_bin(pole_angle, pole_angle_bins),
                            to_bin(cart_velocity, cart_velocity_bins),
                            to_bin(angle_rate_of_change, angle_rate_bins))]
        action = learner.set_initial_state(state)

        # Uncertainty parameters
        threshold = 1 - p_env
        cumu_r = 0

        for step in xrange(max_number_of_steps - 1):
            observation, reward, done, info = env.step(action)
            cart_position, pole_angle, cart_velocity, angle_rate_of_change = observation

            rand = np.random.uniform(0, 1)
            if rand > threshold:
                state_prime = get_random_state_tuple(cart_position_bins, pole_angle_bins,
                                                     cart_velocity_bins, angle_rate_bins)
                state_prime = state_dict[state_prime]
            else:
                state_prime = state_dict[(to_bin(cart_position, cart_position_bins),
                                          to_bin(pole_angle, pole_angle_bins),
                                          to_bin(cart_velocity, cart_velocity_bins),
                                          to_bin(angle_rate_of_change, angle_rate_bins))]

            if done:
                reward = -200

            cumu_r = reward + learner.gamma * cumu_r
            action = learner.move(state_prime, reward, robust, p_est, update)
            if done:
                kappa = 0.01
                if ave_cumu_r is None:
                    ave_cumu_r = cumu_r
                else:
                    ave_cumu_r = kappa * cumu_r + (1 - kappa) * ave_cumu_r
                last_time_steps = np.append(last_time_steps, [int(step + 1)])
                if len(last_time_steps) > number_of_iterations_to_average:
                    last_time_steps = np.delete(last_time_steps, 0)
                break

        history.append([episode, cumu_r, ave_cumu_r])
        if last_time_steps.mean() > goal_average_steps:
            print "Goal reached!"
            print "Episodes before solve: ", episode + 1
            print u"Best 100-episode performance {} {}".format(last_time_steps.max(),
                                                               last_time_steps.std())

    # env.monitor.close()
    if fname is not None:
        with open(fname, "wb") as f:
            pickle.dump(learner.qtable, f)
    return np.array(history), learner.qtable


def plot_cumulative_reward(robust_history, nominal_history, file_name):
    fig = plt.figure()
    plt.xlabel('Episode')
    plt.ylabel('Average cumulative reward')
    plt.plot(nominal_history[:, 0], nominal_history[:, 2], ls="-", label='Nominal')
    plt.plot(robust_history[:, 0], robust_history[:, 2], ls="-", label='Robust')
    plt.legend(loc='best', fontsize=14)
    fig.savefig(file_name)


def plot_tail_distribution(robust_history, nominal_history, file_name):
    fig = plt.figure()
    plt.xlabel('a')
    plt.ylabel('Pr[r > a]')

    num_bins = 5000
    values, base = np.histogram(nominal_history[:, 2], bins=num_bins)
    cumulative = np.cumsum(values[::-1])[::-1]
    cumulative = np.array(cumulative, dtype=np.float64)
    cumulative /= np.max(cumulative)

    robust_values, robust_base = np.histogram(robust_history[:, 2], bins=num_bins)
    robust_cumulative = np.cumsum(robust_values[::-1])[::-1]
    robust_cumulative = np.array(robust_cumulative, dtype=np.float64)
    robust_cumulative /= np.max(robust_cumulative)
    plt.plot(base[:-1], cumulative, ls='-', label='Nominal')
    plt.plot(robust_base[:-1], robust_cumulative, ls='-', label='Robust')
    plt.legend(loc='best', fontsize=14)
    fig.savefig(file_name)


def compare_nominal(environment, num_episode, p_env, p_est):
    random.seed(0)
    robust_fname = environment + "-robust-qtable.pkl"
    nominal_fname = environment + "-nominal-qtable.pkl"

    robust_learning_history, robust_qtable = q_learning(environment, robust=True, p_env=p_env,
                                                        p_est=p_est, num_episode=num_episode,
                                                        qtable=None, update=True, fname=robust_fname)
    learning_history, nominal_qtable = q_learning(environment, robust=False, p_env=p_env, p_est=p_est,
                                                  num_episode=num_episode, qtable=None,
                                                  update=True, fname=nominal_fname)

    history, _ = q_learning(environment, robust=True, p_env=0, p_est=p_est, num_episode=num_episode,
                            qtable=nominal_qtable, update=False, fname=None)
    robust_history, _ = q_learning(environment, robust=True, p_env=0, p_est=p_est, num_episode=num_episode,
                                   qtable=robust_qtable, update=False, fname=None)

    print "nominal_qtable = {}".format(nominal_qtable)
    print "robust_qtable = {}".format(robust_qtable)

    # Plot cumulative reward of learning phase
    learning_file_name = 'cum_rewards_learning_{0}_{1}.png'.format(environment, format_e(p_env))
    plot_cumulative_reward(robust_learning_history, learning_history, learning_file_name)

    cum_file_name = 'cum_rewards_{0}_{1}.png'.format(environment, format_e(p_env))
    plot_cumulative_reward(robust_history, history, cum_file_name)

    # Plot tail distribution
    tail_file_name = 'cdf_comparison_{0}_{1}.png'.format(environment, format_e(p_env))
    plot_tail_distribution(robust_history, history, tail_file_name)


def cross_validate(environment, p_env, num_episode, folds=5):
    p_est = 1e-9
    nu = 10
    rewards = []
    p_est_list = []
    num_runs = 1
    for _ in range(folds):
        r = 0
        for _ in range(num_runs):
            robust_learning_history, _ = q_learning(environment, robust=True, p_env=p_env, p_est=p_est,
                                                    qtable=None, update=True, num_episode=num_episode, fname=None)
            r += robust_learning_history[:, 2][-1]
        r /= num_runs
        p_est_list.append(p_est)
        rewards.append(r)
        p_est *= nu
    nominal_history, _ = q_learning(environment, robust=True, p_env=p_env, p_est=0, qtable=None,
                                    update=True, num_episode=num_episode, fname=None)
    nominal_rewards = [nominal_history[:, 2][-1]] * len(rewards)
    # Plot stuff
    cv_file_name = 'cross_validation_{0}_{1}.png'.format(environment, format_e(p_env))
    fig = plt.figure()
    p_est_list = np.array(p_est_list, dtype=np.float64)
    rewards = np.array(rewards, dtype=np.float64)
    p_est_list_log = np.log10(p_est_list)
    plt.xlabel('Estimated log probability')
    plt.ylabel('Average Reward')
    plt.plot(p_est_list_log, nominal_rewards, ls="-", label="Nominal")
    plt.plot(p_est_list_log, rewards, ls="-", label="Robust")
    plt.legend(loc='best', fontsize=14)
    fig.savefig(cv_file_name)
    print("rewards = {}".format(rewards))
    print("p_est = {}".format(p_est_list))

    # Return max p_est
    max_idx = np.argmax(rewards)
    return p_est_list[max_idx]


def format_e(n):
    a = '%e' % n
    return a.split('e')[0].rstrip('0').rstrip('.') + 'e' + a.split('e')[1]


if __name__ == "__main__":
    num_cv_episode = 1000
    num_episode = 8000
    environment = "CartPole-v1"
    p_env = 0.1
    p_est = cross_validate(environment, p_env, num_cv_episode, folds=10)
    compare_nominal(environment, num_episode, p_env, p_est)