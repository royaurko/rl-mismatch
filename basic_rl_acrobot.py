import gym
from gym import wrappers
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")


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

    def move(self, state_prime, reward, robust=False, p=0.1):
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
        noise = 0
        if robust:
            value = np.amax(qtable, axis=1)
            assert value.shape[0] == self.num_states
            noise = get_sigma(value, p)
        # print "noise = {}".format(noise)
        # noise = 0

        qtable[state, action] = (1 - alpha) * qtable[state, action] +\
            alpha * (reward + gamma * noise + gamma * qtable[state_prime, action_prime])

        self.state = state_prime
        self.action = action_prime

        return self.action


def get_sigma(value, p):
    return - p * np.linalg.norm(value)


def cart_pole_with_qlearning(environment, robust, p_env, p_est):
    env = gym.make('Acrobot-v1')
    experiment_filename = './cartpole-experiment-1'
    directory = 'cartpole'
    env = wrappers.Monitor(env, directory, force=True)
    # env.monitor.start(experiment_filename, force=True)

    goal_average_steps = 195
    max_number_of_steps = 600
    # max_number_of_steps = 200
    number_of_iterations_to_average = 100

    print "obs type = {}".format(type(env.observation_space.shape))
    print "obs shape = {}".format(env.observation_space.shape)
    number_of_features = env.observation_space.shape[0]
    last_time_steps = np.ndarray(0)

    cart_position_bins = pd.cut([-2.4, 2.4], bins=10, retbins=True)[1][1:-1]
    pole_angle_bins = pd.cut([-2, 2], bins=10, retbins=True)[1][1:-1]
    cart_velocity_bins = pd.cut([-1, 1], bins=10, retbins=True)[1][1:-1]
    angle_rate_bins = pd.cut([-3.5, 3.5], bins=10, retbins=True)[1][1:-1]
    x1_bins = pd.cut([-3.5, 3.5], bins=10, retbins=True)[1][1:-1]
    x2_bins = pd.cut([-3.5, 3.5], bins=10, retbins=True)[1][1:-1]

    def build_state(features):
        return int("".join(map(lambda feature: str(int(feature)), features)))

    def to_bin(value, bins):
        return np.digitize(x=[value], bins=bins)[0]

    def get_random_state(cart_position_bins, pole_angle_bins,
                         cart_velocity_bins, angle_rate_bins, x1_bins, x2_bins):
        cart_position = np.random.choice(len(cart_position_bins))
        pole_angle = np.random.choice(len(pole_angle_bins))
        cart_velocity_bins = np.random.choice(len(cart_velocity_bins))
        angle_rate_bins = np.random.choice(len(angle_rate_bins))
        x1 = np.random.choice(len(x1_bins))
        x2 = np.random.choice(len(x2_bins))
        features = [cart_position, pole_angle, cart_velocity_bins,
                    angle_rate_bins, x1, x2]
        return build_state(features)

    learner = QLearner(num_states=10 ** number_of_features,
                       num_actions=env.action_space.n,
                       alpha=0.2,
                       gamma=1,
                       random_action_rate=0.5,
                       random_action_decay_rate=0.99)

    history = []
    ave_cumu_r = None
    for episode in xrange(5000):
        observation = env.reset()
        cart_position, pole_angle, cart_velocity, angle_rate_of_change, x1, x2 = observation
        state = build_state([to_bin(cart_position, cart_position_bins),
                             to_bin(pole_angle, pole_angle_bins),
                             to_bin(cart_velocity, cart_velocity_bins),
                             to_bin(angle_rate_of_change, angle_rate_bins),
                             to_bin(x1, x1_bins),
                             to_bin(x2, x2_bins)])
        action = learner.set_initial_state(state)

        # Uncertainty parameters
        threshold = 1 - p
        cumu_r = 0

        for step in xrange(max_number_of_steps - 1):
            observation, reward, done, info = env.step(action)
            cart_position, pole_angle, cart_velocity, angle_rate_of_change, x1, x2 = observation

            rand = np.random.uniform(0, 1)
            if rand > threshold:
                state_prime = get_random_state(cart_position_bins, pole_angle_bins,
                                               cart_velocity_bins, angle_rate_bins,
                                               x1_bins, x2_bins)
            else:
                state_prime = build_state([to_bin(cart_position, cart_position_bins),
                                        to_bin(pole_angle, pole_angle_bins),
                                        to_bin(cart_velocity, cart_velocity_bins),
                                        to_bin(angle_rate_of_change, angle_rate_bins),
                                        to_bin(x1, x1_bins),
                                        to_bin(x2, x2_bins)])

            if done:
                reward = -200

            cumu_r = reward + learner.gamma * cumu_r
            action = learner.move(state_prime, reward, p, robust)
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
    return np.array(history)


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
    environment = "Acrobot-v0"
    p_env = 0.3
    p_est = cross_validate(environment, p_env, num_cv_episode, folds=10)
    compare_nominal(environment, num_episode, p_env, p_est)
