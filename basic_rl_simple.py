import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("darkgrid")
import argparse
parser = argparse.ArgumentParser(description='Use SARSA/Q-learning algorithm with epsilon-greedy/softmax polciy.')
parser.add_argument('-a', '--algorithm', default='q_learning', choices=['sarsa', 'q_learning'],
                    help="Type of learning algorithm. (Default: sarsa)")
parser.add_argument('-p', '--policy', default='epsilon_greedy', choices=['epsilon_greedy', 'softmax'],
                    help="Type of policy. (Default: epsilon_greedy)")
parser.add_argument('-e', '--environment', default='Roulette-v0',
                    help="Name of the environment provided in the OpenAI Gym. (Default: Roulette-v0)")
parser.add_argument('-n', '--nepisode', default='5000', type=int,
                    help="Number of episode. (Default: 20000)")
parser.add_argument('-al', '--alpha', default='0.1', type=float,
                    help="Learning rate. (Default: 0.1)")
parser.add_argument('-be', '--beta', default='0.0', type=float,
                    help="Initial value of an inverse temperature. (Default: 0.0)")
parser.add_argument('-bi', '--betainc', default='0.01', type=float,
                    help="Linear increase rate of an inverse temperature. (Default: 0.01)")
parser.add_argument('-ga', '--gamma', default='0.99', type=float,
                    help="Discount rate. (Default: 0.99)")
parser.add_argument('-ep', '--epsilon', default='0.8', type=float,
                    help="Fraction of random exploration in the epsilon greedy. (Default: 0.8)")
parser.add_argument('-ed', '--epsilondecay', default='0.995', type=float,
                    help="Decay rate of epsilon in the epsilon greedy. (Default: 0.995)")
parser.add_argument('-ms', '--maxstep', default='1000', type=int,
                    help="Maximum step allowed in a episode. (Default: 200)")
parser.add_argument('-ka', '--kappa', default='0.1', type=float,
                    help="Weight of the most recent cumulative reward for computing its running average. (Default: 0.01)")
parser.add_argument('-qm', '--qmean', default='0.0', type=float,
                    help="Mean of the Gaussian used for initializing Q table. (Default: 0.0)")
parser.add_argument('-qs', '--qstd', default='1.0', type=float,
                    help="Standard deviation of the Gaussian used for initializing Q table. (Default: 1.0)")
parser.add_argument('-ro', '--noisy', action='store_true', default=False,
                    help='Noisy/robust updates used for the algorithm')
parser.add_argument('-pr', '--prob', default=0.2, type=float,
                    help="Probability of perturbation")
args = parser.parse_args()

import gym
import numpy as np
import os

import matplotlib.pyplot as plt

def softmax(q_value, beta=1.0):
    assert beta >= 0.0
    q_tilde = q_value - np.max(q_value)
    factors = np.exp(beta * q_tilde)
    return factors / np.sum(factors)


def select_a_with_softmax(curr_s, q_value, beta=1.0):
    prob_a = softmax(q_value[curr_s, :], beta=beta)
    cumsum_a = np.cumsum(prob_a)
    return np.where(np.random.rand() < cumsum_a)[0][0]


def select_a_with_epsilon_greedy(curr_s, q_value, epsilon=0.1):
    a = np.argmax(q_value[curr_s, :])
    if np.random.rand() < epsilon:
        a = np.random.randint(q_value.shape[1])
    return a


def select_a_greedy(curr_s, q_value):
    return np.argmax(q_value[curr_s, :])


def get_value(q_value):
    # Get shape
    n_s, n_a = q_value.shape
    value = np.amax(q_value, axis=1)
    return value


def get_sigma(value, p):
    # Ball of radius r
    sigma = - p * np.linalg.norm(value)
    # sigma = - p
    return sigma


def q_learning(robust, p_env, p_est, epsilon=0, qtable=None, update=True, n_episode=10000):
    env_type = args.environment
    algorithm_type = args.algorithm
    policy_type = args.policy

    # Random seed
    np.random.RandomState(42)

    # Selection of the problem
    env = gym.envs.make(env_type)

    # Constraints imposed by the environment
    n_a = env.action_space.n
    n_s = env.observation_space.n

    print "Number of states = {}".format(n_s)
    # Meta parameters for the RL agent
    alpha = args.alpha
    beta = args.beta
    beta_inc = args.betainc
    gamma = args.gamma
    epsilon_decay = args.epsilondecay
    q_mean = args.qmean
    q_std = args.qstd

    # Experimental setup
    max_step = args.maxstep

    # Running average of the cumulative reward, which is used for controlling an exploration rate
    # (This idea of controlling exploration rate by the terminal reward is suggested by JKCooper2)
    # See https://gym.openai.com/evaluations/eval_xSOlwrBsQDqUW7y6lJOevQ
    kappa = args.kappa
    ave_cumu_r = None

    # Initialization of a Q-value table
    q_value = np.zeros([n_s, n_a])

    # If qtable is not none set q_value to it
    if qtable is not None:
        q_value = qtable

    # Initialization of a list for storing simulation history
    history = []

    print "algorithm_type: {}".format(algorithm_type)
    print "policy_type: {}".format(policy_type)

    env.reset()

    np.set_printoptions(precision=3, suppress=True)

    result_dir = 'results-{0}-{1}-{2}'.format(env_type, algorithm_type, policy_type)

    # Start monitoring the simulation for OpenAI Gym
    env = gym.wrappers.Monitor(env, result_dir, force=True)
    threshold = 1 - p_env

    for i_episode in xrange(n_episode):
        # Reset a cumulative reward for this episode
        cumu_r = 0

        # Start a new episode and sample the initial state
        curr_s = env.reset()

        # Print q_table
        # print "qtable = {}".format(q_value)

        # Select the first action in this episode
        if policy_type == 'softmax':
            curr_a = select_a_with_softmax(curr_s, q_value, beta=beta)
        elif policy_type == 'epsilon_greedy':
            curr_a = select_a_with_epsilon_greedy(curr_s, q_value, epsilon=epsilon)
        else:
            raise ValueError("Invalid policy_type: {}".format(policy_type))

        for i_step in xrange(max_step):

            # Get a result of your action from the environment
            next_s, r, done, info = env.step(curr_a)

            # With some probability choose a random state
            rand = np.random.uniform(0, 1)
            if rand > threshold:
                next_s = np.random.randint(0, n_s)

            # Update a cummulative reward
            cumu_r = r + gamma * cumu_r

            # Select an action
            if policy_type == 'softmax':
                next_a = select_a_with_softmax(next_s, q_value, beta=beta)
            elif policy_type == 'epsilon_greedy':
                next_a = select_a_with_epsilon_greedy(next_s, q_value, epsilon=epsilon)
            else:
                raise ValueError("Invalid policy_type: {}".format(policy_type))

            # Calculation of TD error
            if update:
                # Only update table if update set to true
                noise = 0
                if robust:
                    value = get_value(q_value)
                    noise = get_sigma(value, p_est)
                if algorithm_type == 'sarsa':
                    delta = r + gamma * noise + gamma * q_value[next_s, next_a] - q_value[curr_s, curr_a]
                elif algorithm_type == 'q_learning':
                    delta = r + gamma * noise + gamma * np.max(q_value[next_s, :]) - q_value[curr_s, curr_a]
                else:
                    raise ValueError("Invalid algorithm_type: {}".format(algorithm_type))

                # Update a Q value table
                q_value[curr_s, curr_a] += alpha * delta
            curr_s = next_s
            curr_a = next_a
            if done:

                # Running average of the terminal reward, which is used for controlling an exploration rate
                # (This idea of controlling exploration rate by the terminal reward is suggested by JKCooper2)
                # See https://gym.openai.com/evaluations/eval_xSOlwrBsQDqUW7y6lJOevQ
                kappa = 0.01
                if ave_cumu_r == None:
                    ave_cumu_r = cumu_r
                else:
                    ave_cumu_r = kappa * cumu_r + (1 - kappa) * ave_cumu_r

                if cumu_r > ave_cumu_r:
                    # Bias the current policy toward exploitation

                    if policy_type == 'epsilon_greedy':
                        # epsilon is decayed expolentially
                        epsilon = epsilon * epsilon_decay
                    elif policy_type == 'softmax':
                        # beta is increased linearly
                        beta = beta + beta_inc

                if policy_type == 'softmax':
                    print "Episode: {0}\t Steps: {1:>4}\tCumuR: {2:>5.2f}\tTermR: {3}\tAveCumuR: {4:.3f}\tBeta: {5:.3f}".format(
                        i_episode, i_step, cumu_r, r, ave_cumu_r, beta)
                    history.append([i_episode, i_step, cumu_r, r, ave_cumu_r, beta])
                elif policy_type == 'epsilon_greedy':
                    print "Episode: {0}\t Steps: {1:>4}\tCumuR: {2:>5.2f}\tTermR: {3}\tAveCumuR: {4:.3f}\tEpsilon: {5:.3f}".format(
                        i_episode, i_step, cumu_r, r, ave_cumu_r, epsilon)
                    history.append([i_episode, i_step, cumu_r, r, ave_cumu_r, epsilon])
                else:
                    raise ValueError("Invalid policy_type: {}".format(policy_type))

                break

    # Stop monitoring the simulation for OpenAI Gym
    # env.monitor.close()
    history = np.array(history)

    print "Q_value = {0}".format(q_value)

    if policy_type == 'softmax':
        print "Action selection probability:"
        print np.array([softmax(q, beta=beta) for q in q_value])
    elif policy_type == 'epsilon_greedy':
        print "Greedy action"
        greedy_action = np.zeros([n_s, n_a])
        greedy_action[np.arange(n_s), np.argmax(q_value, axis=1)] = 1
        print greedy_action

    return history, q_value


def plot_cumulative_reward(robust_history, nominal_history, file_name):
    fig = plt.figure()
    plt.xlabel('Episode')
    plt.ylabel('Average cumulative reward')
    plt.plot(nominal_history[:, 0], nominal_history[:, 4], ls="-", label='Nominal')
    plt.plot(robust_history[:, 0], robust_history[:, 4], ls="-", label='Robust')
    plt.legend(loc='best', fontsize=14)
    fig.savefig(file_name)


def plot_tail_distribution(robust_history, nominal_history, file_name):
    fig = plt.figure()
    plt.xlabel('a')
    plt.ylabel('Pr[r > a]')

    num_bins = 5000
    values, base = np.histogram(nominal_history[:, 4], bins=num_bins)
    cumulative = np.cumsum(values[::-1])[::-1]
    cumulative = np.array(cumulative, dtype=np.float64)
    cumulative /= np.max(cumulative)

    robust_values, robust_base = np.histogram(robust_history[:, 4], bins=num_bins)
    robust_cumulative = np.cumsum(robust_values[::-1])[::-1]
    robust_cumulative = np.array(robust_cumulative, dtype=np.float64)
    robust_cumulative /= np.max(robust_cumulative)
    plt.plot(base[:-1], cumulative, ls='-', label='Nominal')
    plt.plot(robust_base[:-1], robust_cumulative, ls='-', label='Robust')
    plt.legend(loc='best', fontsize=14)
    fig.savefig(file_name)


def cross_validate(folds=5):
    p_env = args.prob
    epsilon = args.epsilon
    n_episode = args.nepisode
    p_est = 1e-9
    nu = 10
    rewards = []
    p_est_list = []
    num_runs = 1
    for _ in range(folds):
        r = 0
        nominal_r = 0
        for _ in range(num_runs):
            robust_learning_history, _ = q_learning(robust=True, p_env=p_env, p_est=p_est,
                                                    epsilon=epsilon, qtable=None,
                                                    update=True, n_episode=n_episode)
            r += robust_learning_history[:, 4][-1]
        r /= num_runs
        nominal_r /= num_runs
        p_est_list.append(p_est)
        rewards.append(r)
        p_est *= nu
    nominal_history, _ = q_learning(robust=True, p_env=p_env, p_est=0,
                                    epsilon=epsilon, qtable=None,
                                    update=True, n_episode=n_episode)
    nominal_rewards = [nominal_history[:, 4][-1]] * len(rewards)
    # Plot stuff
    cv_file_name = 'cross_validation_{0}_{1}.png'.format(args.environment, format_e(p_env))
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


def compare_nominal(p_est):
    p_env = args.prob
    epsilon = args.epsilon
    n_episode = args.nepisode
    robust_learning_history, robust_qtable = q_learning(robust=True, p_env=p_env, p_est=p_est,
                                                        epsilon=epsilon, qtable=None,
                                                        update=True, n_episode=n_episode)
    learning_history, nominal_qtable = q_learning(robust=False, p_env=p_env, p_est=p_est,
                                                  epsilon=epsilon, qtable=None,
                                                  update=True, n_episode=n_episode)

    history, _ = q_learning(robust=True, p_env=0, p_est=0, epsilon=0, qtable=nominal_qtable,
                            update=False, n_episode=n_episode)
    robust_history, _ = q_learning(robust=True, p_env=0, p_est=0, epsilon=0, qtable=robust_qtable,
                                   update=False, n_episode=n_episode)

    # Plot cumulative reward of learning phase
    learning_file_name = 'cum_rewards_learning_{0}_{1}.png'.format(args.environment, format_e(p_env))
    plot_cumulative_reward(robust_learning_history, learning_history, learning_file_name)

    cum_file_name = 'cum_rewards_{0}_{1}.png'.format(args.environment, format_e(p_env))
    plot_cumulative_reward(robust_history, history, cum_file_name)

    # Plot tail distribution
    tail_file_name = 'cdf_comparison_{0}_{1}.png'.format(args.environment, format_e(p_env))
    plot_tail_distribution(robust_history, history, tail_file_name)


def format_e(n):
    a = '%e' % n
    return a.split('e')[0].rstrip('0').rstrip('.') + 'e' + a.split('e')[1]


if __name__ == "__main__":
    # Cross validate
    p_est = cross_validate(folds=10)
    # First compare nominal
    compare_nominal(p_est=p_est)
    print("p_est = {}".format(p_est))
