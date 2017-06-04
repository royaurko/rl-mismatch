import gym
import tensorflow as tf
import neural_network
import numpy as np
import gymrunner
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")


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


def main(env, num_episode, p_env, p_est, gamma=0.99):
    layer_param_list = [neural_network.RELULayerParams(50, name="hl1")]
    value_param_list = [neural_network.RELULayerParams(50, name="value1")]
    advantage_param_list = [neural_network.RELULayerParams(50, name="adv1")]
    duel_layer_params = neural_network.DuelLayersParams()
    duel_layer_params.value_layers = value_param_list
    duel_layer_params.advantage_layers = advantage_param_list

    layer_param_list.append(duel_layer_params)

    params = neural_network.DeepQParams()
    params.env = gym.make(env)
    params.layer_param_list = layer_param_list
    params.summary_location = "balancing_summary"
    params.train_freq = 32
    params.batch_size = 500
    params.update_param_freq = 128
    params.learning_rate = 1e-1
    params.memory_size = 100000
    params.p_est = p_est
    params.discount_rate = gamma

    with tf.device("/cpu:0"):
        nn = neural_network.DeepQ(params)
        training_params = gymrunner.TrainingParams()
        training_params.max_episode = num_episode
        training_params.max_step = 199
        training_params.random_decay = 0.995

        runner = gymrunner.GymRunner(params.env, nn, p_env, p_est, gamma)
        history = runner.train(training_params)
        print "history = {}".format(history)
    tf.reset_default_graph()
    return history


def format_e(n):
    a = '%e' % n
    return a.split('e')[0].rstrip('0').rstrip('.') + 'e' + a.split('e')[1]


def cross_validate(env, p_env, num_episode, folds=5):
    p_est = 1e-9
    nu = 10
    rewards = []
    p_est_list = []
    num_runs = 1
    for _ in range(folds):
        r = 0
        for _ in range(num_runs):
            robust_learning_history = main(env, num_episode, p_env=p_env, p_est=p_est)
            r += robust_learning_history[:, 2][-1]
        r /= num_runs
        p_est_list.append(p_est)
        rewards.append(r)
        p_est *= nu
    nominal_history = main(env, num_episode, p_env=p_env, p_est=0)
    nominal_rewards = [nominal_history[:, 2][-1]] * len(rewards)
    # Plot stuff
    cv_file_name = 'cross_validation_{0}_{1}.png'.format(env, format_e(p_env))
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


if __name__ == "__main__":
    p_environment = 3e-1
    environment = "CartPole-v0"
    num_cv_episode = 500
    p_estimate = cross_validate(env=environment, p_env=p_environment, num_episode=num_cv_episode, folds=10)
    num_episodes = 5000
    nominal_transient_history = main(env=environment, num_episode=num_episodes, p_env=p_environment,
                                     p_est=0)
    robust_transient_history = main(env=environment, num_episode=num_episodes, p_env=p_environment,
                                    p_est=p_estimate)
    cum_file_name = 'cum_rewards_{0}_{1}.png'.format(environment, format_e(p_environment))
    plot_cumulative_reward(robust_transient_history, nominal_transient_history, cum_file_name)

    # Plot tail distribution
    tail_file_name = 'cdf_comparison_{0}_{1}.png'.format(environment, format_e(p_environment))
    plot_tail_distribution(robust_transient_history, nominal_transient_history, tail_file_name)

