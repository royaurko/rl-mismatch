import tensorflow as tf
import numpy as np


class TrainingParams:
    def __init__(self):
        self.initial_random_chance = 1.
        self.max_episode = 1000
        self.max_step = 199
        self.sess = None
        self.show_display = True
        self.show_freq = 10
        self.random_decay = 0.9925
        self.p_env = 0
        self.p_est = 0
        self.line_search = False


class GymRunner:
    def __init__(self, env, actor, p_env, p_est, gamma):
        self.actor = actor
        self.env = env
        self.sess = None
        self.random_chance = None
        self.tick = None
        self.p_env = p_env
        self.p_est = p_est
        self.gamma = gamma

    def train(self, training_params):
        if not training_params.sess:
            training_params.sess = tf.Session()

        self.sess = training_params.sess
        self.actor.build(self.sess)

        # todo move random chance to actors
        self.random_chance = training_params.initial_random_chance
        self.tick = 0
        history = []
        ave_cumu_r = None
        for i_episode in range(training_params.max_episode):
            obs = self.env.reset()
            score = 0
            for t in range(training_params.max_step):
                self.tick += 1
                if training_params.show_display and i_episode % training_params.show_freq == 0:
                    self.env.render()

                # start off with mostly random actions
                # slowly take away the random actions
                if np.random.random() < self.random_chance:
                    action = self.env.action_space.sample()
                else: 
                    action = self.actor.get_action(obs)

                threshold = 1 - self.p_env
                rand = np.random.uniform(0, 1)
                new_obs, reward, done, info = self.env.step(action)
                if rand > threshold:
                    # Random state
                    shape = new_obs.shape[0]
                    new_obs = np.random.randn(shape)
                    # print "random_obs = {}".format(new_obs)

                # print("New_obs = {}".format(new_obs))
                score = reward + self.gamma * score
                self.actor.step([obs, action, reward, new_obs, done])

                if done:
                    kappa = 0.01
                    if ave_cumu_r is None:
                        ave_cumu_r = score
                    else:
                        ave_cumu_r = kappa * score + (1 - kappa) * ave_cumu_r
                    break
                # update environment
                obs = new_obs
            history.append([i_episode, score, ave_cumu_r])
            # how did we do?
            print "Episode ", i_episode, "\tScore ", score, "\tRandom ", self.random_chance
            # slowly take away the random exploration
            self.random_chance *= training_params.random_decay
        self.actor.done()
        return np.array(history, dtype=np.float64)
