from tensorflow_core import reshape
import tensorflow.keras.backend as K
from sls.agents import AbstractAgent
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Lambda
from tensorflow.keras.optimizers import Adam
from pysc2.lib.features import SCREEN_FEATURES
from baselines.deepq.replay_buffer import PrioritizedReplayBuffer


class Transition:
    def __init__(self, s_old, a_old, r_new, s_new, done):
        self.s_old = s_old
        self.a_old = a_old
        self.r_new = r_new
        self.s_new = s_new
        self.done = done


class CNNAgent(AbstractAgent):

    def __init__(self, train, screen_size, eps=0.0, eps_step_size=0.0, alpha=0.4, beta=0.6, beta_inc=0.000005,
                 gamma=0.9, eps_replay=0.000001):
        super(CNNAgent, self).__init__(screen_size)

        self.delta = 2
        self.eps = eps
        self.eps_step_size = eps_step_size
        self.s_old = None
        self.a_old = None
        self.alpha = alpha
        self.beta = beta
        self.beta_inc = beta_inc
        self.eps_replay = eps_replay
        self.gamma = gamma
        self.train = train
        self.unit_density_index = SCREEN_FEATURES.unit_density.index

        if train:
            # create new neuronal network
            self.model = Sequential()
            self.model.add(Conv2D(filters=16, kernel_size=5, strides=1, activation='relu', kernel_initializer='he_normal', input_shape=(16, 16, 1)))
            self.model.add(Conv2D(filters=32, kernel_size=3, strides=1, activation='relu', kernel_initializer='he_normal'))
            self.model.add(Flatten())
            self.model.add(Dense(units=64, activation='relu'))
            self.model.add(Dense(units=9, activation='linear'))
            self.model.add(Lambda(self.lmd))
            self.model.compile(loss='mse', optimizer=Adam(learning_rate=0.00025))

            self.target_model = Sequential()
            self.target_model.add(Conv2D(filters=16, kernel_size=5, strides=1, activation='relu', kernel_initializer='he_normal', input_shape=(16, 16, 1)))
            self.target_model.add(Conv2D(filters=32, kernel_size=3, strides=1, activation='relu', kernel_initializer='he_normal'))
            self.target_model.add(Flatten())
            self.target_model.add(Dense(units=64, activation='relu'))
            self.target_model.add(Dense(units=9, activation='linear'))
            self.target_model.add(Lambda(self.lmd))
            self.target_model.compile(loss='mse', optimizer=Adam(learning_rate=0.00025))
            self.target_model.set_weights(self.model.get_weights())

            self.prioritized_experience_replay = PrioritizedReplayBuffer(size=100000, alpha=0.4)

    def lmd(self, prm):
        value = prm[:, 0]
        value = reshape(value, [-1, 1])
        advantages = prm[:, 1:]
        mean = K.mean(advantages, axis=1)
        mean = reshape(mean, [-1, 1])
        result = value + (advantages - mean)
        return result

    def step(self, obs):
        if self._MOVE_SCREEN.id in obs.observation.available_actions:
            marine = self._get_marine(obs)
            if marine is None:
                return self._NO_OP

            # 1. get state
            unit_density = obs.observation["feature_screen"][self.unit_density_index]
            s_new = unit_density[:, :, np.newaxis]

            if self.train:
                # 2. select action a according to policy
                a_new = self.apply_policy(s_new)

                # 3. get reward for s_new
                r_new = obs.reward
                # 4.  store transition (if usable)
                if self.s_old is not None and self.a_old is not None:
                    self.prioritized_experience_replay.add(obs_t=self.s_old, action=self.a_old, reward=r_new,
                                                           obs_tp1=s_new, done=(obs.last() or r_new == 1))

                # 5. train model
                if len(self.prioritized_experience_replay) >= 6000:
                    # 5.1. sample random mini batch of transitions
                    (obs_batch, act_batch, rew_batch, next_obs_batch, done_mask, weights, idxes) = \
                        self.prioritized_experience_replay.sample(batch_size=32, beta=self.beta)
                    mini_batch = [Transition(s_old=obs_batch[i], a_old=act_batch[i], r_new=rew_batch[i],
                                             s_new=next_obs_batch[i], done=done_mask[i]) for i in range(len(obs_batch))]

                    # 5.2. execute gradient descent step
                    self.gradient_descent(mini_batch, idxes)

                # 6. safe state and action for next iteration and update eps eventually
                if obs.reward == 1 or obs.last():
                    self.s_old = None
                else:
                    self.s_old = s_new
                self.a_old = a_new
                if obs.last() and self.eps >= 0.05 + self.eps_step_size:
                    self.eps -= self.eps_step_size
            else:
                # 2. select action
                a_new = self.get_best_action_key(self.model, s_new)

            # 7. | 3. take action
            marine_coords = self._get_unit_pos(marine)
            return self._dir_to_sc2_action(a_new, marine_coords)
        else:
            return self._SELECT_ARMY

    def save_model(self, filename):
        self.model.save(filename + "/CNNetwork.h5")

    def load_model(self, filename):
        self.model = load_model(filename + "/CNNetwork.h5", custom_objects={'lmd': self.lmd})

    def apply_policy(self, s_new):
        # 3. select action a according to policy
        if np.random.rand() <= self.eps:
            best_action = np.random.choice(list(self._DIRECTIONS.keys()))
        else:
            # predict utility estimate for every action and take the best one
            best_action = self.get_best_action_key(self.model, s_new)

        return best_action

    def get_best_action_key(self, model, state):
        utilities = model.predict(np.asarray([state]))[0]
        u_max = np.amax(utilities)
        indices = np.where(utilities == u_max)[0]
        idx = indices[np.random.randint(0, len(indices))]
        return list(self._DIRECTIONS.keys())[idx]

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def gradient_descent(self, batch, idxes):
        # create training data
        x_train = np.asarray([b.s_old for b in batch])
        y_train = []
        y_model_old = self.model.predict_on_batch(x_train)  # predict for s_old
        y_model_new = self.model.predict_on_batch(np.asarray([b.s_new for b in batch]))  # predict for s_new
        y_target_new = self.target_model.predict_on_batch(np.asarray([b.s_new for b in batch]))  # predict for s_new

        for trans, y_model_old, y_model_new, y_target_new in zip(batch, y_model_old, y_model_new, y_target_new):
            # get index of used action in s_old
            idx = list(self._DIRECTIONS.keys()).index(trans.a_old)

            # set value of target output for action used in s_old according to pseudocode
            if trans.done:
                new_value = trans.r_new
            else:
                # get prediction of model in order to create error = 0 for all actions not taken in s_old
                idx_max = np.argmax(y_model_new)
                new_value = trans.r_new + self.gamma * y_target_new[idx_max]
            # collect predictions as training target data
            y_t = np.asarray([value for value in y_model_old])
            y_t[idx] = new_value
            y_train.append(y_t)

        # calculate and update new priorities
        errors = np.sum(np.abs(y_model_old - np.asarray(y_train)), axis=1)
        new_priorities = errors + self.eps_replay
        self.prioritized_experience_replay.update_priorities(idxes, new_priorities)

        # increase beta linear to 1
        self.beta = 1 if (self.beta + self.beta_inc) >= 1 else self.beta + self.beta_inc

        # apply gradient descent to model
        self.model.fit(np.array(x_train), np.array(y_train), verbose=False)
