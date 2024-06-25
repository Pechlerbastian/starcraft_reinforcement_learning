from sls.agents import AbstractAgent
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


class Transition:
    def __init__(self, s_old, a_old, r_new, s_new, done):
        self.s_old = s_old
        self.a_old = a_old
        self.r_new = r_new
        self.s_new = s_new
        self.done = done


class DDQNAgent(AbstractAgent):

    def __init__(self, train, screen_size, eps=0.0, eps_step_size=0.0, alpha=0.1, gamma=0.9):
        super(DDQNAgent, self).__init__(screen_size)

        self.delta = 2
        self.eps = eps
        self.eps_step_size = eps_step_size
        self.s_old = None
        self.a_old = None
        self.alpha = alpha
        self.gamma = gamma
        self.train = train

        if train:
            # create new neuronal network
            self.model = Sequential()
            self.model.add(Dense(units=16, activation='relu', input_dim=2))
            self.model.add(Dense(units=32, activation='relu'))
            self.model.add(Dense(units=8, activation='linear'))
            self.model.compile(loss='mse', optimizer=Adam(learning_rate=0.00025))

            self.target_model = self.model = Sequential()
            self.target_model.add(Dense(units=16, activation='relu', input_dim=2))
            self.target_model.add(Dense(units=32, activation='relu'))
            self.target_model.add(Dense(units=8, activation='linear'))
            self.target_model.compile(loss='mse', optimizer=Adam(learning_rate=0.00025))
            self.target_model.set_weights(self.model.get_weights())

            self.experience_replay = []
            self.experience_replay_length = 0

    def step(self, obs):
        if self._MOVE_SCREEN.id in obs.observation.available_actions:
            marine = self._get_marine(obs)
            if marine is None:
                return self._NO_OP

            # 1. get state
            s_new = self.get_state(obs)

            if self.train:
                # 2. select action a according to policy
                a_new = self.apply_policy(s_new)

                # 3. get reward for s_new
                r_new = obs.reward

                # 4.  store transition
                self.update_experience_replay(Transition(self.s_old, self.a_old, r_new, s_new, obs.last() or r_new == 1))

                # 5. train model
                if self.experience_replay_length >= 6000:
                    # 5.1. sample random mini batch of transitions
                    mini_batch = self.generate_mini_batch()

                    # 5.2. execute gradient descent step
                    self.gradient_descent(mini_batch)

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
                a_new = self.apply_policy(s_new)

            # 7. | 3. take action
            marine_coords = self._get_unit_pos(marine)
            return self._dir_to_sc2_action(a_new, marine_coords)
        else:
            return self._SELECT_ARMY

    def save_model(self, filename):
        self.model.save(filename + "/DDQNetwork.h5")

    def load_model(self, filename):
        self.model = load_model(filename + "/DDQNetwork.h5")

    def get_state(self, obs):
        marine = self._get_marine(obs)
        if marine is None:
            return "None"
        marine_coords = self._get_unit_pos(marine)

        beacon = self._get_beacon(obs)
        if beacon is None:
            return self._NO_OP
        beacon_coords = self._get_unit_pos(beacon)

        # state s defines the relative position of the marine to the beacon
        s = (beacon_coords - marine_coords) / self.screen_size

        return s

    def apply_policy(self, s_new):
        # 3. select action a according to policy
        if np.random.rand() <= self.eps:
            best_action = np.random.choice(list(self._DIRECTIONS.keys()))
        else:
            # predict utility estimate for every action and take the best one
            best_action = self.get_best_action_key(self.model, s_new)

        return best_action

    def get_best_action_key(self, model, state):
        utilities = model.predict(np.array([[state[0], state[1]]]))[0]
        u_max = np.amax(utilities)
        indices = np.where(utilities == u_max)[0]
        idx = indices[np.random.randint(0, len(indices))]
        return list(self._DIRECTIONS.keys())[idx]

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def update_experience_replay(self, trans):
        if trans.s_old is None:
            return
        self.experience_replay.append(trans)
        self.experience_replay_length += 1
        if self.experience_replay_length > 100000:
            self.experience_replay.pop(0)
            self.experience_replay_length -= 1

    def generate_mini_batch(self):
        return [self.experience_replay[index] for index in np.random.choice(range(0, self.experience_replay_length), 32)]

    def gradient_descent(self, batch):
        # create training data
        x_train = np.asarray([b.s_old for b in batch])
        y_train = []
        y_model_old = self.model.predict_on_batch(x_train)                                          # predict for s_old
        y_model_new = self.model.predict_on_batch(np.asarray([b.s_new for b in batch]))             # predict for s_new
        y_target_new = self.target_model.predict_on_batch(np.asarray([b.s_new for b in batch]))     # predict for s_new

        for trans,  y_model_old, y_model_new, y_target_new in zip(batch, y_model_old, y_model_new, y_target_new):
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

        # apply gradient descent to model
        self.model.fit(np.array(x_train), np.array(y_train), verbose=False)
