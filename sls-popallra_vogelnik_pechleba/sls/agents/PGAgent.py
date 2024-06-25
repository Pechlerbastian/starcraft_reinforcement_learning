from sls.agents import AbstractAgent
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K


class Transition:
    def __init__(self, state, action, reward):
        self.state = state
        self.action = action
        self.reward = reward


def get_reward(obs):
    if obs.reward == 1:
        return 100
    else:
        return -0.1


class PGAgent(AbstractAgent):
    def __init__(self, train, screen_size=16, gamma=0.99, alpha=0.00025):
        super(PGAgent, self).__init__(screen_size)
        self.train = train
        self.gamma = gamma
        self.alpha = alpha
        self.s_old = None
        self.a_old = None
        self.p_old = None
        self.fit = None

        if train:
            self.experience_replay = []

            # create new neuronal network
            self.model = Sequential()
            self.model.add(Dense(units=128, activation='relu', input_dim=2))
            self.model.add(Dense(units=256, activation='relu'))
            self.model.add(Dense(units=8, activation='softmax'))
            self.optimizer = Adam(learning_rate=self.alpha)
            self.build_train()

    def step(self, obs):
        if self._MOVE_SCREEN.id in obs.observation.available_actions:
            marine = self._get_marine(obs)
            if marine is None:
                return self._NO_OP

            # 1. get state
            state = self.get_state(obs)

            if self.train:
                # 2. get reward
                reward = get_reward(obs)
                self.update_experience_replay(self.s_old, self.a_old, reward)

                # 3. select action a according to policy
                action = self.apply_policy(state)

                # 4. update variables

                self.s_old = state
                self.a_old = action
                # Abbruch: Episode vorbei oder Ziel erreicht, dann Training und neuen Pfad bestimmen
                if obs.last() or obs.reward == 1:
                    # 5. da update von Expirience Replay zu beginn muss bei Abbruch noch ein Wert eingefügt werden
                    reward = get_reward(obs)
                    self.update_experience_replay(self.s_old, self.a_old, reward)
                    # 6. Training
                    self.gradient_ascent()
                    self.experience_replay.clear()
                    self.s_old = None
                    self.a_old = None

            else:
                # 2. select action
                action = self.apply_policy(state)

            # 7. | 3. take action
            marine_coords = self._get_unit_pos(marine)
            return self._dir_to_sc2_action(action, marine_coords)
        else:
            return self._SELECT_ARMY

    def save_model(self, filename):
        self.model.save(filename + "/PGNetwork.h5")

    def load_model(self, filename):
        self.model = load_model(filename + "/PGNetwork.h5")

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

    def update_experience_replay(self, state, action, reward):
        if state is None or action is None:
            return
        if action == self._NO_OP:
            return
        trans = Transition(state=state, action=action, reward=reward)
        self.experience_replay.append(trans)

    def apply_policy(self, s_new):
        # select action a according to policy via probabilities from the model
        # predict utility estimate for every action and take one according to the probabilities
        probabilities = self.model.predict(np.array([[s_new[0], s_new[1]]]))[0]
        action = np.random.choice(list(self._DIRECTIONS.keys()), p=probabilities)
        return action

    def loss(self, predicted, actions, target):
        one_hot = K.one_hot(K.cast(actions, dtype='int32'), len(self._DIRECTIONS))
        # hier wandeln wir die gewählten Actionen über die Steps in eine Matrix um,
        # in der jeweils  die gewählte Aktion eine 1 bekommt, die nicht gewählten eine 0
        # dann müssen wir die one hot encoded Actions nehmen
        # und sie mit dem output des Modells multiplizieren, um die Wahrscheinlichkeit
        # für die gewählte Aktion zu erhalten

        # Am Ende muss über die Aktionen addiert werden, die nicht gewählten (0 in hot enc) haben keinen Einfluss

        action_probabilities = K.sum(one_hot * predicted, axis=-1)

        # loss berechnen mit Matrixmultiplikation
        loss = -1 * K.log(action_probabilities) * target

        # dann Error mit K.mean
        error = K.mean(loss)

        return error

    def calculate_G(self):
        # TODO / CANDO: von hinten durchgehen -> letzter GT dann GT-1 .. immer mit gammas multiplizieren für Effizienz
        G = []
        for i in range(len(self.experience_replay)):
            G_t = 0
            for j in range(i, len(self.experience_replay)):
                G_t += self.experience_replay[j].reward * self.gamma ** (j - i)
            G.append(G_t)
        return G

    def build_train(self):
        target = K.placeholder()
        actions = K.placeholder()

        # TODO / CANDO Tipp: prüfen ob state Actions reward zusammenpassen --> letzte nach Ziel muss 100, state muss nah dran sein
        prediction = self.model.output
        error = self.loss(prediction, actions, target)
        params = self.model.trainable_weights
        update = self.optimizer.get_updates(loss=error, params=params)
        self.fit = K.function(inputs=[self.model.input, actions, target], outputs=[error], updates=update)

    def gradient_ascent(self):
        # x_train enthält Wahrscheinlichkeiten, von Modell predicted
        x_train = np.asarray([transition.state for transition in self.experience_replay])
        # gewählte actions werden gemappt auf Zahlen, diese werden in Array gespeichert
        # pro Step eine Zahl, die gewählter Aktoin entspricht
        actions = np.asarray(
            [list(self._DIRECTIONS.keys()).index(transition.action) for transition in self.experience_replay])
        # y_train enthält die Liste der diskontierten Rewards der Episode zu jedem Step t
        y_train = np.asarray(self.calculate_G())

        err = self.fit([x_train, actions, y_train])
