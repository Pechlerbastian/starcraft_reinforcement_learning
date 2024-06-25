from sls.agents import AbstractAgent
import numpy as np
import pandas as pd


class QLAgent(AbstractAgent):

    def __init__(self, train, screen_size, eps=0.0, eps_step_size=0.0, alpha=0.1, gamma=0.9):
        super(QLAgent, self).__init__(screen_size)

        self.delta = 2
        self.eps = eps
        self.eps_step_size = eps_step_size
        self.s_old = None
        self.a_old = None
        self.alpha = alpha
        self.gamma = gamma
        self.train = train

        if train:
            # create new Q-table
            states = ["(0,0)"]
            for i in range(-(int((screen_size / self.delta)) + 1), int((screen_size / self.delta)) + 1):
                for j in range(-(int((screen_size / self.delta)) + 1), int((screen_size / self.delta)) + 1):
                    states.append(f"({i},{j})")
            actions = self._DIRECTIONS.keys()
            data = np.zeros((len(states), len(actions)))
            self.Q_table = pd.DataFrame(data=data, index=states, columns=actions)

    def step(self, obs):
        if self._MOVE_SCREEN.id in obs.observation.available_actions:
            marine = self._get_marine(obs)
            if marine is None:
                return self._NO_OP

            # 1. get state
            s_new = self.get_state(obs)

            if self.train:
                # 2. update Q table
                if self.s_old is not None:
                    self.update(obs, s_new)

                # 3. select action a according to policy
                if np.random.rand() <= self.eps:
                    action = np.random.choice(list(self._DIRECTIONS.keys()))
                else:
                    # action = self.Q_table.loc[s_new].idxmax() does not work properly as it always returns the first
                    # appearance of the max value -> almost always returns same action, especially in the beginning after
                    # the Q table has been initialized with 0s -> almost no exploring
                    series = self.Q_table.loc[s_new]
                    indices = series[series == series.max()].index
                    action = indices[np.random.randint(0, len(indices))]

                # 4. safe state and action for next iteration and update eps eventually
                if obs.reward != 1:
                    self.s_old = s_new
                else:
                    self.s_old = None
                self.a_old = action
                if obs.last():
                    self.eps -= self.eps_step_size
                    self.s_old = None
            else:
                # 2. select action
                # action = self.Q_table.loc[s_new].idxmax() is unsuitable here as it always returns the first
                # appearance of the max value -> if several actions with same value -> choose one randomly
                series = self.Q_table.loc[s_new]
                indices = series[series == series.max()].index
                action = indices[np.random.randint(0, len(indices))]

            # 5. | 3. take action
            marine_coords = self._get_unit_pos(marine)
            return self._dir_to_sc2_action(action, marine_coords)
        else:
            return self._SELECT_ARMY

    def save_model(self, filename):
        self.Q_table.to_pickle(filename + "/QTable.pkl")

    def load_model(self, filename):
        self.Q_table = pd.read_pickle(filename + "/QTable.pkl")

    def get_state(self, obs):
        """
        Returns the current state as string description.

        :param obs: current observation object
        :return: current state as string
        """
        marine = self._get_marine(obs)
        if marine is None:
            return "None"
        marine_coords = self._get_unit_pos(marine)

        beacon = self._get_beacon(obs)
        if beacon is None:
            return self._NO_OP
        beacon_coords = self._get_unit_pos(beacon)

        # state s defines the relative position of the marine to the beacon
        s = (beacon_coords - marine_coords) / self.delta
        for i in [0, 1]:
            if s[i] > 0:
                s[i] = np.ceil(s[i])
            else:
                s[i] = np.floor(s[i])
        s = s.astype(int)

        return f"({s[0]},{s[1]})"

    def update(self, obs, s_new):
        """
        Updates the agent`s Q table.

        :param obs: current observation object
        :param s_new: current state (equals new state of last iteration on which update belongs to)
        """
        # get reward r
        r = obs.reward

        # update Q table
        if r != 1:
            self.Q_table.at[self.s_old, self.a_old] += self.alpha * (r + self.gamma * self.Q_table.loc[s_new].max() -
                                                                     self.Q_table.at[self.s_old, self.a_old])
        else:
            self.Q_table.at[self.s_old, self.a_old] += self.alpha * (r - self.Q_table.at[self.s_old, self.a_old])
