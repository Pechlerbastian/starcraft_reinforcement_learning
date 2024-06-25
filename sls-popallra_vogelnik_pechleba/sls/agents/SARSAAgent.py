from sls.agents import AbstractAgent
import numpy as np
import pandas as pd


class SARSAAgent(AbstractAgent):

    def __init__(self, train, screen_size, alpha=0.1, gamma=0.9):
        super(SARSAAgent, self).__init__(screen_size)

        self.delta = 2
        self.s_old = None
        self.a_old = None
        self.alpha = alpha
        self.gamma = gamma
        self.train = train
        self.P_best = 0.2

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
                # 2. select action a according to policy
                a_new = self.select_action(s_new)

                # 3. update Q table
                if self.s_old is not None:
                    self.update(obs, s_new, a_new)

                # 4. safe state and action for next iteration and update eps eventually
                self.s_old = s_new
                self.a_old = a_new
                if obs.last() or obs.reward == 1:
                    self.s_old = None
            else:
                # 2. select action
                # action = self.Q_table.loc[s_new].idxmax() is unsuitable here as it always returns the first
                # appearance of the max value -> if several actions with same value -> choose one randomly
                series = self.Q_table.loc[s_new]
                indices = series[series == series.max()].index
                a_new = indices[np.random.randint(0, len(indices))]

            # 5. | 3. take action
            marine_coords = self._get_unit_pos(marine)
            return self._dir_to_sc2_action(a_new, marine_coords)
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

    def select_action(self, s_new):
        """
        Selects an action according to the semi-uniform distributed exploration strategy. If several actions have the
        same highest utility estimate, select a random one amongst them.

        Description:
        "Rather than generating actions randomly with uniform probability, an alternative would
        be to generate actions with a probability distribution that is based on the utility estimates
        that are currently available to the agent. One such approach selects the action having the
        highest current utility estimate with some predefined probability P_best. Each of the other
        actions is selected with probability 1 – P_best regardless of its currently utility estimate.

        if a maximizes utility estimate:
            P(a) = P_best + (1 - P_best) / # of actions
        else:
            P(a) = (1 - P_best) / # of actions

        The P_best parameter facilitates a continuous blend of exploration and exploitation that
        ranges from purely random exploration (P_best = 0) to pure exploitation (P_best = 1)."

        Literatur: https://www.cs.mcgill.ca/~cs526/roger.pdf

        :return: the new action
        """
        actions = self.Q_table.loc[s_new]
        indices = actions[actions == actions.max()].index
        a_best = indices[np.random.randint(0, len(indices))]

        prob = []
        for a in actions.index:
            if a == a_best:
                prob.append(self.P_best + (1 - self.P_best) / len(actions))
            else:
                prob.append((1 - self.P_best) / len(actions))
        prob = np.cumsum(prob)
        idx = np.where(prob >= np.random.rand())[0][0]

        return actions.index[idx]

    def update(self, obs, s_new, a_new):
        # TODO: "if s' is not terminal" -> was genau heißt hier terminal? Beacon erreicht oder Episode zu Ende? Beides?
        """
        Updates the agent`s Q table.

        :param obs: current observation object
        :param s_new: current state (equals new state of last iteration on which update belongs to)
        :param a_new: current action (equals new action of last iteration on which update belongs to)
        """
        # get reward r
        r = obs.reward

        # update Q table
        if r != 1 and not obs.last():
            self.Q_table.at[self.s_old, self.a_old] += self.alpha * (r + self.gamma * self.Q_table.at[s_new, a_new] -
                                                                     self.Q_table.at[self.s_old, self.a_old])
        else:
            self.Q_table.at[self.s_old, self.a_old] += self.alpha * (r - self.Q_table.at[self.s_old, self.a_old])
