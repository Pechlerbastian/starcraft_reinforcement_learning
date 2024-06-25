from sls.agents import AbstractAgent
import numpy as np


class BasicAgent(AbstractAgent):

    def __init__(self, train, screen_size):
        super(BasicAgent, self).__init__(screen_size)

    def step(self, obs):
        if self._MOVE_SCREEN.id in obs.observation.available_actions:
            marine = self._get_marine(obs)
            if marine is None:
                return self._NO_OP
            marine_coords = self._get_unit_pos(marine)

            beacon = self._get_beacon(obs)
            if beacon is None:
                return self._NO_OP
            beacon_coords = self._get_unit_pos(beacon)

            direction = beacon_coords - marine_coords                                                                   # compute direction vector
            abs_max = np.amax(np.abs(direction))                                                                        # get maximum absolute value
            direction = np.rint(direction / abs_max)                                                                    # transform values into elements of {-1, 0, +1}
            key = [k for k, v in self._DIRECTIONS.items() if np.array_equal(v, direction)][0]                           # get corresponding action key

            return self._dir_to_sc2_action(key, marine_coords)
        else:
            return self._SELECT_ARMY

    def save_model(self, filename):
        pass

    def load_model(self, filename):
        pass
