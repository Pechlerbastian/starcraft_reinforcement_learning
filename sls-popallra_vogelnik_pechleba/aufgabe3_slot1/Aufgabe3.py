from copy import deepcopy


class Aufgabe3:
    def __init__(self):
        self.grid = [[0.0, 0.0, 0.0],
                ["X", 0.0, 0.0],
                [0.0, 0.0, 0.0]]
        self.print_grid()
        self.blocked = (1, 0)
        self.start = (2, 0)
        self.loss_per_iteration = -1.0
        self.positive_reward = 100.0
        self.negative_reward = -100.0
        self.positive_reward_position = (0, 2)
        self.negative_reward_position = (0, 0)

        self.iterations = 5

        self.alpha = 1 / 3
        self.gamma = 0.9

    def __run__(self):
        # step 1
        self.grid[self.positive_reward_position[0]][self.positive_reward_position[1]] = self.positive_reward
        self.grid[self.negative_reward_position[0]][self.negative_reward_position[1]] = self.negative_reward
        for i in range(len(self.grid)):
            for j in range(len(self.grid[:])):
                if self.grid[i][j] == 0:
                    self.grid[i][j] = self.loss_per_iteration
        self.print_grid()

        # step 2-5
        for i in range(1, self.iterations):
            self.grid_old = deepcopy(self.grid)
            self.update_grid()
            self.print_grid()

    def print_grid(self):
        print(self.grid[0])
        print(self.grid[1])
        print(self.grid[2])
        print("\n")

    def update_grid(self):
        for i in range(len(self.grid)):
            for j in range(len(self.grid[:])):
                if (i, j) != self.blocked and (i, j) != self.positive_reward_position and (i, j) != self.negative_reward_position:
                    self.grid[i][j] = self.find_max(i, j)

    def find_max(self, i, j):

        # north
        straight = (i - 1, j) if i - 1 >= 0 and (i - 1, j) != self.blocked else (i, j)
        right = (i, j + 1) if j + 1 <= 2 and (i, j + 1) != self.blocked else (i, j)
        left = (i, j - 1) if j - 1 >= 0 and (i, j - 1) != self.blocked else (i, j)

        north_reward = self.calc_reward(straight, right, left)

        # east
        straight = (i, j + 1) if j + 1 <= 2 and (i, j + 1) != self.blocked else (i, j)
        left = (i - 1, j) if i - 1 >= 0 and (i - 1, j) != self.blocked else (i, j)
        right = (i + 1, j) if i + 1 <= 2 and (i + 1, j) != self.blocked else (i, j)

        east_reward = self.calc_reward(straight, right, left)

        # south
        straight = (i + 1, j) if i + 1 <= 2 and (i + 1, j) != self.blocked else (i, j)
        right = (i, j - 1) if j - 1 >= 0 and (i, j - 1) != self.blocked else (i, j)
        left = (i, j + 1) if j + 1 <= 2 and (i, j + 1) != self.blocked else (i, j)

        south_reward = self.calc_reward(straight, right, left)

        # west
        straight = (i, j - 1) if j - 1 >= 0 and (i, j - 1) != self.blocked else (i, j)
        left = (i + 1, j) if i + 1 <= 2 and (i + 1, j) != self.blocked else (i, j)
        right = (i - 1, j) if i - 1 >= 0 and (i - 1, j) != self.blocked else (i, j)

        west_reward = self.calc_reward(straight, right, left)

        return max([north_reward, east_reward, south_reward, west_reward])

    def calc_reward(self, straight, right, left):
        result = self.alpha * (self.loss_per_iteration + self.gamma * self.grid_old[straight[0]][straight[1]])\
                 + self.alpha * (self.loss_per_iteration + self.gamma * self.grid_old[right[0]][right[1]]) \
                 + self.alpha * (self.loss_per_iteration + self.gamma * self.grid_old[left[0]][left[1]])
        return result


def main():
    Aufgabe3().__run__()


if __name__ == '__main__':
    main()
