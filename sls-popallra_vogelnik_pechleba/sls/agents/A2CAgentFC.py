from multiprocessing import Pipe, Process
from tensorflow_core.python.keras.layers import Activation
from sls import Env
from sls.agents import AbstractAgent
import numpy as np
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow import stop_gradient
from pysc2.lib.features import SCREEN_FEATURES
import tensorflow.keras.backend as K
import tensorflow as tf


class Network:
    def __init__(self, alpha=0.0007, c_H=0.005, c_val=0.5):
        self.fit = None
        self.c_val = c_val
        self.c_H = c_H
        self._DIRECTIONS = {'N': [0, -1],
                            'NE': [1, -1],
                            'E': [1, 0],
                            'SE': [1, 1],
                            'S': [0, 1],
                            'SW': [-1, 1],
                            'W': [-1, 0],
                            'NW': [-1, -1]}

        # create new neuronal network
        self.input = Input(shape=(16, 16, 1))
        self.hidden_conv_1 = Conv2D(filters=16, kernel_size=5, strides=1, activation='relu', padding="same")(self.input)
        self.hidden_conv_2 = Conv2D(filters=32, kernel_size=3, strides=1, activation='relu', padding="same")(
            self.hidden_conv_1)

        self.conv_out = Conv2D(filters=1, kernel_size=1, strides=1, padding="same")(self.hidden_conv_2)
        # a permuted layer to move the channels to the last position:
        self.flatten_out = Flatten()(self.conv_out)
        # the softmax, now considering that channels sum 1.
        self.output_actor = Activation('softmax')(self.flatten_out)
        self.flatten = Flatten()(self.hidden_conv_2)
        self.hidden_layer = Dense(units=256, activation='relu')(self.flatten)
        self.output_critic = Dense(units=1, activation='linear')(self.hidden_layer)
        self.model = Model(self.input, outputs=[self.output_actor, self.output_critic])

        self.optimizer = Adam(learning_rate=alpha)
        self.build_train()

    def get_directions(self):
        return self._DIRECTIONS

    def save_model(self, filename):
        self.model.save(filename + "/AC2FCNetwork.h5")

    def load_model(self, filename):
        self.model = load_model(filename + "/AC2FCNetwork.h5")

    def loss(self, critic_predicted, actor_predicted, actions, targets):
        one_hot = K.one_hot(K.cast(actions, dtype='int32'), 256)
        actor_predicted = tf.clip_by_value(actor_predicted, 0.00001, 0.99999)
        # calculate advantages
        advantages = targets - critic_predicted[:, 0]

        # 1. value loss
        value_loss = tf.reduce_mean(K.pow(advantages, 2))

        # 2. policy loss
        # get probabilities of chosen actions
        const_advantages = stop_gradient(advantages)
        action_probabilities = tf.reduce_sum(one_hot * actor_predicted, axis=-1)
        # calculate policy loss
        policy_loss = tf.negative(tf.math.reduce_mean(const_advantages * K.log(action_probabilities)))

        # 3. entropy loss
        H = tf.math.reduce_sum(actor_predicted * K.log(actor_predicted), axis=-1)
        entropy_loss = tf.math.reduce_mean(H)

        loss = policy_loss + self.c_val * value_loss + self.c_H * entropy_loss

        return loss

    def gradient_ascent(self, states, actions, targets):
        error = self.fit([states, actions, targets])
        return error

    def build_train(self):
        actor_predicted = self.model.output[0]
        critic_predicted = self.model.output[1]
        actions = K.placeholder()
        targets = K.placeholder()

        loss = self.loss(critic_predicted, actor_predicted, actions, targets)
        params = self.model.trainable_weights
        update = self.optimizer.get_updates(loss=loss, params=params)

        self.fit = K.function(inputs=[self.model.input, actions, targets], outputs=[loss], updates=update)


class TrainingData:
    class Transition:
        def __init__(self, state, action, reward, done, critic):
            self.state = state
            self.action = action
            self.reward = reward
            self.done = done
            self.critic = critic

    class Sample:
        def __init__(self, state, action, target):
            self.state = state
            self.action = action
            self.target = target

    def __init__(self, nReturn, gamma):
        self.transitions = []
        self.samples = []
        self.nReturn = nReturn
        self.gamma = gamma

    def append_transition(self, state, action, reward, done, critic):
        self.transitions.append(self.Transition(state, action, reward, done, critic))
        if len(self.transitions) >= self.nReturn:
            self.__make_sample()

    def __make_sample(self):
        rewards = []
        last_critic = self.transitions[self.nReturn-1].critic
        for i in range(self.nReturn):
            rewards.append(self.transitions[i].reward)
            # only use rewards until final state (beacon reached or episode finished)
            if self.transitions[i].done:
                last_critic = 0
                break
        discounted_reward = self.__discount_rewards(rewards, last_critic)[0]
        state = self.transitions[0].state
        action = self.transitions[0].action
        self.samples.append(self.Sample(state=state, action=action, target=discounted_reward))
        self.transitions.pop(0)

    def __discount_rewards(self, reward, critic):
        # Compute the gamma-discounted rewards over an episode
        running_add = critic
        discounted_r = np.zeros_like(reward)
        for i in reversed(range(0, len(reward))):
            running_add = running_add * self.gamma + reward[i]
            discounted_r[i] = running_add
        return discounted_r

    def get_number_of_samples(self):
        return len(self.samples)

    def get_n_samples(self, n=None):
        if n is None:
            samples = self.samples
            self.samples = []
            return samples
        else:
            samples = self.samples[:n]
            [self.samples.pop(0) for _ in range(n)]
            return samples


def worker(screen_size, worker_pipe, identification, visualize):
    env = Env(screen_size, 64, visualize)
    obs = env.reset()

    def get_observation():
        worker_pipe.send([obs, identification])

    def step(sc2_action):
        nonlocal obs
        obs = env.step(sc2_action)
        get_observation()

    def reset():
        nonlocal obs
        obs = env.reset()
        get_observation()

    def close():
        worker_pipe.close()

    while True:
        action = worker_pipe.recv()
        if action[0] == "get":
            get_observation()
        elif action[0] == "reset":
            reset()
        elif action[0] == "action":
            step(action[1])
        elif action[0] == "close":
            close()
            break


class A2CAgentFC(AbstractAgent):
    def __init__(self, screen_size, alpha=0.0007, gamma=0.99, c_val=0.5, c_H=0.005, nWorkers=8, train=True,
                 batch_size=64, nReturn=5, visualize=False):
        super(A2CAgentFC, self).__init__(screen_size)
        self.gamma = gamma
        self.unit_density_index = SCREEN_FEATURES.unit_density.index
        self.network = Network(alpha=alpha, c_val=c_val, c_H=c_H)
        self.data = [TrainingData(nReturn=nReturn, gamma=gamma) for _ in range(nWorkers)]
        self.batch_size = batch_size
        self.score = 0
        self.nWorkers = nWorkers
        self.train = train
        self.worker_pipes = []
        self.processes = []
        if train:
            # initialize workers
            for i in range(self.nWorkers):
                pipe = Pipe()
                self.processes.append(Process(target=worker, args=(self.screen_size, pipe[0], i, visualize)))
                self.worker_pipes.append(pipe[1])
        else:
            self.nWorkers = 1
            pipe = Pipe()
            self.processes.append(Process(target=worker, args=(self.screen_size, pipe[0], 0, visualize)))
            self.worker_pipes.append(pipe[1])
        [process.start() for process in self.processes]
        self.sorted_obs = self.get_sorted_observation()
        print(f"{self.nWorkers} new workers initialized!")

    def load_model(self, filename):
        self.network.load_model(filename)

    def save_model(self, filename):
        self.network.save_model(filename)

    def get_sorted_observation(self):
        # send "get" command
        [worker_pipe.send(["get"]) for worker_pipe in self.worker_pipes]
        # receive observation for each worker
        unsorted_obs = [pipe.recv() for pipe in self.worker_pipes]
        # sort observations
        unsorted_obs.sort(key=lambda x: x[1])
        # only return observations, without identifier
        return [item[0] for item in unsorted_obs]

    def execute_actions(self, sc2_actions):
        for i in range(self.nWorkers):
            if self._MOVE_SCREEN.id in self.sorted_obs[i].observation.available_actions:
                marine = self._get_marine(self.sorted_obs[i])
                if marine is None:
                    self.worker_pipes[i].send(["action", self._NO_OP])
                else:
                    self.worker_pipes[i].send(["action", sc2_actions[i]])
            else:
                self.worker_pipes[i].send(["action", self._SELECT_ARMY])

        # receive new observation for each worker
        unsorted_obs = [pipe.recv() for pipe in self.worker_pipes]
        # sort observations
        unsorted_obs.sort(key=lambda x: x[1])
        # only return observations, without identifier
        return [item[0] for item in unsorted_obs]

    def step(self, observation):        # parameter 'observation' only needed for overwriting of abstract method
        # 1. get states
        unit_densities = [obs.observation["feature_screen"][self.unit_density_index] for obs in self.sorted_obs]
        old_states = [np.asarray(u_d[:, :, np.newaxis]) for u_d in unit_densities]

        # 2. predict actor and critic
        model_output = self.network.model.predict_on_batch(np.asarray(old_states))[:]
        action_probabilities = model_output[0]
        action_indices = [np.random.choice(np.arange(stop=256), size=1, p=prob)[0] for prob in action_probabilities]
        goal_points = [np.asarray([a % 16, a // 16]) for a in action_indices]
        sc2_actions = [self._MOVE_SCREEN("now", g) for g in goal_points]

        # 3. execute actions and receive new observations
        new_sorted_obs = self.execute_actions(sc2_actions)

        # 4. update score
        self.score += sum([obs.reward for obs in new_sorted_obs])
        new_done = [obs.last() for obs in new_sorted_obs]
        if self.train:
            # 5. add transitions
            unit_densities = [obs.observation["feature_screen"][self.unit_density_index] for obs in new_sorted_obs]
            new_states = [np.asarray(u_d[:, :, np.newaxis]) for u_d in unit_densities]
            model_output_new = self.network.model.predict_on_batch(np.asarray(new_states))[:]
            critic_predictions = [critic[0] for critic in model_output_new[1]]
            new_rewards = [1 if obs.reward == 1 else -0.01 for obs in new_sorted_obs]

            for i in range(self.nWorkers):
                # only append transition if last state was not a final state (the last one of an episode)
                if not self.sorted_obs[i].last():
                    # put action index into transition instead of action key in order to comply with the implemented
                    # loss function for gradient descent
                    self.data[i].append_transition(state=old_states[i], action=action_indices[i], reward=new_rewards[i],
                                                   done=(new_done[i] or new_sorted_obs[i].reward == 1),
                                                   critic=critic_predictions[i])

            # 6. update observation
            self.sorted_obs = new_sorted_obs

            # 7. check for possible network update
            nSamples = [d.get_number_of_samples() for d in self.data]
            if sum(nSamples) >= self.batch_size:
                batch = []
                for i in range(self.nWorkers):
                    if self.batch_size - len(batch) >= nSamples[i]:
                        batch.append(self.data[i].get_n_samples())
                    else:
                        batch.append(self.data[i].get_n_samples(self.batch_size - len(batch)))
                        break

                batch = np.asarray(batch).flatten()
                states = np.asarray([b.state for b in batch])
                actions = np.asarray([b.action for b in batch])
                targets = np.asarray([b.target for b in batch])
                self.network.gradient_ascent(states=states, actions=actions, targets=targets)
        self.sorted_obs = new_sorted_obs
        # 8. return score if any worker finished its episode, None otherwise
        if True in new_done:
            return self.score
        else:
            return None

    def reset(self):
        # reset workers
        [worker_pipe.send(["reset"]) for worker_pipe in self.worker_pipes]
        # receive new observation for each worker
        unsorted_obs = [pipe.recv() for pipe in self.worker_pipes]
        # sort observations
        unsorted_obs.sort(key=lambda x: x[1])
        # # reset observation
        self.sorted_obs = [item[0] for item in unsorted_obs]
        self.score = 0

    def close(self):
        # close child connections of worker pipes and terminate processes
        [worker_pipe.send(["close"]) for worker_pipe in self.worker_pipes]
        # close parents connections of worker pipes
        [worker_pipe.close() for worker_pipe in self.worker_pipes]
        # sync
        [process.join() for process in self.processes]
        # close all existing processes
        [process.close() for process in self.processes]
        # return something to archive blocking function call
        return True
