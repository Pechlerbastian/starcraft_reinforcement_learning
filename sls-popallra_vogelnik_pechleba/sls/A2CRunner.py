import datetime
import os
import tensorflow as tf


class A2CRunner(object):
    def __init__(self, agent, train, load_path, save_path):
        self.agent = agent
        self.train = train
        self.episode = 1

        self.path = save_path + datetime.datetime.now().strftime("%y%m%d_%H%M") \
                    + ('_train_' if self.train else '_run_') \
                    + type(agent).__name__

        # Tensorflow 1.X
        # self.writer = tf.summary.FileWriter(self.path, tf.get_default_graph())
        # Tensorflow 2.X mit ausgeschalteter eager_execution
        # Alle weiteren tf.summary Aufrufe müssen durch tf.compat.v1.summary ersetzt werden
        self.writer = tf.compat.v1.summary.FileWriter(self.path, tf.compat.v1.get_default_graph())

        if not self.train and load_path is not None and os.path.isdir(load_path):
            self.agent.load_model(load_path)
        self.moving_average_scores = []

    def summarize(self, score):
        self.writer.add_summary(tf.compat.v1.Summary(
             value=[tf.compat.v1.Summary.Value(tag='Average Score per Episode', simple_value=score)]),
             self.episode
        )

        # with self.writer.as_default():
        #    tf.summary.scalar('Score per Episode', self.score, step=self.episode)
        if self.train and self.episode % 10 == 0:
            self.agent.save_model(self.path)
            try:
                self.agent.update_target_model()
            except AttributeError:
                ...

        print(f"Episode {self.episode} finished with an average score of {score} over {self.agent.nWorkers} workers!")
        self.episode += 1

    def run(self, episodes):
        while self.episode <= episodes:
            while True:
                score = self.agent.step(None)
                # if score is not None, episode has been terminated
                if score is not None:
                    self.summarize(score / self.agent.nWorkers)
                    # reset workers if there are more episodes to go
                    if self.episode <= episodes:
                        self.agent.reset()
                    break
        self.agent.close()
        print(f"A2C Agent successfully closed. Application finished!")
