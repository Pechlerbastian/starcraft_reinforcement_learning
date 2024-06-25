import tensorflow as tf
from sls.runner import Runner


class SARSARunner(Runner):
    def __init__(self, agent, env, train, load_path, save_path, moving_average_window_size):
        super(SARSARunner, self).__init__(agent, env, train, load_path, save_path)
        self.moving_average_window_size = moving_average_window_size
        self.moving_average_scores = []

    def summarize(self):
        self.moving_average_scores.append(self.score)
        if len(self.moving_average_scores) > self.moving_average_window_size:
            del self.moving_average_scores[0]

        self.writer.add_summary(tf.compat.v1.Summary(
             value=[tf.compat.v1.Summary.Value(tag='Score per Episode', simple_value=self.score)]),
             self.episode
        )

        self.writer.add_summary(tf.compat.v1.Summary(
            value=[tf.compat.v1.Summary.Value(tag=f'Moving Average Score of the last {self.moving_average_window_size} Episodes',
                                              simple_value=sum(self.moving_average_scores) / len(self.moving_average_scores))]),
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
        self.episode += 1
        self.score = 0
