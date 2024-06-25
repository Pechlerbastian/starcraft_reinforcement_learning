from sls.agents import PGAgent
from absl import app
from sls import Env
from sls.DQNRunner import DQNRunner

_CONFIG = dict(
    episodes=200,
    screen_size=16,
    minimap_size=64,
    visualize=True,
    train=False,
    agent=PGAgent,
    load_path='./graphs/PG/220710_1118_train_PGAgent',
    save_path='./graphs/PG/',
    moving_average_window_size=50,
    gamma=0.99,
    alpha=0.00025
)


def main(unused_argv):

    agent = _CONFIG['agent'](
        train=_CONFIG['train'],
        screen_size=_CONFIG['screen_size'],
        alpha=_CONFIG['alpha'],
        gamma=_CONFIG['gamma']
    )

    env = Env(
        screen_size=_CONFIG['screen_size'],
        minimap_size=_CONFIG['minimap_size'],
        visualize=_CONFIG['visualize']
    )

    runner = DQNRunner(
        agent=agent,
        env=env,
        train=_CONFIG['train'],
        load_path=_CONFIG['load_path'],
        save_path=_CONFIG['save_path'],
        moving_average_window_size=_CONFIG['moving_average_window_size']
    )

    runner.run(episodes=_CONFIG['episodes'])


if __name__ == "__main__":
    app.run(main)
