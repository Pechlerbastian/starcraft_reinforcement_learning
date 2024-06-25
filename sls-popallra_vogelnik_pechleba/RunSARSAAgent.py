from absl import app
from sls import Env
from sls.agents import *
from sls.SARSARunner import SARSARunner

_CONFIG = dict(
    episodes=200,
    screen_size=64,
    minimap_size=64,
    visualize=False,
    train=False,
    agent=SARSAAgent,
    load_path='graphs/SARSA/220526_0950_train_SARSAAgent',
    save_path='./graphs/SARSA/',
    moving_average_window_size=50,
)


def main(unused_argv):

    agent = _CONFIG['agent'](
        train=_CONFIG['train'],
        screen_size=_CONFIG['screen_size']
    )

    env = Env(
        screen_size=_CONFIG['screen_size'],
        minimap_size=_CONFIG['minimap_size'],
        visualize=_CONFIG['visualize']
    )

    runner = SARSARunner(
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
