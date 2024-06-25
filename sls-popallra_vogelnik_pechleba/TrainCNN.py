from absl import app
from sls import Env
from sls.DQNRunner import DQNRunner
from sls.agents.CNNAgent import CNNAgent

_CONFIG = dict(
    episodes=750,
    screen_size=16,
    minimap_size=64,
    visualize=False,
    train=True,
    agent=CNNAgent,
    load_path='./graphs/CNN/...',
    save_path='./graphs/CNN/',
    moving_average_window_size=50,
    eps=1.0,
    gamma=0.9
)


def main(unused_argv):

    agent = _CONFIG['agent'](
        train=_CONFIG['train'],
        screen_size=_CONFIG['screen_size'],
        eps=_CONFIG['eps'],
        eps_step_size=(_CONFIG['eps'] - 0.05) / 500,
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
