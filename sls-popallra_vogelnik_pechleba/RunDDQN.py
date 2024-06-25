from absl import app
from sls import Env
from sls.DQNRunner import DQNRunner
from sls.agents.DDQNAgent import DDQNAgent

_CONFIG = dict(
    episodes=200,
    screen_size=16,
    minimap_size=16,
    visualize=False,
    train=False,
    agent=DDQNAgent,
    load_path='./graphs/DDQN/220626_0927_train_DDQNAgent',
    save_path='./graphs/DDQN/',
    moving_average_window_size=50,
    eps=0.05,
    alpha=0.2,
    gamma=0.9
)


def main(unused_argv):

    agent = _CONFIG['agent'](
        train=_CONFIG['train'],
        screen_size=_CONFIG['screen_size'],
        eps=_CONFIG['eps'],
        eps_step_size=(_CONFIG['eps'] - 0.05) / 500,
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
