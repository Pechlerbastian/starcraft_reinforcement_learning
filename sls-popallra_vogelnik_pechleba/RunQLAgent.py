from absl import app
from sls import Env, Runner
from sls.agents import *

_CONFIG = dict(
    episodes=100,
    screen_size=64,
    minimap_size=64,
    visualize=False,
    train=False,
    agent=QLAgent,
    load_path='graphs/QL/220512_2129_train_QLAgent'
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

    runner = Runner(
        agent=agent,
        env=env,
        train=_CONFIG['train'],
        load_path=_CONFIG['load_path']
    )

    runner.run(episodes=_CONFIG['episodes'])


if __name__ == "__main__":
    app.run(main)
