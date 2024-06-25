from sls.agents import A2CAgentFC
from absl import app
from sls.A2CRunner import A2CRunner

_CONFIG = dict(
    episodes=200,
    screen_size=16,
    visualize=True,
    train=False,
    agent=A2CAgentFC,
    load_path='./graphs/A2C-FC/220805_1127_train_A2CAgentFC',
    save_path='./graphs/A2C-FC/'
)


def main(unused_argv):

    agent = _CONFIG['agent'](
        screen_size=_CONFIG['screen_size'],
        train=_CONFIG['train'],
        visualize=_CONFIG['visualize']
    )

    runner = A2CRunner(
        agent=agent,
        train=_CONFIG['train'],
        load_path=_CONFIG['load_path'],
        save_path=_CONFIG['save_path']
    )

    runner.run(episodes=_CONFIG['episodes'])


if __name__ == "__main__":
    app.run(main)
