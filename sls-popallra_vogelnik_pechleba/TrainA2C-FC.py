from sls.agents import A2CAgentFC
from absl import app
from sls.A2CRunner import A2CRunner

_CONFIG = dict(
    episodes=3000,
    screen_size=16,
    minimap_size=64,
    visualize=False,
    train=True,
    agent=A2CAgentFC,
    load_path='./graphs/A2C-FC/...',
    save_path='./graphs/A2C-FC/',
    gamma=0.99,
    alpha=0.0007,
    c_val=0.5,
    c_H=0.005,
    nWorkers=8,
    batch_size=64,
    nReturn=5
)


def main(unused_argv):

    agent = _CONFIG['agent'](
        screen_size=_CONFIG['screen_size'],
        alpha=_CONFIG['alpha'],
        gamma=_CONFIG['gamma'],
        c_val=_CONFIG['c_val'],
        c_H=_CONFIG['c_H'],
        nWorkers=_CONFIG['nWorkers'],
        train=_CONFIG['train'],
        batch_size=_CONFIG['batch_size'],
        nReturn=_CONFIG['nReturn'],
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
