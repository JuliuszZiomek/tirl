from barl.viz.plot import (
    plot_pendulum,
    plot_cartpole,
    plot_pilco_cartpole,
    plot_acrobot,
    # noop,
    plot_lava_path,
    make_plot_obs,
    plot_generic,
    scatter,
    plot
    )
plotters = {
        'bacpendulum-v0': plot_pendulum,
        'bacpendulum-test-v0': plot_pendulum,
        'bacpendulum-tight-v0': plot_pendulum,
        'bacpendulum-medium-v0': plot_pendulum,
        'petscartpole-v0': plot_cartpole,
        'pilcocartpole-v0': plot_pilco_cartpole,
        'bacrobot-v0': plot_acrobot,
        'bacswimmer-v0': plot_generic,
        'bacreacher-v0': plot_generic,
        'bacreacher-tight-v0': plot_generic,
        'lavapath-v0': plot_lava_path,
        'betatracking-v0': plot_generic,
        'betatracking-fixed-v0': plot_generic,
        'plasmatracking-v0': plot_generic,
        }
