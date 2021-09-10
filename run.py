"""
Testing MultiGpfsGp and MultiBaxAcqFunction classes
"""
from argparse import Namespace
import logging
import numpy as np
import gym
from tqdm import trange
from functools import partial
import tensorflow as tf
import hydra
import random
from matplotlib import pyplot as plt

from bax.models.gpfs_gp import BatchMultiGpfsGp
from bax.models.gpflow_gp import get_gpflow_hypers_from_data
from bax.acq.acquisition import MultiBaxAcqFunction, MCAcqFunction
from bax.acq.acqoptimize import AcqOptimizer
from bax.alg.mpc import MPC
from bax import envs
from bax.envs.wrappers import NormalizedEnv, make_normalized_reward_function, make_update_obs_fn
from bax.util.misc_util import Dumper, make_postmean_fn
from bax.util.control_util import get_f_batch_mpc, get_f_batch_mpc_reward, compute_return, evaluate_policy
from bax.util.control_util import rollout_mse, mse
from bax.util.domain_util import unif_random_sample_domain
from bax.util.timing import Timer
from bax.viz import plotters
import neatplot

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

@hydra.main(config_path='cfg', config_name='config')
def main(config):
    dumper = Dumper(config.name)

    # Set plot settings
    neatplot.set_style()
    neatplot.update_rc('figure.dpi', 120)
    neatplot.update_rc('text.usetex', False)

    # Set random seed
    seed = config.seed
    np.random.seed(seed)
    tf.random.set_seed(seed)

    assert (not config.fixed_start_obs) or config.num_samples_mc == 1, f"Need to have a fixed start obs ({config.fixed_start_obs}) or only 1 mc sample ({config.num_samples_mc})"  # NOQA

    # -------------
    # Start Script
    # -------------
    # Set black-box function
    env = gym.make(config.env.name)
    env.seed(seed)
    obs_dim = env.observation_space.low.size
    action_dim = env.action_space.low.size

    plan_env = gym.make(config.env.name)
    plan_env.seed(seed)
    reward_function = envs.reward_functions[config.env.name] if not config.alg.learn_reward else None
    if config.normalize_env:
        env = NormalizedEnv(env)
        plan_env = NormalizedEnv(plan_env)
        if reward_function is not None:
            reward_function = make_normalized_reward_function(plan_env, reward_function)
    if config.alg.learn_reward:
        f = get_f_batch_mpc_reward(plan_env, use_info_delta=config.teleport)
    else:
        f = get_f_batch_mpc(plan_env, use_info_delta=config.teleport)
    update_fn = make_update_obs_fn(env, teleport=config.teleport)

    start_obs = env.reset() if config.fixed_start_obs else None
    logging.info(f"Start obs: {start_obs}")

    # Set domain
    low = np.concatenate([env.observation_space.low, env.action_space.low])
    high = np.concatenate([env.observation_space.high, env.action_space.high])
    domain = [elt for elt in zip(low, high)]

    # Set algorithm
    algo_class = MPC
    algo_params = dict(
            start_obs=start_obs,
            env=plan_env,
            reward_function=reward_function,
            project_to_domain=False,
            base_nsamps=config.mpc.nsamps,
            planning_horizon=config.mpc.planning_horizon,
            n_elites=config.mpc.n_elites,
            beta=config.mpc.beta,
            gamma=config.mpc.gamma,
            xi=config.mpc.xi,
            num_iters=config.mpc.num_iters,
            actions_per_plan=config.mpc.actions_per_plan,
            domain=domain,
            action_lower_bound=env.action_space.low,
            action_upper_bound=env.action_space.high,
            crop_to_domain=config.crop_to_domain,
            update_fn=update_fn,
    )
    algo = algo_class(algo_params)

    # Set data
    data = Namespace()
    data.x = unif_random_sample_domain(domain, config.num_init_data)
    data.y = f(data.x)

    # Make a test set for model evalution separate from the controller
    test_data = Namespace()
    test_data.x = unif_random_sample_domain(domain, config.test_set_size)
    test_data.y = f(test_data.x)

    # Set model
    gp_params = {'ls': config.env.gp.ls, 'alpha': config.env.gp.alpha, 'sigma': config.env.gp.sigma, 'n_dimx': obs_dim +
                 action_dim}
    multi_gp_params = {'n_dimy': obs_dim, 'gp_params': gp_params}
    gp_model_class = BatchMultiGpfsGp

    # Set acqfunction
    acqfn_params = {'n_path': config.n_paths, 'crop': True}
    acqfn_class = MultiBaxAcqFunction

    # Compute true path and associated test set
    true_algo = algo_class(algo_params)
    #full_path, output = true_algo.run_algorithm_on_f(f)
    #true_path = true_algo.get_exe_path_crop()
    #true_planning_data = list(zip(true_algo.exe_path.x, true_algo.exe_path.y))
    #test_points = random.sample(true_planning_data, config.test_set_size)
    #test_mpc_data = Namespace()
    #test_mpc_data.x = [tp[0] for tp in test_points]
    #test_mpc_data.y = [tp[1] for tp in test_points]

    # set plot fn
    plot_fn = partial(plotters[config.env.name], env=plan_env)

    ax_gt, fig_gt = None, None
    # Compute and plot true path (on true function) multiple times
    full_paths = []
    true_paths = []
    returns = []
    path_lengths = []
    all_test_mpc_data = Namespace(x=[], y=[])
    pbar = trange(config.num_eval_trials)
    for i in pbar:
        full_path, output = true_algo.run_algorithm_on_f(f)
        full_paths.append(full_path)
        path_lengths.append(len(full_path.x))
        tp = true_algo.get_exe_path_crop()
        true_paths.append(tp)
        if i==0:
            true_path = tp
            true_planning_data = list(zip(true_algo.exe_path.x, true_algo.exe_path.y))
            test_points = random.sample(true_planning_data, config.test_set_size)
            test_mpc_data = Namespace()
            test_mpc_data.x = [test_pt[0] for test_pt in test_points]
            test_mpc_data.y = [test_pt[1] for test_pt in test_points]
        # -----
        tpd = list(zip(true_algo.exe_path.x, true_algo.exe_path.y))
        test_points = random.sample(tpd, int(config.test_set_size/config.num_eval_trials))
        new_x = [test_pt[0] for test_pt in test_points]
        new_y = [test_pt[1] for test_pt in test_points]
        all_test_mpc_data.x.extend(new_x)
        all_test_mpc_data.y.extend(new_y)
        # -----
        ax_gt, fig_gt = plot_fn(tp, ax_gt, fig_gt, domain, 'samp')
        returns.append(compute_return(output[2], 1))
        stats = {"Mean Return": np.mean(returns), "Std Return:": np.std(returns)}
        pbar.set_postfix(stats)
    returns = np.array(returns)
    dumper.add('GT Returns', returns)
    path_lengths = np.array(path_lengths)
    logging.info(f"GT Returns: returns{returns}")
    logging.info(f"GT Returns: mean={returns.mean()} std={returns.std()}")
    logging.info(f"GT Execution: path_lengths.mean()={path_lengths.mean()} path_lengths.std()={path_lengths.std()}")
    all_x = []
    for fp in full_paths:
        all_x += fp.x
    all_x = np.array(all_x)
    print(f"all_x.shape = {all_x.shape}")
    print(f"all_x.min(axis=0) = {all_x.min(axis=0)}")
    print(f"all_x.max(axis=0) = {all_x.max(axis=0)}")
    print(f"all_x.mean(axis=0) = {all_x.mean(axis=0)}")
    print(f"all_x.var(axis=0) = {all_x.var(axis=0)}")
    neatplot.save_figure(str(dumper.expdir / 'mpc_gt'), 'png', fig=fig_gt)

    #### ----- Attempt re-set data.x as gt data
    #data.x = test_mpc_data.x
    #data.y = test_mpc_data.y
    #
    #data.x = all_test_mpc_data.x
    #data.y = all_test_mpc_data.y
    #
    #data.x = true_path.x
    #data.y = true_path.y
    #
    #data.x, data.y = [], []
    #for tp in true_paths:
        #data.x += tp.x
        #data.y += tp.y

    # Plot initial data
    ax_obs_init, fig_obs_init = plot_fn(path=None, domain=domain)
    x_obs = [xi[0] for xi in data.x]
    y_obs = [xi[1] for xi in data.x]
    ax_obs_init.plot(x_obs, y_obs, 'o', color='k', ms=1)
    neatplot.save_figure(str(dumper.expdir / f'mpc_obs_init'), 'png', fig=fig_obs_init)
    #### ----- ----- ----- ----- ----- -----

    # Optionally: print fit for GP hyperparameters (only prints; still uses hypers in config)
    if config.fit_hypers:
        print('***** Fitting GP hyperparameters *****')
        #fit_data = test_mpc_data
        fit_data = data ##### NOTE
        print(f'Number of observations in fit_data: {len(fit_data.x)}')
        assert len(fit_data.x) <= 3000, "fit_data larger than preset limit (can cause memory issues)"
        for idx in range(len(data.y[0])):
            data_fit = Namespace(x=fit_data.x, y=[yi[idx] for yi in fit_data.y])
            gp_params = get_gpflow_hypers_from_data(data_fit, print_fit_hypers=True)
            print(f'gp_params for output {idx} = {gp_params}')
        return

    if config.alg.rollout_sampling:
        current_obs = start_obs.copy() if config.fixed_start_obs else plan_env.reset()
        current_t = 0

    posterior_returns = None

    for i in range(config.num_iters):
        logging.info('---' * 5 + f' Start iteration i={i} ' + '---' * 5)
        logging.info(f'Length of data.x: {len(data.x)}')
        logging.info(f'Length of data.y: {len(data.y)}')

        # Initialize set of figures and axes
        #ax_all, fig_all = None, None
        #ax_postmean, fig_postmean = None, None
        #ax_samp, fig_samp = None, None
        #ax_obs, fig_obs = None, None
        #### ----
        ax_all, fig_all = plot_fn(path=None, domain=domain)
        ax_postmean, fig_postmean = plot_fn(path=None, domain=domain)
        ax_samp, fig_samp = plot_fn(path=None, domain=domain)
        ax_obs, fig_obs = plot_fn(path=None, domain=domain)

        # Set model
        model = gp_model_class(multi_gp_params, data)

        if config.alg.use_acquisition:
            # Set and optimize acquisition function
            acqfn_base = acqfn_class(acqfn_params, model, algo)
            acqfn = MCAcqFunction(acqfn_base, {"num_samples_mc": config.num_samples_mc})
            acqopt = AcqOptimizer()
            acqopt.initialize(acqfn)
            if config.alg.rollout_sampling:
                x_test = [np.concatenate([current_obs, env.action_space.sample()]) for _ in range(config.n_rand_acqopt)]
            elif config.sample_exe:
                all_x = []
                for path in acqfn.exe_path_full_list:
                    all_x += path.x
                x_test = random.sample(all_x, config.n_rand_acqopt)
            else:
                x_test = unif_random_sample_domain(domain, n=config.n_rand_acqopt)
            x_next, acq_val = acqopt.optimize(x_test)
            dumper.add('Acquisition Function Value', acq_val)

            # Plot true path and posterior path samples
            ax_all, fig_all = plot_fn(true_path, ax_all, fig_all, domain, 'true')
            if ax_all is not None:
                # Plot observations
                x_obs = [xi[0] for xi in data.x]
                y_obs = [xi[1] for xi in data.x]
                ax_all.scatter(x_obs, y_obs, color='grey', s=5, alpha=0.1)
                ax_obs.plot(x_obs, y_obs, 'o', color='k', ms=1)

                for path in acqfn.exe_path_list:
                    ax_all, fig_all = plot_fn(path, ax_all, fig_all, domain, 'samp')
                    ax_samp, fig_samp = plot_fn(path, ax_samp, fig_samp, domain, 'samp')

                # Plot x_next
                ax_all.scatter(x_next[0], x_next[1], facecolors='deeppink', edgecolors='k', s=120, zorder=100)
                ax_obs.plot(x_next[0], x_next[1], 'o', mfc='deeppink', mec='k', ms=12, zorder=100)
            posterior_returns = [compute_return(output[2], 1) for output in acqfn.output_list]
            dumper.add('Posterior Returns', posterior_returns)
        else:
            algo.initialize()

            policy = partial(algo.execute_mpc, f=make_postmean_fn(model))
            action = policy(current_obs)
            x_next = np.concatenate([current_obs, action])

        save_figure = False
        if i % config.eval_frequency == 0 or i + 1 == config.num_iters:
            if posterior_returns:
                logging.info(f"Current posterior returns: {posterior_returns}")
                logging.info(f"Current posterior returns: mean={np.mean(posterior_returns)}, std={np.std(posterior_returns)}")
            with Timer("Evaluate the current MPC policy"):
                # execute the best we can
                # this is required to delete the current execution path
                algo.initialize()

                postmean_fn = make_postmean_fn(model)
                policy = partial(algo.execute_mpc, f=postmean_fn)
                real_returns = []
                mses = []
                pbar = trange(config.num_eval_trials)
                for j in pbar:
                    real_obs, real_actions, real_rewards = evaluate_policy(env, policy, start_obs=start_obs,
                                                                           mpc_pass=True)
                    real_return = compute_return(real_rewards, 1)
                    real_returns.append(real_return)
                    real_path_mpc = Namespace()
                    real_path_mpc.x = real_obs
                    ax_all, fig_all = plot_fn(real_path_mpc, ax_all, fig_all, domain, 'postmean')
                    ax_postmean, fig_postmean = plot_fn(real_path_mpc, ax_postmean, fig_postmean, domain, 'samp')
                    mses.append(rollout_mse(algo.old_exe_paths[-1], f))
                    stats = {"Mean Return": np.mean(real_returns), "Std Return:": np.std(real_returns),
                             "Model MSE": np.mean(mses)}
                    pbar.set_postfix(stats)
                real_returns = np.array(real_returns)
                algo.old_exe_paths = []
                dumper.add('Eval Returns', real_returns)
                dumper.add('Eval ndata', len(data.x))
                logging.info(f"Eval Results: real_returns={real_returns}")
                logging.info(f"Eval Results: mean={np.mean(real_returns)}, std={np.std(real_returns)}")
                current_mpc_mse = np.mean(mses)
                test_y_hat = postmean_fn(test_data.x)
                random_mse = mse(test_data.y, test_y_hat)
                gt_mpc_y_hat = postmean_fn(test_mpc_data.x)
                gt_mpc_mse = mse(test_mpc_data.y, gt_mpc_y_hat)
                dumper.add('Model MSE (current MPC)', current_mpc_mse)
                dumper.add('Model MSE (random test set)', random_mse)
                dumper.add('Model MSE (GT MPC)', gt_mpc_mse)
                logging.info(f"Current MPC MSE: {current_mpc_mse:.3f}")
                logging.info(f"Random MSE: {random_mse:.3f}")
                logging.info(f"GT MPC MSE: {gt_mpc_mse:.3f}")

            save_figure = True
        if save_figure:
            neatplot.save_figure(str(dumper.expdir / f'mpc_all_{i}'), 'png', fig=fig_all)
            neatplot.save_figure(str(dumper.expdir / f'mpc_postmean_{i}'), 'png', fig=fig_postmean)
            neatplot.save_figure(str(dumper.expdir / f'mpc_samp_{i}'), 'png', fig=fig_samp)
            neatplot.save_figure(str(dumper.expdir / f'mpc_obs_{i}'), 'png', fig=fig_obs)
        dumper.save()

        # Query function, update data
        y_next = f([x_next])[0]
        data.x.append(x_next)
        data.y.append(y_next)
        if config.alg.rollout_sampling:
            current_t += 1
            if current_t > env.horizon:
                current_t = 0
                current_obs = start_obs.copy() if config.fixed_start_obs else plan_env.reset()
            else:
                current_obs += y_next[-obs_dim:]
        plt.close('all')


if __name__ == '__main__':
    main()
