import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from modpy.random import normal_pdf, UniformDist, JointDistribution
from modpy.design import kennard_stone_design
from modpy.optimize import mmo, nlprog, cma_es, bayesian_proposal, NonlinearConstraint
from modpy.optimize._bayesian import _expected_improvement, _probability_of_improvement, _upper_confidence_bound
from modpy.proxy import OrdinaryKrigingModel
from modpy.stats import hamiltonian_mc, posterior_ensemble
from modpy.plot.plot_util import subplot_layout, cm_parula, default_color, set_font_sizes
from modpy.illustration.illustration_util import CASE_PATH, test_function_2d


def _configure_axes_1D(ax, lim):
    ax.set_xlim(*lim)
    ax.set_ylim([-0.03, 0.15])
    ax.set_xlabel('$\\theta$')
    ax.set_ylabel('$f_{obj}(\\theta)$')
    ax.grid(True)
    ax.legend()


def _configure_axes_2D(ax):
    ax.set_xlim([-5., 5.])
    ax.set_ylim([-5., 5.])
    ax.set_xlabel('$\\theta_1$')
    ax.set_ylabel('$\\theta_2$')


def _plot_acquisition_functions():
    seed = 1234

    def obj(x):
        return normal_pdf(x, 15., 5.) + normal_pdf(x, 30., 4.) + normal_pdf(x, 40., 5.)

    # initial design of experiment
    x_doe = np.array([5., 18., 25., 44.])
    y_doe = obj(x_doe)

    # for illustration of underlying objective function
    x_true = np.linspace(0., 50., 500)
    y_true = obj(x_true)

    # set up optimizer -------------------------------------------------------------------------------------------------
    # set bounds
    x_min = 0.
    x_max = 50.

    # set up proxy model -----------------------------------------------------------------------------------------------
    proxy = OrdinaryKrigingModel(x_doe, y_doe, 'gaussian', seed=seed)

    theta0 = np.array([10.])
    proxy.initialize_ML(theta0)

    proxy.define_weights(x_true)
    mu = proxy.mean()
    sigma = proxy.std()

    # plot figure ------------------------------------------------------------------------------------------------------
    r, c = subplot_layout(4)
    fig, axes = plt.subplots(r, c, figsize=(20, 14))
    axes = axes.flatten()

    def plot_problem(ax):

        # plot true objective function
        ax.plot(x_true, y_true, color='gray', lw=2, label='True', zorder=2.5)
        ax.scatter(x_doe, y_doe, 20, color='k', label='DoE', zorder=5)
        ax.plot(x_true, mu, color='m', label='Proxy', zorder=3)

    # plot the problem
    ax = axes[0]
    plot_problem(ax)
    ax.plot(x_true, mu + 2. * sigma, color='m', ls='--', label='95% conf.', zorder=3)
    ax.plot(x_true, mu - 2. * sigma, color='m', ls='--', zorder=3)
    ax.set_title('Optimization Problem')
    _configure_axes_1D(ax, [x_min, x_max])
    set_font_sizes(ax, 13)

    # calculate acquisition functions
    kappa = 3.
    xi = 0.01
    y_max = np.amax(y_doe)

    a_ei = _expected_improvement(mu, sigma, y_max, xi)
    a_pi = _probability_of_improvement(mu, sigma, y_max, xi)
    a_ucb = _upper_confidence_bound(mu, sigma, kappa=kappa)

    # plot expected improvement
    ax = axes[1]
    plot_problem(ax)
    axt = ax.twinx()

    axt.plot(x_true, a_ei, color='b', label='Acquisition', zorder=4)
    _configure_axes_1D(ax, [x_min, x_max])
    ax.set_title('Expected Improvement')

    ax.legend(handles=[Line2D([], [], color='gray', label='True'),
                       Line2D([], [], color='m', label='Proxy'),
                       Line2D([], [], color='b', label='Bayesian'),
                       Line2D([], [], color='k', label='DoE', marker='o', ls='')],
              loc='upper left')

    axt.yaxis.set_visible(False)
    axt.yaxis.set_ticks([])
    set_font_sizes(ax, 13)

    # plot probability of improvement
    ax = axes[2]
    plot_problem(ax)
    axt = ax.twinx()
    axt.plot(x_true, a_pi, color='b', label='Acquisition', zorder=4)
    _configure_axes_1D(ax, [x_min, x_max])
    ax.set_title('Probability of Improvement')

    ax.legend(handles=[Line2D([], [], color='gray', label='True'),
                       Line2D([], [], color='m', label='Proxy'),
                       Line2D([], [], color='b', label='Bayesian'),
                       Line2D([], [], color='k', label='DoE', marker='o', ls='')],
              loc='upper left')

    axt.yaxis.set_visible(False)
    axt.yaxis.set_ticks([])
    set_font_sizes(ax, 13)

    # plot ubber confidence bounds
    ax = axes[3]
    plot_problem(ax)
    ax.plot(x_true, a_ucb, color='b', label='Acquisition', zorder=4)
    _configure_axes_1D(ax, [x_min, x_max])
    ax.set_title('Upper Confidence Bound')
    set_font_sizes(ax, 13)

    fig.savefig(CASE_PATH + 'acquisition_functions_1D.png')


def _plot_bayesian_sequential_1D():

    seed = 1234

    def obj(x):
        return normal_pdf(x, 15., 5.) + normal_pdf(x, 30., 4.) + normal_pdf(x, 40., 5.)

    # initial design of experiment
    x_doe = np.array([5., 18., 25., 44.])
    y_doe = obj(x_doe)

    # for illustration of underlying objective function
    x_true = np.linspace(0., 50., 500)
    y_true = obj(x_true)

    # set up optimizer -------------------------------------------------------------------------------------------------
    # set bounds
    x_min = 0.
    x_max = 50.
    bounds_ = ((x_min, x_max),)

    # set up optimization algorithm
    x0 = np.array([25.])

    def opt(obj_):
        return (cma_es(obj_, x0, bounds=bounds_, constraints=(), sigma0=10., lam=100, lbound=-1e-5, max_restart=10),)

    # set up proxy model -----------------------------------------------------------------------------------------------
    proxy = OrdinaryKrigingModel(x_doe, y_doe, 'gaussian', seed=seed)

    theta0 = np.array([10.])
    proxy.initialize_ML(theta0)

    # prepare figure ---------------------------------------------------------------------------------------------------
    loops = 4

    r, c = subplot_layout(loops + 2)
    fig, axes = plt.subplots(r, c, figsize=(20, 14))
    axes = axes.flatten()

    # plot true objective function
    ax = axes[0]
    ax.plot(x_true, y_true, color='gray', lw=2, label='True', zorder=2.5)
    ax.scatter(x_doe, y_doe, 20, color='k', label='Initial DoE', zorder=5)

    _configure_axes_1D(ax, [x_min, x_max])
    ax.set_title('Design of Experiment')

    # main optimization loop -------------------------------------------------------------------------------------------
    x_all = np.array(x_doe)
    y_all = np.array(y_doe)

    for i in range(loops):

        n_sim = y_all.size

        # prepare proxy model function
        def model(x):
            proxy.define_weights(x)
            return proxy.mean(), proxy.std()

        # find new simulation parameters based on bayesian proposal
        y_max = np.amax(y_all)
        xi = 0.01
        proposals = bayesian_proposal(model, opt, acq='EI', args=(y_max, xi))

        # create progress chart
        ax = axes[i + 1]
        axt = ax.twinx()

        # plot true objective function
        ax.plot(x_true, y_true, color='gray', lw=2, label='True', zorder=2.5)

        # plot proxy model and +- 2 sigma bounds
        proxy.define_weights(x_true)
        mu = proxy.mean()
        sigma = proxy.std()

        ax.plot(x_true, mu, color='m', label='Proxy', zorder=3)

        # plot objective function
        y_obj = _expected_improvement(mu, sigma, y_max, xi)
        axt.plot(x_true, y_obj, color='b', label='Bayesian', zorder=4)

        # plot all existing sample points
        ax.scatter(x_all, y_all, 20, color='k', label='Samples', zorder=5)

        if proposals:

            # extract proposal points and associated objective value
            x_pro = proposals[0].x
            y_pro_obj = proposals[0].f

            # simulate new proposals
            y_pro = obj(x_pro)

            # plot proposed samples
            axt.scatter(x_pro, y_pro_obj, 20, color='g', label='Proposal', zorder=6)

            # update proxy-model
            proxy.update(x_pro, y_pro)

            # add proposals to all samples
            x_all = np.append(x_all, x_pro)
            y_all = np.append(y_all, y_pro)

        # configure figure
        _configure_axes_1D(ax, [x_min, x_max])

        ax.legend(handles=[Line2D([], [], color='gray', label='True'),
                           Line2D([], [], color='m', label='Proxy'),
                           Line2D([], [], color='b', label='Bayesian'),
                           Line2D([], [], color='k', label='DoE', marker='o', ls='')],
                  loc='upper left')

        axt.yaxis.set_visible(False)
        axt.yaxis.set_ticks([])

        ax.set_title('Iteration {} (points: {}, proposals: {})'.format(i + 1, n_sim, len(proposals)))

    # prepare proxy-model for sampling
    def log_like(x):
        proxy.define_weights(x)
        return np.sum(np.log(proxy.mean()))

    par0 = np.array([(x_min + x_max) / 2.])
    mcmc = hamiltonian_mc(log_like, par0, 2000, df='3-point', burn=1000, bounds=bounds_, seed=seed)

    k = 10
    xp, fp = posterior_ensemble(mcmc, k, alpha=0.3)

    # plot sampling of proxy model
    ax = axes[-1]

    ax.plot(x_true, y_true, color='gray', lw=2, label='True', zorder=2.5)
    ax.scatter(xp, np.exp(fp), 20, color='g', label='Posterior Ensemble', zorder=6)

    _configure_axes_1D(ax, [x_min, x_max])
    ax.set_title('Posterior Samples')

    # save figure
    fig.savefig(CASE_PATH + 'SBO_1D.png')


def _plot_bayesian_1D():
    # example from: https://towardsdatascience.com/a-parallel-implementation-of-bayesian-optimization-2ffcdb2733a2

    dim = 1
    seed = 1234

    # constraint options
    use_constraints = True
    nl_r = 3.

    def obj(x):
        return normal_pdf(x, 15., 5.) + normal_pdf(x, 30., 4.) + normal_pdf(x, 40., 5.)

    # initial design of experiment
    x_doe = np.array([5., 18., 25., 44.])
    y_doe = obj(x_doe)

    # for illustration of underlying objective function
    x_true = np.linspace(0., 50., 500)
    y_true = obj(x_true)

    # set up optimizer -------------------------------------------------------------------------------------------------
    # set bounds
    x_min = 0.
    x_max = 50.
    bounds_ = ((x_min, x_max),)

    # define population sampler of MMO using prior probability assumption
    prior = UniformDist(x_min, x_max, seed=seed)

    def population(N):
        return prior.sample((N, dim))

    # set up underlying optimization algorithm of the MMO
    def opt_core(obj_, x0, bounds, constraints):
        return nlprog(obj_, x0, bounds=bounds, constraints=constraints)

    # set up proxy model -----------------------------------------------------------------------------------------------
    proxy = OrdinaryKrigingModel(x_doe, y_doe, 'gaussian', seed=seed)

    theta0 = np.array([10.])
    proxy.initialize_ML(theta0)

    # prepare figure ---------------------------------------------------------------------------------------------------
    loops = 4

    r, c = subplot_layout(loops + 2)
    fig, axes = plt.subplots(r, c, figsize=(20, 14))
    axes = axes.flatten()

    # plot true objective function
    ax = axes[0]
    ax.plot(x_true, y_true, color='gray', lw=2, label='True', zorder=2.5)
    ax.scatter(x_doe, y_doe, 20, color='k', label='Initial DoE', zorder=5)

    _configure_axes_1D(ax, [x_min, x_max])
    ax.set_title('Design of Experiment')

    # main optimization loop -------------------------------------------------------------------------------------------
    x_all = np.array(x_doe)
    y_all = np.array(y_doe)

    for i in range(loops):

        n_sim = y_all.size

        # constraint problem to reduce clustering
        if use_constraints:
            # update constraints to include all points
            def nl_con(x):
                return (x - x_all) ** 2.

            nl_lb = np.array([nl_r ** 2. for _ in y_all])
            nl_ub = np.array([np.inf for _ in y_all])

            constraints = NonlinearConstraint(nl_con, lb=nl_lb, ub=nl_ub)

        else:

            constraints = ()

        def opt(obj_):
            return mmo(obj_, opt_core, population, dim, bounds=bounds_, constraints=constraints, maxit=5).results

        # prepare proxy model function
        def model(x):
            proxy.define_weights(x)
            return proxy.mean(), proxy.std()

        # find new simulation parameters based on bayesian proposal
        # y_max = np.amax(y_all)
        kappa = 3.
        proposals = bayesian_proposal(model, opt, acq='UCB', thresh=0.5, args=(kappa,))

        # create progress chart
        ax = axes[i + 1]

        # plot true objective function
        ax.plot(x_true, y_true, color='gray', lw=2, label='True', zorder=2.5)

        # plot proxy model and +- 2 sigma bounds
        proxy.define_weights(x_true)
        mu = proxy.mean()
        sigma = proxy.std()

        ax.plot(x_true, mu, color='m', label='Proxy', zorder=3)
        ax.plot(x_true, mu + 2. * sigma, color='m', ls='--', label='95% conf.', zorder=3)
        ax.plot(x_true, mu - 2. * sigma, color='m', ls='--', zorder=3)

        # plot objective function
        y_obj = _upper_confidence_bound(mu, sigma, kappa=kappa)
        ax.plot(x_true, y_obj, color='b', label='Bayesian', zorder=4)

        # plot all existing sample points
        ax.scatter(x_all, y_all, 20, color='k', label='Samples', zorder=5)

        if proposals:

            # extract proposal points and associated objective value
            x_pro = np.block([res.x for res in proposals])
            y_pro_obj = np.block([res.f for res in proposals])

            # simulate new proposals
            y_pro = obj(x_pro)

            # plot proposed samples
            ax.scatter(x_pro, y_pro_obj, 20, color='g', label='Proposals', zorder=6)

            # update proxy-model
            proxy.update(x_pro, y_pro)

            # add proposals to all samples
            x_all = np.append(x_all, x_pro)
            y_all = np.append(y_all, y_pro)

        # configure figure
        _configure_axes_1D(ax, [x_min, x_max])
        ax.set_title('Iteration {} (points: {}, proposals: {})'.format(i + 1, n_sim, len(proposals)))

    # prepare proxy-model for sampling
    def log_like(x):
        proxy.define_weights(x)
        return np.sum(np.log(proxy.mean()))

    par0 = np.array([(x_min + x_max) / 2.])
    mcmc = hamiltonian_mc(log_like, par0, 2000, df='3-point', burn=1000, bounds=bounds_, seed=seed)

    k = 10
    xp, fp = posterior_ensemble(mcmc, k, alpha=0.3)

    # plot sampling of proxy model
    ax = axes[-1]

    ax.plot(x_true, y_true, color='gray', lw=2, label='True', zorder=2.5)
    ax.scatter(xp, np.exp(fp), 20, color='g', label='Posterior Ensemble', zorder=6)

    _configure_axes_1D(ax, [x_min, x_max])
    ax.set_title('Posterior Samples')

    con = '_con' if use_constraints else ''

    # save figure
    fig.savefig(CASE_PATH + 'PBO_1D{}.png'.format(con))


def _plot_bayesian_2D():

    dim = 2
    seed = 1234

    m = 100
    x1 = np.linspace(-5., 5., m)
    x2 = np.linspace(-5., 5., m)
    X1, X2 = np.meshgrid(x1, x2)
    P = [X1.flatten(), X2.flatten()]
    S = np.vstack((X1.flatten(), X2.flatten())).T

    s_ = test_function_2d(P)
    a = np.amin(s_)
    b = np.amax(s_)

    def true_obj(p):
        s = test_function_2d(p)
        return np.clip(-(s - a) / (b - a) + 1., 0., 1.)

    # calculate underlying objective function
    y_true = true_obj(P)
    y_true = np.reshape(y_true, (m, m))

    v_min = 0.75
    v_max = 1.
    cmap = cm_parula
    levels = np.block([np.arange(0., v_min, 0.05), np.arange(v_min, v_max, 0.001), v_max])

    # calculate experimental design
    prior = JointDistribution((UniformDist(-4., 4., seed=seed), UniformDist(-4., 4., seed=seed)))
    x_doe = kennard_stone_design(prior, 15, 100, scale=True, seed=seed)
    y_doe = true_obj(x_doe.T)

    # set up optimizer -------------------------------------------------------------------------------------------------
    # set bounds
    x_min = -5.
    x_max = 5.
    bounds_ = ((x_min, x_max), (x_min, x_max))

    # define population sampler of MMO using prior probability assumption
    def population(N):
        return prior.sample(N)

    # set up underlying optimization algorithm of the MMO
    def opt_core(obj_, x0, bounds, constraints):
        return nlprog(obj_, x0, bounds=bounds, constraints=constraints, tol=1e-3)

    def opt(obj_):
        return mmo(obj_, opt_core, population, dim, bounds=bounds_, maxit=5).results

    # set up proxy model -----------------------------------------------------------------------------------------------
    proxy = OrdinaryKrigingModel(x_doe, y_doe, 'gaussian', seed=seed)

    theta0 = np.array([3., 3.])
    #proxy.initialize(np.array([0.8]), theta0, 0.3)
    proxy.initialize_ML(theta0)  # , bounds=((1., 5.), (1., 5.))

    # prepare figures --------------------------------------------------------------------------------------------------
    loops = 4

    r, c = subplot_layout(loops + 2)
    fig_proxy, axes_proxy = plt.subplots(r, c, figsize=(20, 14))
    axes_proxy = axes_proxy.flatten()

    r, c = subplot_layout(loops + 1)
    fig_diff, axes_diff = plt.subplots(r, c, figsize=(20, 14))
    axes_diff = axes_diff.flatten()

    r, c = subplot_layout(loops)
    fig_obj, axes_obj = plt.subplots(r, c, figsize=(20, 14))
    axes_obj = axes_obj.flatten()

    # plot true objective function
    ax_proxy = axes_proxy[0]
    ax_proxy.contourf(X1, X2, y_true, levels=levels, cmap=cmap, vmin=v_min, vmax=v_max, extend='both')
    ax_proxy.scatter(x_doe[:, 0], x_doe[:, 1], 20, color='r', zorder=5)

    # plot difference
    ax_diff = axes_diff[0]

    proxy.define_weights(S)
    y_proxy = np.reshape(proxy.mean(), (m, m))

    ax_diff.contourf(X1, X2, y_true - y_proxy, 300, cmap=mpl.cm.bwr)
    ax_diff.scatter(x_doe[:, 0], x_doe[:, 1], 20, color='k', zorder=5)

    _configure_axes_2D(ax_proxy)
    _configure_axes_2D(ax_diff)
    ax_proxy.set_title('Design of Experiment')
    ax_diff.set_title('Design of Experiment')

    # main optimization loop -------------------------------------------------------------------------------------------
    x_all = np.array(x_doe)
    y_all = np.array(y_doe)

    for i in range(loops):

        n_sim = y_all.size

        # prepare proxy model function
        def model(x):
            proxy.define_weights(np.atleast_2d(x))
            return proxy.mean(), proxy.std()

        # find new simulation parameters based on bayesian proposal
        # y_max = np.amax(y_all)
        kappa = 3.
        proposals = bayesian_proposal(model, opt, acq='UCB', thresh=0.5, args=(kappa,))

        # create proxy chart
        ax_proxy = axes_proxy[i + 1]

        proxy.define_weights(S)
        mu = proxy.mean()
        sigma = proxy.std()

        y_proxy = np.reshape(mu, (m, m))
        ax_proxy.contourf(X1, X2, y_proxy, levels, cmap=cmap, vmin=v_min, vmax=v_max, extend='both')
        ax_proxy.scatter(x_all[:, 0], x_all[:, 1], 20, color='r', zorder=5)

        # create difference chart
        ax_diff = axes_diff[i + 1]
        ax_diff.contourf(X1, X2, y_true - y_proxy, 300, cmap=mpl.cm.bwr)
        ax_diff.scatter(x_all[:, 0], x_all[:, 1], 20, color='k', zorder=5)

        # create objective chart
        ax_obj = axes_obj[i]
        y_obj = _upper_confidence_bound(mu, sigma, kappa=kappa)
        ax_obj.contourf(X1, X2, np.reshape(y_obj, (m, m)), 300, cmap=cmap)

        # add proposed simulations
        if proposals:
            # extract proposal points and associated objective value
            x_pro = np.vstack([res.x for res in proposals])

            # simulate new proposals
            y_pro = true_obj(x_pro.T)

            # plot proposed samples
            ax_obj.scatter(x_pro[:, 0], x_pro[:, 1], 20, color='m', zorder=6)

            # update proxy-model
            proxy.update(x_pro, y_pro)

            # add proposals to all samples
            x_all = np.vstack((x_all, x_pro))
            y_all = np.append(y_all, y_pro)

        # configure figure
        _configure_axes_2D(ax_proxy)
        _configure_axes_2D(ax_diff)
        _configure_axes_2D(ax_obj)

        ax_proxy.set_title('Iteration {} (points: {}, proposals: {})'.format(i + 1, n_sim, len(proposals)))
        ax_diff.set_title('Iteration {} (points: {}, proposals: {})'.format(i + 1, n_sim, len(proposals)))
        ax_obj.set_title('Iteration {} (points: {}, proposals: {})'.format(i + 1, n_sim, len(proposals)))

        print('Iteration {}'.format(i))

    # prepare proxy-model for sampling
    def log_like(x):
        proxy.define_weights(np.atleast_2d(x))
        return np.sum(np.log(proxy.mean()))

    par0 = np.array([0., 3.])
    mcmc = hamiltonian_mc(log_like, par0, 2000, df='3-point', burn=500, bounds=bounds_, seed=seed)

    k = 30
    xp, _ = posterior_ensemble(mcmc, k, alpha=0.75, thresh=0.75)

    # plot sampling of proxy model
    ax_proxy = axes_proxy[-1]

    cont = ax_proxy.contourf(X1, X2, y_true, levels=levels, cmap=cmap, vmin=v_min, vmax=v_max, extend='both')
    ax_proxy.scatter(xp[:, 0], xp[:, 1], 20, color='g', zorder=6)

    _configure_axes_2D(ax_proxy)
    ax_proxy.set_title('Posterior Samples')

    # save figure
    fig_proxy.savefig(CASE_PATH + 'PBO_proxy_2D.png')
    fig_diff.savefig(CASE_PATH + 'PBO_diff_2D.png')
    fig_obj.savefig(CASE_PATH + 'PBO_obj_2D.png')


if __name__ == '__main__':
     _plot_acquisition_functions()
    #_plot_bayesian_sequential_1D()
    #_plot_bayesian_1D()
    #_plot_bayesian_2D()
