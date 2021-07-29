import numpy as np
import numpy.linalg as la
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from modpy.optimize._constraints import prepare_bounds
from modpy.plot.plot_util import subplot_layout, default_color, cm_parula, hollow_circle
from modpy.illustration.illustration_util import OPTIMIZE_PATH
from modpy.optimize.benchmarks.benchmark_suite import G06, G08, G11, TR2


# ======================================================================================================================
# CONVERGENCE PLOTS
# ======================================================================================================================
def plot_convergence_single_algorithm(algorithm, results, constrained=False, bounded=False):
    """
    Generate and save convergence performance plots for a single benchmark test suite.

    Parameters
    ----------
    algorithm : Algorithm
        Class Algorithm.
    results : tuple
        Tuple of tuples (test, res).
    constrained : bool
        Tests included constraints.
    bounded : bool
        Tests included bounds.
    """

    # solution tolerance and function tolerance
    n = 3

    # equality and inequality constrained tolerances
    if constrained:
        n += 2

    # additional parameters
    if algorithm.cma:
        n += 1

    elif algorithm.barrier:
        n += 1

    # create subplots
    r, c = subplot_layout(n)
    fig, axes = plt.subplots(r, c, sharex='all', figsize=(15, 10))
    axes = axes.flatten()

    # create plots
    for i, (test, res) in enumerate(results):

        if not res.success:
            continue

        _plot_x_error(axes[0], res, test, i)
        _plot_f_error(axes[1], res, test, i)
        _plot_ftol_error(axes[2], res, test, i)

        idx = 3

        if constrained:
            _plot_etol_error(axes[idx], res, test, i)
            _plot_itol_error(axes[idx+1], res, test, i)
            idx += 2

        if algorithm.cma:
            _plot_sigma(axes[idx], res, test, i)

        elif algorithm.barrier:
            _plot_mu(axes[idx], res, test, i)

    _format_axes(axes, xlabel='Iteration')
    _format_y_axis(axes, results, algorithm, constrained)

    if constrained:
        idf = ' (con)'
    elif bounded:
        idf = ' (bounds)'
    else:
        idf = ''

    fig.suptitle(algorithm.name + idf)
    fig.savefig(OPTIMIZE_PATH + 'optimization_benchmark_{}{}.png'.format(algorithm.name, idf))
    plt.close(fig)


def _plot_x_error(ax, res, test, i):
    if res.path.xi and (test.x_opt is not None):
        xnorm = la.norm(np.array(res.path.xi) - test.found_optimum(res), axis=1)
        ax.semilogy(np.arange(xnorm.size), xnorm, c=default_color(i), label=test.name)


def _plot_f_error(ax, res, test, i):
    if res.path.fi and (test.f_opt is not None):
        fi = np.array(res.path.fi)
        fd = np.abs(np.array(fi - test.found_optimum_value(res)))
        ax.semilogy(np.arange(fd.size), fd, c=default_color(i), label=test.name)


def _plot_ftol_error(ax, res, test, i):
    if res.path.ftol:
        ftol = np.array(res.path.ftol)
        ax.semilogy(np.arange(ftol.size), ftol, c=default_color(i), label=test.name)


def _plot_etol_error(ax, res, test, i):
    etol = np.array(res.path.etol)
    if etol.size and (np.all(etol > 0.)):
        ax.semilogy(np.arange(etol.size), etol, c=default_color(i), label=test.name)


def _plot_itol_error(ax, res, test, i):
    itol = np.array(res.path.itol)
    if itol.size and (np.all(itol > 0)):
        ax.semilogy(np.arange(itol.size), itol, c=default_color(i), label=test.name)


def _plot_sigma(ax, res, test, i):
    sigma = np.array(res.path.sigma)
    if sigma.size and (np.all(sigma > 0)):
        ax.semilogy(np.arange(sigma.size), sigma, c=default_color(i), label=test.name)


def _plot_mu(ax, res, test, i):
    mu = np.array(res.path.mu)
    if mu.size and (np.all(mu > 0)):
        ax.semilogy(np.arange(mu.size), mu, c=default_color(i), label=test.name)


def _format_axes(axes, xlabel=''):
    for ax in axes:
        ax.grid(True)
        ax.legend()
        ax.set_xlabel(xlabel)


def _format_y_axis(axes, results, algorithm, constrained):

    _, res = results[0]

    # solution tolerance and function tolerance
    n = 3

    # equality and inequality constrained tolerances
    if constrained:
        n += 2

    axes[0].set_ylabel('$||x-x^*||_2$')
    axes[1].set_ylabel('$||f-f^*||_1$')
    axes[2].set_ylabel('$||dL||_{\infty}$')

    idx = 3
    if constrained:
        axes[3].set_ylabel('$||Ax-b||_{\infty}$')
        axes[4].set_ylabel('$||Cx-d||_{\infty}$')

        idx += 2

    if algorithm.cma:
        axes[idx].set_ylabel('$\sigma^{CMA}$')

        idx += 1

    if algorithm.barrier:
        axes[idx].set_ylabel('$\mu$')


# ======================================================================================================================
# PERFORMANCE PLOTS
# ======================================================================================================================
def plot_performance_comparison(algorithms, tests, results, dim, fig_name='unconstrained'):
    """
    Generate and save convergence performance plots for a single benchmark test suite.

    Parameters
    ----------
    algorithms : tuple
        Tuple of class Algorithm.
    tests : tuple
        Tuple of tests the algorithms were run on.
    results : dict
        Dict of same length as `algorithms` with an inner tuple:
        {algorithm.name: (duration, converged, iterations, status)}.
    dim : int
        Dimension of the tests
    fig_name : str
        Name for the type of tests, such as 'unconstrained', 'bounded', 'constrained', ...
    """

    n = len(algorithms)
    m = len(tests)

    ind = np.arange(m)
    w = 1. / (n + 1)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    for i, algorithm in enumerate(algorithms):
        duration, converged, iterations, status = results[algorithm.name]

        axes[0].bar(ind + i * w, duration, width=w, fc=default_color(i), ec='k', label=algorithm.name)
        axes[1].bar(ind + i * w, converged, width=w, fc=default_color(i), ec='k', label=algorithm.name)
        axes[2].bar(ind + i * w, iterations, width=w, fc=default_color(i), ec='k', label=algorithm.name)
        axes[3].bar(ind + i * w, status, width=w, fc=default_color(i), ec='k', label=algorithm.name)

    # set options
    titles = ['Duration', 'Convergence', 'Iterations', 'Status']
    y_labels = ['Average duration [s]', 'Fraction of converged runs',
                'Average number of iterations', 'Most frequent failure status']
    x_labels = [test.name for test in tests]

    for i, ax in enumerate(axes):
        ax.legend()
        ax.set_title(titles[i])
        ax.set_ylabel(y_labels[i])
        ax.set_xticks(ind + w)
        ax.set_xticklabels(x_labels, rotation=15)
        ax.set_ylim([0, None])

    # set time and iterations to logarithmic
    axes[0].set_yscale('log')
    axes[2].set_yscale('log')

    fig.suptitle(fig_name)
    fig.savefig(OPTIMIZE_PATH + 'performance_dim({})_{}.png'.format(dim, fig_name))
    plt.close(fig)



# ======================================================================================================================
# MMO CONVERGENCE PLOTS
# ======================================================================================================================
def plot_convergence_mmo(algorithms, tracks_mmo, suite, maxN, global_only=True):
    """
    Generate and save convergence performance plots for a single benchmark test suite.

    Parameters
    ----------
    algorithms : tuple
        Tuple of class Algorithm.
    tracks_mmo : list
        List of tuples of array_like [[x, y],].
    suite : tuple
        Tuple of TestFunction.
    global_only : bool
        Whether tests included both local and global optimums
    """

    # create subplots
    r, c = subplot_layout(len(suite))
    fig, axes = plt.subplots(r, c, sharex='all', sharey='all', figsize=(15, 10))

    if r * c > 1:
        axes = axes.flatten()
    else:
        axes = (axes,)

    # create plots
    # one res per algorithm
    for i, (algorithm, tracks) in enumerate(zip(algorithms, tracks_mmo)):

        for j, (test, ax) in enumerate(zip(suite, axes)):

            track = tracks[j]

            x, y = track[:, 0], track[:, 1]
            ax.semilogx(x, y, c=default_color(i), label=algorithm.name)

    # configure
    for ax, test in zip(axes, suite):
        ax.set_xlim([1, maxN])
        ax.set_ylim([0., 1.])
        ax.set_xlabel('Sample size')
        ax.set_ylabel('Peak Ratio ({} peaks)'.format(test.n_gopt))
        ax.set_title(test.name + ' ({})'.format(test.dim))
        ax.legend()
        ax.grid(True)

    fig.savefig(OPTIMIZE_PATH + 'MMO_benchmark.png')
    plt.close(fig)


# ======================================================================================================================
# 2D PLOTS
# ======================================================================================================================
def plot_surfaces_2D(tests, identifier=''):
    """
    Generate and save plots of the path an algorithm takes. Assumes dim=2.

    Parameters
    ----------
    tests : tuple
        Tuple of class TestFunction
    identifier : str
        Identifier of the test, added to title and path.
    """

    if not len(tests):
        return

    available = []

    for test in tests:
        if test.dim == 2:
            available.append(test)

    n = len(available)

    if not n:
        return

    # create subplots
    r, c = subplot_layout(n)
    fig = plt.figure(figsize=(20, 11))
    fig.subplots_adjust(wspace=0.25, hspace=0.25)

    # create plots
    for i, test in enumerate(tests):

        ax = fig.add_subplot(r, c, i + 1, projection='3d')
        _plot_2D_surface(ax, test)

    fig.suptitle('Test Functions ({})'.format(identifier))
    fig.savefig(OPTIMIZE_PATH + 'test_functions_2D_{}.png'.format(identifier))
    plt.close(fig)


def plot_unconstrained_2D(algorithm, results, bounded=False, identifier=''):
    """
    Generate and save plots of the path an algorithm takes. Assumes dim=2.

    Parameters
    ----------
    algorithm : Algorithm
        Class Algorithm.
    results : tuple
        Tuple of tuples (test, res).
    bounded : bool
        If the problem has bounds.
    identifier : str
        Identifier of the test, added to title and path.
    """

    if not len(results):
        return

    available = []

    for (test, res) in results:
        if test.dim == 2:
            available.append((test, res))

    n = len(available)

    if not n:
        return

    # create subplots
    r, c = subplot_layout(n)
    fig, axes = plt.subplots(r, c, figsize=(20, 11))
    fig.subplots_adjust(wspace=0.25, hspace=0.25)
    axes = axes.flatten()

    # create plots
    for i, ((test, res), ax) in enumerate(zip(available, axes)):

        _plot_2D_contour(ax, res, test)

    fig.suptitle(algorithm.name + ' ({})'.format(identifier))
    fig.savefig(OPTIMIZE_PATH + '{}_2D_{}.png'.format(algorithm.name, identifier))
    plt.close(fig)


def plot_constrained_2D(algorithm, results):
    """
    Generate and save plots of the path an algorithm takes. Assumes dim=2.

    Parameters
    ----------
    algorithm : Algorithm
        Class Algorithm.
    results : tuple
        Tuple of tuples (test, res).

    """

    if not len(results):
        return

    available = []

    for (test, res) in results:
        if test.name in ('G06', 'G08', 'G11', 'TR2'):
            available.append((test, res))

    n = len(available)

    if not n:
        return

    # create subplots
    r, c = subplot_layout(n)
    fig, axes = plt.subplots(r, c, figsize=(20, 11))
    fig.subplots_adjust(wspace=0.2, hspace=0.2)
    axes = axes.flatten()

    # create plots
    for i, ((test, res), ax) in enumerate(zip(available, axes)):

        if test.name == 'G06':

            plot_G06(ax, res)

        elif test.name == 'G08':

            plot_G08(ax, res)

        elif test.name == 'G11':

            plot_G11(ax, res)

        elif test.name == 'TR2':

            plot_TR2(ax, res)

    fig.suptitle(algorithm.name)
    fig.savefig(OPTIMIZE_PATH + '{}_2D_constrained.png'.format(algorithm.name))
    plt.close(fig)


def plot_mixed_dimensions(tests, identifier=''):
    """
    Generate and save plots of the path an algorithm takes. Assumes dim=2.

    Parameters
    ----------
    tests : tuple
        Tuple of class TestFunction
    identifier : str
        Identifier of the test, added to title and path.
    """

    if not len(tests):
        return

    available = []

    for test in tests:
        if test.dim <= 2:
            available.append(test)

    n = len(available)

    if not n:
        return

    # create subplots
    r, c = subplot_layout(n)
    fig = plt.figure(figsize=(20, 11))
    fig.subplots_adjust(wspace=0.25, hspace=0.25)

    # create plots
    for i, test in enumerate(tests):

        if test.dim == 1:

            ax = fig.add_subplot(r, c, i + 1)
            _plot_1D_line(ax, test)

        elif test.dim == 2:

            ax = fig.add_subplot(r, c, i + 1, projection='3d')
            _plot_2D_surface(ax, test)

    fig.suptitle('Test Functions ({})'.format(identifier))
    fig.savefig(OPTIMIZE_PATH + 'test_functions_2D_{}.png'.format(identifier))
    plt.close(fig)


# ======================================================================================================================
# 2D CONSTRAINED PLOTS
# ======================================================================================================================
def plot_G06(ax, res):
    test = G06()
    levels = 800

    _plot_2D_contour(ax, res, test, levels)

    # plot constrained area
    con1 = plt.Circle((5., 5.), np.sqrt(100.), edgecolor='k', facecolor=np.array([.5, .5, .5]), alpha=0.5)
    con2 = hollow_circle((6., 5.), np.sqrt(82.81), (6., 5.), 160,
                         kwargs={'edgecolor': 'k', 'facecolor': np.array([.5, .5, .5]), 'alpha': 0.5})

    ax.add_artist(con1)
    ax.add_artist(con2)

    ax.legend()


def plot_G08(ax, res):
    test = G08()
    levels = 500

    x1, x2, limits = _plot_2D_contour(ax, res, test, levels)

    # plot constrained area
    lb1, ub1 = limits[0]
    lb2, ub2 = limits[1]
    ax.fill_between(x1,  x1 ** 2. + 1., lb2, alpha=.5, edgecolor='k', facecolor=np.array([.5, .5, .5]))
    ax.fill_betweenx(x2, lb1, x2 ** 2. - 8. * x2 + 17, alpha=.5, edgecolor='k', facecolor=np.array([.5, .5, .5]))

    ax.legend()


def plot_G11(ax, res):
    test = G11()
    levels = 500

    x1, _, limits = _plot_2D_contour(ax, res, test, levels)

    # plot constraints
    lb, ub = limits[0]
    ax.plot(x1, x1 ** 2., c='k', label='Eq. Con.')

    ax.legend()


def plot_TR2(ax, res):
    test = TR2()
    levels = 500

    x1, _, limits = _plot_2D_contour(ax, res, test, levels)

    # plot constraints
    ax.fill_between(x1, 2. - x1, -5., alpha=.5, edgecolor='k', facecolor=np.array([.5, .5, .5]))

    ax.legend()


# ======================================================================================================================
# 3D PLOTS
# ======================================================================================================================
def plot_unconstrained_3D(algorithm, results, bounded=False):
    """
    Generate and save plots of the path an algorithm takes. Assumes dim=3.

    Parameters
    ----------
    algorithm : Algorithm
        Class Algorithm.
    results : tuple
        Tuple of tuples (test, res).
    bounded : bool
        If the problem has bounds.

    """

    if not len(results):
        return

    available = []

    for (test, res) in results:
        if test.dim == 3:
            available.append((test, res))

    n = len(available)

    if not n:
        return

    # create subplots
    r, c = subplot_layout(n)
    fig = plt.figure(figsize=(20, 11))
    fig.subplots_adjust(wspace=0.1, hspace=0.1)

    # create plots
    for i, (test, res) in enumerate(available):

        ax = fig.add_subplot(r, c, i+1, projection='3d')
        _plot_3D_problem(ax, res, test)

    save_name = 'bounded' if bounded else 'unconstrained'

    fig.suptitle(algorithm.name)
    fig.savefig(OPTIMIZE_PATH + '{}_3D_{}.png'.format(algorithm.name, save_name))
    plt.close(fig)


# ======================================================================================================================
# AUXILIARY METHODS
# ======================================================================================================================
def _plot_1D_line(ax, test):

    # get grid/plot limits
    limits = test.bounds[0]

    # create 2D grid

    m = 1000
    x = np.linspace(*limits, m)

    # plot underlying objective function
    obj = np.array([test.obj(x_) for x_ in x])
    ax.plot(x, obj)

    # set axis limits
    ax.set_xlim(*limits)
    #ax.set_ylim([0., None])

    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')

    ax.grid(True)
    ax.set_title(test.name)


def _plot_2D_contour(ax, res, test, levels=500, vmin=None, vmax=None):

    # plot white color outside min/max of contour range
    cmap = cm_parula
    cmap.set_under('w')
    cmap.set_over('w')

    # get grid/plot limits
    limits = _get_grid_limits(res, test)
    limits = _get_axis_limits(limits)

    # create 2D grid
    m = 100
    m2 = m ** 2
    x1_ = np.linspace(*limits[0], m)
    x2_ = np.linspace(*limits[1], m)
    X1, X2 = np.meshgrid(x1_, x2_)
    P = np.vstack((X1.flatten(), X2.flatten())).T

    # plot underlying objective function
    obj = np.zeros((m2,))
    for i in range(m2):
        obj[i] = test.obj(P[i, :])

    cont = ax.contourf(X1, X2, np.reshape(obj, (m, m)), levels, cmap=cmap, vmin=vmin, vmax=vmax)

    # plot bounds
    _plot_bounds(ax, test, limits)

    # plot the optimization path
    _plot_path_2D(ax, res, test)

    # set axis limits
    ax.set_xlim(*limits[0])
    ax.set_ylim(*limits[1])

    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')

    ax.legend()
    ax.set_title(test.name)

    return x1_, x2_, limits


def _plot_2D_surface(ax, test, levels=500, vmin=None, vmax=None):

    # plot white color outside min/max of contour range
    cmap = cm_parula
    #cmap.set_under('w')
    #cmap.set_over('w')

    # get grid/plot limits
    limits = test.bounds

    if limits is None:
        limits = ((-5., 5.), (-5., 5.))

    # create 2D grid
    m = 1000
    m2 = m ** 2
    x1_ = np.linspace(*limits[0], m)
    x2_ = np.linspace(*limits[1], m)
    X1, X2 = np.meshgrid(x1_, x2_)
    S = [X1.flatten(), X2.flatten()]
    P = np.vstack((X1.flatten(), X2.flatten())).T

    # plot underlying objective function
    try:
        obj = test.obj(S)
    except:
        obj = np.zeros((m2,))
        for i in range(m2):
            obj[i] = test.obj(P[i, :])

    surf = ax.plot_surface(X1, X2, np.reshape(obj, (m, m)), cmap=cmap, antialiased=False)  # , linewidth=0.1

    # set axis limits
    ax.set_xlim(*limits[0])
    ax.set_ylim(*limits[1])

    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')

    #ax.legend()
    ax.set_title(test.name)

    return x1_, x2_, limits


def _plot_3D_problem(ax, res, test):

    # plot white color outside min/max of contour range
    cmap = cm_parula
    cmap.set_under('w')
    cmap.set_over('w')

    # get grid/plot limits
    limits = _get_grid_limits(res, test)
    limits = _get_axis_limits(limits)

    # plot the optimization path
    _plot_path_3D(ax, res, test)

    # set axis limits
    ax.set_xlim(*limits[0])
    ax.set_ylim(*limits[1])

    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('$x_3$')

    ax.legend()
    ax.set_title(test.name)


def _get_grid_limits(res, test):

    # bounds
    if test.bounds is not None:
        lb, ub = prepare_bounds(test.bounds, test.dim)
    else:
        lb = None
        ub = None

    # start-guess
    if test.x_ini is not None:
        x0 = test.x_ini
    elif res.path and res.path.xi:
        x0 = res.path.xi[0]
    else:
        x0 = None

    # optimum
    if test.x_opt is not None:

        if isinstance(test.x_opt, tuple):  # multiple optimums
            x_opt = test.x_opt

        else:  # single optimum
            x_opt = (test.x_opt,)

    elif res.success:
        x_opt = (res.x,)
    else:
        x_opt = None

    if (lb is None) and (ub is None) and (x0 is None) and (x_opt is None):
        return (-10., 10.), (-10., 10.)

    limits = []
    for i in range(test.dim):
        vec = []

        if (lb is not None) and not np.isinf(lb[i]):
            vec.append(lb[i])

        if (ub is not None) and not np.isinf(ub[i]):
            vec.append(ub[i])

        if x0 is not None:
            vec.append(x0[i])

        if x_opt is not None:
            for x in x_opt:
                vec.append(x[i])

        vec = np.array(vec)

        if res.success:
            if res.path.xi:
                path = np.array(res.path.xi)
                vec = np.append(vec, path[:, :])

            if res.path.candidates is not None:
                candidates = np.atleast_2d(res.path.candidates)
                vec = np.append(vec, candidates[:, i])

        min_ = np.amin(vec)
        max_ = np.amax(vec)

        limits.append((min_, max_))

    return tuple(limits)


def _get_axis_limits(limits):
    axis_lim = []
    for lim in limits:
        lb, ub = lim

        if lb == 0.:
            axis_lb = -1.
        else:
            if lb < 0.:
                axis_lb = lb * 1.1
            else:
                axis_lb = lb * 0.9

        if ub == 0.:
            axis_ub = 1.
        else:
            if ub < 0.:
                axis_ub = ub * 0.9
            else:
                axis_ub = ub * 1.1

        axis_lim.append((axis_lb, axis_ub))

    return tuple(axis_lim)


def _plot_path_2D(ax, res, test):

    if res.path:
        # plot optimization path
        xi = np.array(res.path.xi)
        ax.plot(xi[:, 0], xi[:, 1], zorder=1, marker='o', c='b', lw=1, label='Steps')

        # add start-guess
        ax.scatter(xi[0, 0], xi[0, 1], 30, zorder=2, c='r', label='$x_0$')

        # if population based algorithm plot sampled candidates
        if res.path.candidates is not None:
            _plot_candidates_2D(ax, res.path.candidates)

    # add true optimum
    if test.x_opt is not None:

        if isinstance(test.x_opt, tuple):  # multiple optimums
            x_opt = test.x_opt

        else:  # single optimum
            x_opt = (test.x_opt,)

        for i, x in enumerate(x_opt):
            if i == 0:
                label = 'True $x_{opt}$'
            else:
                label = None

            ax.scatter(x[0], x[1], 40, zorder=2, c='m', label=label)

    # add found optimum
    if res.success:
        ax.scatter(res.x[0], res.x[1], 30, zorder=2, c='g', label='Found $x_{opt}$')


def _plot_path_3D(ax, res, test):

    if res.path:
        # plot optimization path
        xi = np.array(res.path.xi)
        ax.plot3D(xi[:, 0], xi[:, 1], xi[:, 2], marker='o', c='b', lw=1, label='Steps')

        # add start-guess
        ax.scatter3D(xi[0, 0], xi[0, 1], xi[0, 2], s=30, c='r', label='$x_0$')

        # if population based algorithm plot sampled candidates
        #if res.path.candidates is not None:
        #    _plot_candidates_3D(ax, res.path.candidates)

    # add true optimum
    if test.x_opt.size:
        ax.scatter3D(test.x_opt[0], test.x_opt[1], test.x_opt[2], s=40, c='m', label='True $x_{opt}$')

    # add found optimum
    if res.success:
        ax.scatter3D(res.x[0], res.x[1], res.x[2], s=50, c='g', label='Found $x_{opt}$')


def _plot_candidates_2D(ax, cand, color='k'):

    if cand.ndim == 2:  # (mu/lam)-CMA-ES
        ax.scatter(cand[:, 0], cand[:, 1], 20, zorder=1, c=color, label='ES cand.')

    else:  # (1+1)-CMA-ES
        ax.scatter(cand[:, 0], cand[:, 1], 20, zorder=1, c=color, label='ES cand.')


def _plot_candidates_3D(ax, cand, color='k'):
    if not cand.size:
        return

    if cand.ndim == 2:  # (mu/lam)-CMA-ES

        ax.scatter3D(cand[:, 0], cand[:, 1], cand[:, 2], c=color, label='ES cand.')

    else:  # (1+1)-CMA-ES
        ax.scatter3D(cand[:, 0], cand[:, 1], cand[:, 2], c=color, label='ES cand.')


def _plot_bounds(ax, test, axis_limits):

    if test.bounds is None:
        return

    # bounds color (# last value in facecolor is alpha)
    kwargs = {'edgecolor': 'k',
              'facecolor': np.array([.5, .5, .5, 0.8]),
              'lw': .3}

    alb1, aub1 = axis_limits[0]
    alb2, aub2 = axis_limits[1]
    lb1, ub1 = test.bounds[0]
    lb2, ub2 = test.bounds[1]

    ax.fill_between([alb1, aub1], alb2, lb2, **kwargs, label='Bounds')
    ax.fill_between([alb1, aub1], ub2, aub2, **kwargs)

    ax.fill_betweenx([alb2, aub2], alb1, lb1, **kwargs)
    ax.fill_betweenx([alb2, aub2], ub1, aub1, **kwargs)
