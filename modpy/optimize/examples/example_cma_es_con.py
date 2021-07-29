import numpy as np
import matplotlib.pyplot as plt

from modpy.optimize import cma_es, LinearConstraint, NonlinearConstraint
from modpy.optimize._constraints import _prepare_constraints
from modpy.plot.plot_util import cm_parula, subplot_layout
from modpy.illustration.illustration_util import OPTIMIZE_PATH, test_function_2d
from modpy.optimize._line_search import MeritNormal

from modpy.optimize.benchmarks.benchmark_suite import G06, G08, G11, TR2
from modpy.optimize.benchmarks.benchmark_plots import plot_G06, plot_G08, plot_G11, plot_TR2

from modpy.optimize._cma_es import CMAES


def plot_cma_es_con():
    m = 100
    x_ = np.linspace(-5., 5., m)
    y_ = np.linspace(-5., 5., m)
    X, Y = np.meshgrid(x_, y_)
    P = [X.flatten(), Y.flatten()]

    # prepare non-nonlinear bounds
    def ci_iq_nl(x):
        return np.array([(x[0] + 2.) ** 2. - x[1]], dtype=np.float64)

    # prepare bounds
    bounds = ((-5., 5.), (-5., 5.))

    # prepare linear bounds
    A = np.array([-4., 10.], dtype=np.float64)

    # assemble bounds
    con = (NonlinearConstraint(ci_iq_nl, lb=(0.,), ub=(np.inf,)),
           LinearConstraint(A, lb=(0.,), ub=(np.inf,)))

    x0 = np.array([0., 0.])
    #res = cma_es(test_function_2d, x0, method='mu-lam', bounds=bounds, constraints=con, seed=3)  # seed=2 is good
    constraints = _prepare_constraints(bounds, con, 2)
    opt = CMAES(test_function_2d, x0, method='IPOP', constraints=constraints, seed=6)  # seed=49 leads to numerical instability
    opt.run()
    res = opt.get_result()

    print('success', res.success, 'x', res.x, res.message)

    # plot white color outside min/max of contour range
    cmap = cm_parula
    cmap.set_under('w')
    cmap.set_over('w')

    fig = plt.figure()
    ax = fig.gca()
    obj = test_function_2d(P)
    levels = np.block([np.arange(0., 20., 2.5), np.arange(20., 100., 5.), np.arange(100., 200., 10.)])
    cont = ax.contourf(X, Y, np.reshape(obj, (m, m)), levels, cmap=cmap, vmin=0., vmax=200.)

    # plot constrained area
    ax.fill_between(x_, (x_ + 2.) ** 2., 5., alpha=.5, edgecolor='k', facecolor=np.array([.5, .5, .5]))
    ax.fill_between(x_, 0.4 * x_, -5., alpha=.5, edgecolor='k', facecolor=np.array([.5, .5, .5]))

    # add colorbar
    plt.colorbar(cont)

    # add optimum
    #if res.success:
    ax.scatter(res.x[0], res.x[1], 20, zorder=2, c='r')

    # set axis limits
    ax.set_xlim([-5., 5.])
    ax.set_ylim([-5., 5])

    fig.savefig(OPTIMIZE_PATH + 'cma_es_con.png')


def plot_cma_es_con_merit():
    tr2 = TR2()
    g06 = G06()
    g08 = G08()
    g11 = G11()

    tol = 1e-2

    ord_ = np.inf
    seed = 12345
    x0_ = np.array([-5., 5])
    mu = 1000.  # penalty parameter
    method = 'IPOP'

    # set up TR2 problem -----------------------------------------------------------------------------------------------
    x0 = tr2.start_guess(x0_)
    s0 = tr2.start_deviation()
    con = _prepare_constraints(tr2.bounds, tr2.constraints, x0.size)
    merit = MeritNormal(tr2.obj, None, con, ord_=ord_)

    def obj_(x):
        return merit.eval(x, mu)

    res_tr2 = cma_es(obj_, x0, method=method, sigma0=s0, tol=tol, seed=seed, keep_path=True)

    # set up G06 problem -----------------------------------------------------------------------------------------------
    x0 = g06.start_guess(x0_)
    s0 = g06.start_deviation()
    con = _prepare_constraints(g06.bounds, g06.constraints, x0.size)
    merit = MeritNormal(g06.obj, None, con, ord_=ord_)

    def obj_(x):
        return merit.eval(x, mu)

    res_g06 = cma_es(obj_, x0, method=method, sigma0=s0, tol=tol, seed=seed, keep_path=True)

    # set up G08 problem -----------------------------------------------------------------------------------------------
    x0 = g08.start_guess(x0_)
    s0 = g08.start_deviation()
    con = _prepare_constraints(g08.bounds, tr2.constraints, x0.size)
    merit = MeritNormal(g08.obj, None, con, ord_=ord_)

    def obj_(x):
        return merit.eval(x, mu)

    res_g08 = cma_es(obj_, x0, method=method, sigma0=s0, tol=tol, seed=seed, keep_path=True)

    # set up G11 problem -----------------------------------------------------------------------------------------------
    x0 = g11.start_guess(x0_)
    s0 = g11.start_deviation()
    con = _prepare_constraints(g11.bounds, g11.constraints, x0.size)
    merit = MeritNormal(g11.obj, None, con, ord_=ord_)

    def obj_(x):
        return merit.eval(x, mu)

    res_g11 = cma_es(obj_, x0, method=method, sigma0=s0, tol=tol, seed=seed, keep_path=True)

    # plot -------------------------------------------------------------------------------------------------------------
    r, c = subplot_layout(4)
    fig, axes = plt.subplots(r, c, figsize=(20, 11))
    fig.subplots_adjust(wspace=0.2, hspace=0.2)
    ax1, ax2, ax3, ax4 = axes.flatten()

    plot_TR2(ax1, res_tr2)
    plot_G06(ax2, res_g06)
    plot_G08(ax3, res_g08)
    plot_G11(ax4, res_g11)

    # overwrite x/y limits
    ax1.set_xlim([-10., 55.])
    ax1.set_ylim([-10., 55.])

    ax2.set_xlim([-5., 110.])
    ax2.set_ylim([-5., 110.])

    ax3.set_xlim([-5., 12.])
    ax3.set_ylim([-1., 12.])

    ax4.set_xlim([-5., 1.])
    ax4.set_ylim([-1.5, 5.])

    fig.savefig(OPTIMIZE_PATH + 'cma_es_con_merit.png')


if __name__ == '__main__':
    plot_cma_es_con()
    #plot_cma_es_con_merit()