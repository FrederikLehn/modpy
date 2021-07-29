import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from modpy.optimize import mmo, cma_es
from modpy.plot.plot_util import cm_parula
from modpy.illustration.illustration_util import OPTIMIZE_PATH, test_function_2d
from modpy.optimize import LinearConstraint, NonlinearConstraint


if __name__ == '__main__':
    m = 100
    x_ = np.linspace(-5., 5., m)
    y_ = np.linspace(-5., 5., m)
    X, Y = np.meshgrid(x_, y_)
    P = [X.flatten(), Y.flatten()]

    # wrap core search function
    def core_wrap(obj_, x0, bounds=None, constraints=(), sigma0=0.3, C0=None):
        return cma_es(obj_, x0, bounds=bounds, constraints=constraints, sigma0=sigma0, C0=C0)

    # wrap population generation function
    def pop_wrap(N):
        return np.random.uniform(-5., 5., (N, 2))

    # prepare non-nonlinear bounds
    def ci_iq_nl(x):
        return np.array([(x[0] + 2.) ** 2. - x[1]], dtype=np.float64)

    # prepare linear bounds
    A = np.array([-4., 10.], dtype=np.float64)

    # assemble bounds
    con = (NonlinearConstraint(ci_iq_nl, lb=(0.,), ub=(np.inf,)),
           LinearConstraint(A, lb=(0.,), ub=(np.inf,)))

    E = mmo(test_function_2d, core_wrap, pop_wrap, 2, constraints=con, peaks=4, V=100.).results

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
    for res in E:
        ax.scatter(res.x[0], res.x[1], 20, zorder=1, c='r')

    # set axis limits
    ax.set_xlim([-5., 5.])
    ax.set_ylim([-5., 5])

    fig.savefig(OPTIMIZE_PATH + 'mmo_hill_valley_con.png')
