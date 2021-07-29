import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from modpy.optimize import nlprog
from modpy.plot.plot_util import cm_parula, hollow_circle
from modpy.illustration.illustration_util import OPTIMIZE_PATH
from modpy.optimize.benchmarks.benchmark_suite import G06, TR2


def _plot_path(ax, res):
    # plot step in each iteration
    for xi in res.path.xi[1:]:
        ax.scatter(xi[0], xi[1], 20, zorder=1, c='b')

    # plot connecting lines between step in each iteration
    xi = np.array(res.path.xi)
    ax.plot(xi[:, 0], xi[:, 1], zorder=1, c='b', lw=1)

    # add start-guess
    ax.scatter(xi[0, 0], xi[0, 1], 20, zorder=2, c='r')

    # add optimum
    if res.success:
        ax.scatter(res.x[0], res.x[1], 20, zorder=2, c='g')


if __name__ == '__main__':

    # plot white color outside min/max of contour range
    cmap = cm_parula
    cmap.set_under('w')
    cmap.set_over('w')

    methods = ('SQP', 'IP')

    m = 100
    x1_ = np.linspace(-10., 110, m)
    x2_ = np.linspace(-10., 110., m)
    X1, X2 = np.meshgrid(x1_, x2_)
    P = [X1.flatten(), X2.flatten()]

    for method in methods:

        # Problem: G06 -------------------------------------------------------------------------------------------------
        g06 = G06()
        x0 = np.array([50., 70.])

        res = nlprog(g06.obj, x0=x0, bounds=g06.bounds, constraints=g06.constraints, method=method,
                     ftol=1e-3, itol=1e-3, maxiter=1000, keep_path=True)

        print(res.x, res.f)
        print('STATUS', res.status)

        fig = plt.figure()
        ax = fig.gca()
        obj = g06.obj(P)
        levels = 1000
        cont = ax.contourf(X1, X2, np.reshape(obj, (m, m)), levels, cmap=cmap, vmin=-40000., vmax=2000000.)

        # plot bounds
        lb1, ub1 = g06.bounds[0]
        # last value in facecolor is alpha
        ax.fill_between([-10, lb1], -10, 110, edgecolor='k', facecolor=np.array([.5, .5, .5, 0.8]), lw=0.3)
        ax.fill_between([ub1, 110], -10, 110., edgecolor='k', facecolor=np.array([.5, .5, .5, 0.8]), lw=0.3)

        lb2, ub2 = g06.bounds[1]
        ax.fill_between(x1_, -10., lb2, alpha=.8, edgecolor='k', facecolor=np.array([.5, .5, .5, 0.8]), lw=0.3)
        ax.fill_between(x1_, ub2, 110, alpha=.8, edgecolor='k', facecolor=np.array([.5, .5, .5, 0.8]), lw=0.3)

        # plot constrained area
        con1 = plt.Circle((5., 5.), np.sqrt(100.), edgecolor='k', facecolor=np.array([.5, .5, .5]), alpha=0.5)
        con2 = hollow_circle((6., 5.), np.sqrt(82.81), (6., 5.), 160, kwargs={'edgecolor': 'k', 'facecolor': np.array([.5, .5, .5]), 'alpha':0.5})
        ax.add_artist(con1)
        ax.add_artist(con2)

        # add colorbar
        plt.colorbar(cont)

        # plot the path
        _plot_path(ax, res)

        # set axis limits
        ax.set_xlim(-10., 110.)
        ax.set_ylim(-10., 110.)

        fig.savefig(OPTIMIZE_PATH + 'G06 ({}).png'.format(method))

    m = 100
    x1_ = np.linspace(-3., 3, m)
    x2_ = np.linspace(-3., 3., m)
    X1, X2 = np.meshgrid(x1_, x2_)
    P = [X1.flatten(), X2.flatten()]

    for method in methods:

        # Problem: TR2 -------------------------------------------------------------------------------------------------
        tr2 = TR2()

        res = nlprog(tr2.obj, x0=tr2.x_ini, bounds=tr2.bounds, constraints=tr2.constraints, method=method,
                     ftol=1e-6, itol=1e-6, maxiter=1000, keep_path=True)

        print(res.x, res.f)
        print('STATUS', res.status)

        fig = plt.figure()
        ax = fig.gca()
        obj = tr2.obj(P)
        levels = 50
        cont = ax.contourf(X1, X2, np.reshape(obj, (m, m)), levels, cmap=cmap, vmin=0., vmax=18.)

        # plot constraints
        ax.fill_between(x1_, 2. - x1_, -5., alpha=.5, edgecolor='k', facecolor=np.array([.5, .5, .5]))

        # add colorbar
        plt.colorbar(cont)

        # plot the path
        _plot_path(ax, res)

        # set axis limits
        ax.set_xlim(-3., 3.)
        ax.set_ylim(-3., 3.)

        fig.savefig(OPTIMIZE_PATH + 'TR2 ({}).png'.format(method))
