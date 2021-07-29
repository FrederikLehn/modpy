import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from modpy.proxy import *
from modpy.illustration.illustration_util import PROXY_PATH, test_function_2d
from modpy.plot.plot_util import cm_parula


if __name__ == '__main__':
    methods = ['linear', 'quadratic', 'cubic']

    n = 30
    L = np.sqrt(2.)
    x_ = np.linspace(-.5 * L, .5 * L, n) + .5 * L
    y_ = np.linspace(-.5 * L, .5 * L, n) + .5 * L
    X, Y = np.meshgrid(x_, y_)
    P = np.vstack((X.flatten(), Y.flatten())).T

    np.random.seed(0)
    x1 = x_[np.random.randint(0, n, 8)]
    x2 = y_[np.random.randint(0, n, 8)]
    z = test_function_2d(np.vstack((x1, x2)))
    x = np.vstack((x1, x2)).T

    for method in methods:
        model = PolynomialModel(x, z, method)
        model.initialize()
        Z = model.eval(P)
        Z = np.reshape(Z, (n, n))

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(X, Y, Z, cmap=cm_parula, edgecolors='k', lw=0.4, alpha=0.7)
        data = ax.scatter(x[:, 0], x[:, 1], z, s=40, c='r')
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$y$')
        ax.set_zlabel(r'$Z(x,y)$')
        ax.view_init(elev=50, azim=120)
        fig.savefig(PROXY_PATH + 'polynomial_{}.png'.format(method))
