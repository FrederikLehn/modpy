import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from modpy.proxy._cross_validation import k_fold
from modpy.proxy._proxy_util import _ensure_matrix
from modpy.proxy._rbf import RBFModel, linear_rbf, cubic_rbf, thin_plate_rbf, multi_quadratic_rbf,\
    inverse_quadratic_rbf, inverse_multi_quadratic_rbf, gaussian_rbf, _cross_validation_objective, _calculate_weights, _predict


from modpy.illustration.illustration_util import PROXY_PATH
from modpy.plot.plot_util import default_color, set_font_sizes, subplot_layout, add_point, cm_parula
from modpy._util import point_distance
from modpy.optimize import cma_es


def _plot_K_fold_1D():
    k = 5
    n = 15

    x = np.linspace(0., 1., 15)
    z = np.exp(x * np.cos(3. * np.pi * x))
    x = _ensure_matrix(x)

    ng = 100
    x0 = np.linspace(0., 1., ng)

    phi = multi_quadratic_rbf
    method = 'multi-quad'

    sets = k_fold(n, k, seed=1234)

    r, c = subplot_layout(k)
    fig = plt.figure(figsize=(20, 14))

    for i, (train, test) in enumerate(sets):

        ax = fig.add_subplot(r, c, i + 1)

        for j, kappa in enumerate([0.5, 10.]):
            # train the model
            w = _calculate_weights(x[train, :], z[train], phi, kappa)

            # predict the model
            z_test = _predict(x[test, :], x[train, :], w, phi, kappa)

            model = RBFModel(x, z, method)
            model.initialize(kappa)
            z0 = model.predict(x0)

            ax.scatter(x[test, :], z_test, s=20, color=default_color(j), label='$Z_t, \kappa=${}'.format(kappa), zorder=5)
            ax.plot(x0, z0, c=default_color(j + 2), zorder=3, label='$Z_0, \kappa=${}'.format(kappa))

        ax.scatter(x[train, :], z[train], s=20, c='k', zorder=4, label='Data')

        ax.set_xlabel('$x$')
        ax.set_ylabel('$Z(x)$')
        ax.set_title('Test {}'.format(i))
        ax.grid(True)
        ax.legend()

    fig.savefig(PROXY_PATH + 'RBF_K_fold_CV_1D.png')


def _plot_CV_obj_1D():
    x = np.linspace(0., 1., 15)
    z = np.exp(x * np.cos(3. * np.pi * x))
    x = _ensure_matrix(x)

    rbfs = (linear_rbf, cubic_rbf, thin_plate_rbf, multi_quadratic_rbf, inverse_quadratic_rbf,
            inverse_multi_quadratic_rbf, gaussian_rbf)

    methods = ('Linear', 'Cubic', 'Thin-Plate', 'Multi-Quadratic', 'Inverse-Quadratic', 'Inverse-Multi-Quadratic',
               'Gaussian')

    #rbfs = (multi_quadratic_rbf,)
    #methods = ('Multi-Quadratic',)

    k = 5
    bounds = ((1., None),)
    par0 = np.array([3.])

    ng = 1000
    kappa_max = 10.
    kappa = np.linspace(1., kappa_max, ng)

    r, c = subplot_layout(len(rbfs) + 1)
    fig = plt.figure(figsize=(20, 14))
    plt.subplots_adjust(wspace=0.2, hspace=0.2)

    # plot data
    ax = fig.add_subplot(r, c, 1)
    ax.scatter(x, z, s=20, c='k')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$Z(x)$')
    ax.set_title('Data')
    ax.grid(True)

    # plot objective functions
    for i, (phi, method) in enumerate(zip(rbfs, methods)):
        obj = _cross_validation_objective(x, z, phi, k, seed=1234)

        # calculate objective function value and condition number
        obj_val = np.zeros((ng,))
        for j in range(ng):
            obj_val[j] = obj(np.atleast_1d(kappa[j]))

        res = cma_es(obj, par0.copy(), bounds=bounds, sigma0=1., tol=1e-3, seed=1234)
        print('Optimization: {}, kappa: {}'.format(res.success, res.x))

        ax = fig.add_subplot(r, c, i + 2)
        ax.plot(kappa, obj_val, c=default_color(0), label='Obj($\kappa$)', zorder=3)

        if res.x < 10.:
            ax.scatter(res.x, obj(res.x), c='r', label='Optimum', zorder=4)

        ax.set_xlim([0., kappa_max])
        ax.set_ylim([0., 1.])
        ax.set_xlabel('$\kappa$')
        ax.set_ylabel('$Obj(\kappa)$')
        ax.grid(True)
        ax.legend()

        ax.set_title(method)

    fig.savefig(PROXY_PATH + 'RBF_CV_obj_1D.png')


def _plot_rbf_1D():
    x = np.linspace(0., 1., 15)
    z = np.exp(x * np.cos(3. * np.pi * x))

    n = 100
    x0 = np.linspace(0., 1., n)

    methods = ['lin', 'cubic', 'thin-plate', 'multi-quad', 'inv-quad', 'inv-multi-quad', 'gaussian']

    fig = plt.figure()
    ax = fig.gca()

    for i, method in enumerate(methods):
        model = RBFModel(x, z, method)
        model.initialize_CV(np.array([3.]), 5, seed=1234)

        z0 = model.predict(x0)
        label = '{} ($\kappa$={})'.format(method, np.round(model.kappa, 1))

        ax.scatter(x, z, s=20, c='k', zorder=4)
        ax.plot(x0, z0, c=default_color(i), label=label, zorder=3)

    ax.set_xlabel('$x$')
    ax.set_ylabel('$Z(x)$')
    ax.grid(True)
    ax.legend()

    fig.savefig(PROXY_PATH + 'RBF_1D.png')


def _plot_rbf_2D():
    x = np.array([[1., 1.],
                  [0.5, 0.3],
                  [0.1, 0.1],
                  [0.8, 0.3],
                  [0.25, 0.75]])

    z = np.array([1., 10., 12., 5., 7.])

    methods = ['lin', 'cubic', 'thin-plate', 'multi-quad', 'inv-quad', 'inv-multi-quad', 'gaussian']
    names = ('Linear', 'Cubic', 'Thin-Plate', 'Multi-Quadratic', 'Inverse-Quadratic', 'Inverse-Multi-Quadratic', 'Gaussian')

    ng = 100
    L = np.sqrt(2.)
    x1 = np.linspace(-.5 * L, .5 * L, ng) + .5 * L
    x2 = np.linspace(-.5 * L, .5 * L, ng) + .5 * L
    X1, X2 = np.meshgrid(x1, x2)
    x0 = np.vstack((X1.flatten(), X2.flatten())).T

    r, c = subplot_layout(len(methods) + 1)
    fig = plt.figure(figsize=(20, 14))

    for i, (method, name) in enumerate(zip(methods, names)):
        model = RBFModel(x, z, method)
        model.initialize_CV(np.array([3.]), 5, seed=1234)

        Z = model.predict(x0)
        Z = np.reshape(Z, (ng, ng))

        ax = fig.add_subplot(r, c, i + 1, projection='3d')
        ax.plot_surface(X1, X2, Z, cmap=cm_parula, edgecolors='k', lw=0.2)

        for j in range(5):
            add_point(ax, x[j, 0], x[j, 1], z[j], fc='r', ec='r', radius=0.03)

        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.set_zlabel('$Z(x)$')
        ax.set_title(name)
        set_font_sizes(ax, 12)

        ax.view_init(elev=50, azim=120)

    fig.savefig(PROXY_PATH + 'RBF_2D.png')


if __name__ == '__main__':
    #_plot_K_fold_1D()
    _plot_CV_obj_1D()
    #_plot_rbf_1D()
    #_plot_rbf_2D()
