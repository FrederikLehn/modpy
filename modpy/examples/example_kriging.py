import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MaxNLocator
from numpy.random import Generator, PCG64

from modpy.proxy._kriging import SimpleKrigingModel, OrdinaryKrigingModel, UniversalKrigingModel,\
    _maximum_likelihood_objective, exponential_correlation, gaussian_correlation, matern32_correlation,\
    matern52_correlation, power_exponential_correlation
from modpy.illustration.illustration_util import PROXY_PATH
from modpy.plot.plot_util import default_color, cm_parula, subplot_layout, add_point, set_font_sizes
from modpy.optimize import nlprog, cma_es
from modpy.proxy._kriging import _ensure_matrix


def _plot_ML_obj_1D():

    x, z = np.array([
        [-5.01, 1.06], [-4.90, 0.92], [-4.82, 0.35], [-4.69, 0.49], [-4.56, 0.52],
        [-4.52, 0.12], [-4.39, 0.47], [-4.32, -0.19], [-4.19, 0.08], [-4.11, -0.19],
        [-4.00, -0.03], [-3.89, -0.03], [-3.78, -0.05], [-3.67, 0.10], [-3.59, 0.44],
        [-3.50, 0.66], [-3.39, -0.12], [-3.28, 0.45], [-3.20, 0.14], [-3.07, -0.28],
        [-3.01, -0.46], [-2.90, -0.32], [-2.77, -1.58], [-2.69, -1.44], [-2.60, -1.51],
        [-2.49, -1.50], [-2.41, -2.04], [-2.28, -1.57], [-2.19, -1.25], [-2.10, -1.50],
        [-2.00, -1.42], [-1.91, -1.10], [-1.80, -0.58], [-1.67, -1.08], [-1.61, -0.79],
        [-1.50, -1.00], [-1.37, -0.04], [-1.30, -0.54], [-1.19, -0.15], [-1.06, -0.18],
        [-0.98, -0.25], [-0.87, -1.20], [-0.78, -0.49], [-0.68, -0.83], [-0.57, -0.15],
        [-0.50, 0.00], [-0.38, -1.10], [-0.29, -0.32], [-0.18, -0.60], [-0.09, -0.49],
        [0.03, -0.50], [0.09, -0.02], [0.20, -0.47], [0.31, -0.11], [0.41, -0.28],
        [0.53, 0.40], [0.61, 0.11], [0.70, 0.32], [0.94, 0.42], [1.02, 0.57],
        [1.13, 0.82], [1.24, 1.18], [1.30, 0.86], [1.43, 1.11], [1.50, 0.74],
        [1.63, 0.75], [1.74, 1.15], [1.80, 0.76], [1.93, 0.68], [2.03, 0.03],
        [2.12, 0.31], [2.23, -0.14], [2.31, -0.88], [2.40, -1.25], [2.50, -1.62],
        [2.63, -1.37], [2.72, -0.99], [2.80, -1.92], [2.83, -1.94], [2.91, -1.32],
        [3.00, -1.69], [3.13, -1.84], [3.21, -2.05], [3.30, -1.69], [3.41, -0.53],
        [3.52, -0.55], [3.63, -0.92], [3.72, -0.76], [3.80, -0.41], [3.91, 0.12],
        [4.04, 0.25], [4.13, 0.16], [4.24, 0.26], [4.32, 0.62], [4.44, 1.69],
        [4.52, 1.11], [4.65, 0.36], [4.74, 0.79], [4.84, 0.87], [4.93, 1.01],
        [5.02, 0.55]
    ]).T

    x = np.atleast_2d(x).T

    n = z.size
    m = 1

    correlations = (exponential_correlation, gaussian_correlation, matern32_correlation,
                    matern52_correlation, power_exponential_correlation)

    methods = ('Exponential', 'Gaussian', 'Matérn $\\nu=3/2$', 'Matérn $\\nu=5/2$', 'Power-Exponential')

    # ordinary kriging
    def f(x_):
        return np.ones((x_.shape[0], 1))

    bounds = tuple([(1e-5, None) for _ in range(m)])

    par0 = (np.array([0.6665]), np.array([0.0087]), np.array([0.1635]), np.array([0.1253]), np.array([0.1235]))

    ng = 100
    theta = np.linspace(1e-5, 1., ng)

    r, c = subplot_layout(len(correlations))
    fig = plt.figure(figsize=(20, 11))

    # plot data
    ax = fig.add_subplot(r, c, 1)
    ax.scatter(x, z, s=20, c='k')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$Z(x)$')
    ax.set_title('Data')
    ax.grid(True)

    # plot objective functions
    for i, (corr, method) in enumerate(zip(correlations, methods)):
        obj = _maximum_likelihood_objective('ordinary', x, z, f, corr)

        vals = np.zeros((ng,))
        for j in range(ng):
            print('j = {}'.format(j))
            vals[j] = obj(np.atleast_1d(theta[j]))

        res = cma_es(obj, par0[i], bounds=bounds, sigma0=0.02, tol=1e-3, seed=1234)
        print('Optimization: {}, theta: {}'.format(res.success, res.x))

        ax = fig.add_subplot(r, c, i + 2)
        ax.plot(theta, vals, c=default_color(0), label='Obj($\\theta$)', zorder=1)
        ax.scatter(res.x, obj(res.x), c='r', label='Optimum', zorder=2)

        ax.set_xlim([0., None])
        ax.set_xlabel('$\\theta$')
        ax.set_ylabel('$Obj(\\theta)$')
        ax.grid(True)
        ax.legend()

        ax.set_title(method)

    fig.savefig(PROXY_PATH + 'kriging_ML_obj_1D.png')


def _plot_ML_obj_2D():

    x = np.array([[1., 1.],
                  [0.5, 0.3],
                  [0.1, 0.1],
                  [0.8, 0.3],
                  [0.25, 0.75]])

    z = np.array([1., 10., 12., 5., 7.])

    n = z.size
    m = 2

    correlations = (exponential_correlation, gaussian_correlation, matern32_correlation,
                    matern52_correlation, power_exponential_correlation)

    methods = ('Exponential', 'Gaussian', 'Matérn $\\nu=3/2$', 'Matérn $\\nu=5/2$', 'Power-Exponential ($p=2$)')

    # ordinary kriging
    def f(x_):
        return np.ones((x_.shape[0], 1))

    bounds = tuple([(1e-5, None) for _ in range(m)])

    # run optimization. TODO: change to MMO optimizer
    par0 = np.array([0.5, 1.])

    ng = 100
    theta1 = np.linspace(1e-5, 3., ng)
    theta2 = np.linspace(1e-5, 3., ng)
    X, Y = np.meshgrid(theta1, theta2)
    S = np.vstack((X.flatten(), Y.flatten())).T

    r, c = subplot_layout(len(correlations))
    fig = plt.figure(figsize=(20, 11))

    # plot data
    ax = fig.add_subplot(r, c, 1, projection='3d')
    ax.scatter(x[:, 0], x[:, 1], z, c='r')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('$Z(x)$')
    ax.set_title('Data')
    ax.view_init(elev=40., azim=-110)

    # plot objective functions
    for i, (corr, method) in enumerate(zip(correlations, methods)):
        obj = _maximum_likelihood_objective('ordinary', x, z, f, corr)

        vals = np.zeros((ng ** 2,))
        for j in range(ng ** 2):
            vals[j] = obj(S[j])

        Z = np.reshape(vals, (ng, ng))

        res = cma_es(obj, par0, bounds=bounds, sigma0=0.1, seed=1234)
        print('Optimization: {}, theta: {}'.format(res.success, res.x))

        ax = fig.add_subplot(r, c, i + 2, projection='3d')
        ax.plot_surface(X, Y, Z, cmap=cm_parula, edgecolors='k', lw=0.2, alpha=0.7, zorder=1)
        add_point(ax, res.x[0], res.x[1], res.f, fc='r', ec='r', radius=0.08)

        ax.set_xlabel('$\\theta_1$')
        ax.set_ylabel('$\\theta_2$')
        ax.set_zlabel('$Obj(\\theta)$')
        ax.zaxis.set_major_locator(MaxNLocator(integer=True))

        ax.set_title(method)

        ax.view_init(elev=50., azim=-120)

    #plt.show()
    fig.savefig(PROXY_PATH + 'kriging_ML_obj_2D.png')


def _plot_covariance_functions():
    correlations = (exponential_correlation, gaussian_correlation, matern32_correlation,
                    matern52_correlation, power_exponential_correlation)

    names = ('Exponential', 'Gaussian', 'Matérn $\\nu=3/2$', 'Matérn $\\nu=5/2$', 'Power-Exponential ($p=2$)')

    nb = 1000
    x_min = 0.
    x_max = 1.8
    x0 = np.linspace(x_min, x_max, nb)
    theta = 0.5
    sigma = 4.

    fig = plt.figure()
    ax = fig.gca()

    for i, (corr, name) in enumerate(zip(correlations, names)):

        c = sigma ** 2. * corr(x0, theta)
        ax.plot(x0, c, c=default_color(i), label=name)

    #ax.set_xlim([x.min() * 0.9, x.max() * 1.1])
    #ax.set_ylim([z.min() * 1.5, z.max() * 1.5])

    ax.set_xlim([x_min, x_max])
    ax.set_ylim([0., sigma ** 2. + 1.])
    ax.set_xlabel(r'$h$')
    ax.set_ylabel(r'$C(h)$')
    ax.grid(True)
    ax.legend()

    fig.savefig(PROXY_PATH + 'covariance_functions.png')


def _plot_kernel_impact_1D():

    x = np.array([1., 1.8, 3.8, 4.7, 5.7, 7.3, 7.8, 9.2, 11.1, 12.9])
    z = np.array([0.9, 2.2, -2.3, -4.9, -3.7, 7.3, 7.3, 0.8, -10.8, 7.1])

    methods = ['exp', 'gaussian', 'matern32', 'matern52', 'pow-exp']
    names = ('Exponential', 'Gaussian', 'Matérn $\\nu=3/2$', 'Matérn $\\nu=5/2$', 'Power-Exponential $p=1.5$')

    theta0 = np.array([1.1])

    nb = 200
    ns = 10
    x_min = 0.
    x_max = 15.
    x0 = np.linspace(x_min, x_max, nb)
    h = np.linspace(0., 5., nb)

    r, c = subplot_layout(len(methods))
    fig = plt.figure(figsize=(20, 11))
    ax1 = fig.add_subplot(r, c, 1)

    fill_col = np.array([0.7, 0.7, 0.7, 0.3])

    for i, (method, name) in enumerate(zip(methods, names)):

        if method == 'pow-exp':
            args = (1.5,)
        else:
            args = ()

        model = OrdinaryKrigingModel(x, z, method, seed=1324, args=args)
        model.initialize_ML(theta0)
        model.define_weights(x0)

        z0 = model.mean()
        v0 = model.variance()
        conf = 1.96 * np.sqrt(v0)
        samples = model.sample(ns)

        # add covariance function to ax1
        ax1.plot(h, model.sigma ** 2. * model.corr(h, model.theta), c=default_color(i), label='{} ($\\theta=${})'.format(name, *np.round(model.theta, 2)))

        ax = fig.add_subplot(r, c, i + 2)

        ax.scatter(x, z, s=20, c='k', zorder=4)
        ax.plot(x0, z0, c=default_color(i), label='$Z(x_0)$', zorder=3)
        ax.fill_between(x0, z0 - conf, z0 + conf, color=fill_col, label='Conf.', zorder=1)

        ax.plot(x0, samples[0, :], c='gray', label='Samples', lw=0.5)
        for j in range(1, ns):
            ax.plot(x0, samples[j, :], c='gray', lw=0.5, zorder=2)

        ax.set_xlim([x_min, x_max])
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$Z(x)$')
        ax.set_title(name)
        ax.grid(True)
        ax.legend()

    ax1.set_xlim([0., 5.])
    ax1.set_ylim([0., None])
    ax1.set_xlabel(r'$h$')
    ax1.set_ylabel(r'$C(h)$')
    ax1.set_title('Kernels')
    ax1.grid(True)
    ax1.legend()

    fig.savefig(PROXY_PATH + 'kernel_impact_1D.png')


def _plot_prior_posterior_process():

    x = np.array([1., 1.8, 3.8, 4.7, 5.7, 7.3, 7.8, 9.2, 11.1, 12.9])
    z = np.array([0.9, 2.2, -2.3, -4.9, -3.7, 7.3, 7.3, 0.8, -10.8, 7.1])

    theta0 = np.array([1.1])

    nb = 1000
    ns = 15
    x_min = 0.
    x_max = 15.
    x0 = np.linspace(x_min, x_max, nb)

    fig = plt.figure(figsize=(24, 10))

    fill_col = np.array([0.7, 0.7, 0.7, 0.5])

    model = OrdinaryKrigingModel(x, z, 'gaussian', seed=1324)
    model.initialize_ML(theta0)
    model.define_weights(x0)

    # plot prior process
    m = np.mean(z)
    s = model.sigma
    conf = 1.96 * s
    samples = model.sample(ns, posterior=False)

    ax = fig.add_subplot(1, 2, 1)
    ax.plot([x_min, x_max], [m, m], c=default_color(0), label='$Z(x_0)$', zorder=3)
    ax.fill_between([x_min, x_max], m - conf, m + conf, color=fill_col, label='Conf.', zorder=1)

    ax.plot(x0, samples[0, :], c='gray', label='Samples', lw=1., zorder=2)
    for j in range(1, ns):
        ax.plot(x0, samples[j, :], c='gray', lw=1., zorder=2)

    ax.set_xlim([x_min, x_max])
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$Z(x)$')
    ax.set_title('Prior Probability')
    ax.grid(True)
    ax.legend()
    set_font_sizes(ax, 16)

    # plot posterior process
    z0 = model.mean()
    v0 = model.variance()
    conf = 1.96 * np.sqrt(v0)
    samples = model.sample(ns)

    ax = fig.add_subplot(1, 2, 2)
    ax.scatter(x, z, s=20, c='k', zorder=4)
    ax.plot(x0, z0, c=default_color(0), label='$Z(x_0)$', zorder=3)
    ax.fill_between(x0, z0 - conf, z0 + conf, color=fill_col, label='Conf.', zorder=1)

    ax.plot(x0, samples[0, :], c='gray', label='Samples', lw=1., zorder=2)
    for j in range(1, ns):
        ax.plot(x0, samples[j, :], c='gray', lw=1., zorder=2)

    ax.set_xlim([x_min, x_max])
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$Z(x)$')
    ax.set_title('Posterior Probability')
    ax.grid(True)
    ax.legend(loc='upper left')
    set_font_sizes(ax, 16)

    fig.savefig(PROXY_PATH + 'gaussian_process_prior_posterior.png')


def _plot_kriging_types_1D():

    gen = Generator(PCG64(1234))

    def f(x_):
        x2d = _ensure_matrix(x_)
        n = x2d.shape[0]
        return np.hstack((np.ones((n, 1)), x2d, x2d ** 2.))

    x = np.array([1., 1.8, 3.8, 4.7, 5.7, 7.3, 7.8, 9.2, 11.1, 12.9])
    beta_true = np.array([40., -20., 1.5])
    sigma_true = 4.
    z = f(x) @ beta_true + gen.normal(0., sigma_true, x.size)

    theta0 = np.array([0.9])

    nb = 600
    x0 = np.linspace(0., 15., nb)

    fig = plt.figure()
    ax = fig.gca()

    # simple kriging
    simple = SimpleKrigingModel(x, z, 'gaussian', seed=1324)
    simple.initialize_ML(theta0)
    simple.define_weights(x0)
    z_sim = simple.mean()
    v_sim = simple.variance()
    conf_sim = 1.96 * np.sqrt(v_sim)

    print('SK ($\\beta$={}, $\\theta$={}, $\sigma$={})'\
        .format(np.round(simple.beta, 1), *np.round(simple.theta, 2), np.round(simple.sigma, 1)))

    ax.plot(x0, z_sim, c=default_color(0), label='Simple')
    ax.plot(x0, z_sim - conf_sim, c=default_color(0), lw=0.5, ls='--')
    ax.plot(x0, z_sim + conf_sim, c=default_color(0), lw=0.5, ls='--')

    # ordinary kriging
    ordinary = OrdinaryKrigingModel(x, z, 'gaussian', seed=1324)
    ordinary.initialize_ML(theta0)
    ordinary.define_weights(x0)
    z_ord = ordinary.mean()
    v_ord = ordinary.variance()
    conf_ord = 1.96 * np.sqrt(v_ord)

    print('OK ($\\beta$={}, $\\theta$={}, $\sigma$={})' \
        .format(*np.round(ordinary.beta, 1), *np.round(ordinary.theta, 2), np.round(ordinary.sigma, 1)))

    ax.plot(x0, z_ord, c=default_color(1), label='Ordinary')
    ax.plot(x0, z_ord - conf_ord, c=default_color(1), lw=0.5, ls='--')
    ax.plot(x0, z_ord + conf_ord, c=default_color(1), lw=0.5, ls='--')

    # universal kriging
    universal = UniversalKrigingModel(x, z, f, 'gaussian', seed=1324)
    universal.initialize_ML(theta0)
    universal.define_weights(x0)
    z_uni = universal.mean()
    v_uni = universal.variance()
    conf_uni = 1.96 * np.sqrt(v_uni)

    print('UK ($\\beta$={}, $\\theta$={}, $\sigma$={})' \
        .format(np.round(universal.beta, 1), *np.round(universal.theta, 3), np.round(universal.sigma, 1)))

    ax.plot(x0, z_uni, c=default_color(2), label='Universal')
    ax.plot(x0, z_uni - conf_uni, c=default_color(2), lw=0.5, ls='--')
    ax.plot(x0, z_uni + conf_uni, c=default_color(2), lw=0.5, ls='--')

    # data
    ax.scatter(x, z, s=20, c='k', zorder=2.5)

    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$Z(x)$')
    ax.grid(True)
    ax.legend()

    fig.savefig(PROXY_PATH + 'kriging_types_1D.png')


def _plot_kriging_types_2D():

    def f(x_):
        if x_.ndim == 1:
            x_ = np.reshape(x_, (1, x_.size))

        n, _ = x_.shape
        return np.hstack((np.ones((n, 1)), np.atleast_2d(np.prod(x_, axis=1)).T))

    x = np.array([[1., 1.],
                  [0.5, 0.3],
                  [0.1, 0.1],
                  [0.8, 0.3],
                  [0.25, 0.75]])

    z = np.array([1., 10., 12., 5., 7.])

    theta0 = (np.array([0.4136, 0.5822]), np.array([0.4363, 0.5614]), np.array([0.0381, 0.0812]))
    bounds = ((1e-3, 1.), (1e-3, 1.))

    ng = 100
    L = np.sqrt(2.)
    x1 = np.linspace(-.5 * L, .5 * L, ng) + .5 * L
    x2 = np.linspace(-.5 * L, .5 * L, ng) + .5 * L
    X1, X2 = np.meshgrid(x1, x2)
    x0 = np.vstack((X1.flatten(), X2.flatten())).T

    r, c = 2, 2
    fig = plt.figure(figsize=(20, 14))
    fig.subplots_adjust(hspace=0.05, wspace=0.05)

    ax = fig.add_subplot(r, c, 1, projection='3d')
    ax.scatter(x[:, 0], x[:, 1], z, c='r')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('$Z(x)$')
    ax.set_title('Data')
    ax.view_init(elev=50, azim=120)

    models = (SimpleKrigingModel(x, z, 'gaussian', seed=1324),
              OrdinaryKrigingModel(x, z, 'gaussian', seed=1324),
              UniversalKrigingModel(x, z, f, 'gaussian', seed=1324))

    names = ('Simple', 'Ordinary', 'Universal')

    for i, (model, name) in enumerate(zip(models, names)):
        model.initialize_ML(theta0[i], bounds=bounds)
        model.define_weights(x0)
        Z = model.mean()
        Z = np.reshape(Z, (ng, ng))

        print(name, ' theta={}'.format(model.theta))

        ax = fig.add_subplot(r, c, i + 2, projection='3d')

        ax.plot_surface(X1, X2, Z, cmap=cm_parula, edgecolors='k', lw=0.2, alpha=0.7)

        #ax.scatter(x[:, 0], x[:, 1], z, s=40, c='r', zorder=2.5)
        for j in range(5):
            add_point(ax, x[j, 0], x[j, 1], z[j], fc='r', ec='r', radius=0.03)

        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.set_zlabel('$Z(x)$')
        ax.set_title(name)
        set_font_sizes(ax, 12)

        ax.view_init(elev=50, azim=120)

    fig.savefig(PROXY_PATH + 'kriging_types_2D.png', bbox_inches='tight')


if __name__ == '__main__':
    #_plot_ML_obj_1D()
    #_plot_ML_obj_2D()
    #_plot_covariance_functions()
    #_plot_kernel_impact_1D()
    #_plot_kriging_types_1D()
    _plot_kriging_types_2D()
    #_plot_prior_posterior_process()

