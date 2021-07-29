import numpy as np
import matplotlib.pyplot as plt

from modpy.proxy._geostat import *
from modpy.illustration.illustration_util import PROXY_PATH
from modpy.plot.plot_util import default_color

if __name__ == '__main__':
    methods = ['linear', 'cubic', 'spherical', 'circular', 'exponential', 'gaussian', 'thin plate']
    nb = 10
    ran = np.sqrt(2.)

    fig = plt.figure()
    ax = fig.gca()

    x = np.array([[1., 1.],
                  [0.5, 0.3],
                  [0.1, 0.1],
                  [0.8, 0.3],
                  [0.25, 0.75]])

    z = np.array([1., 10., 12., 5., 7.])

    X = np.linspace(0., ran)

    H = point_distance(x)
    h, v, c, r = empirical_variogram(H, z, nb)

    for i, method in enumerate(methods):
        b = semi_variogram_fit(h, v, r, method=method, nugget=False)
        cov = semi_variogram_callable(r, method, b)

        ax.plot(X, cov(X), c=default_color(i), label=method)

    ax.scatter(h, v, c='r')
    ax.grid(True)

    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$\gamma(x)$')
    ax.set_xlim((0., ran))
    ax.set_ylim((0., None))
    ax.legend()
    fig.savefig(PROXY_PATH + 'semivariograms.png')
