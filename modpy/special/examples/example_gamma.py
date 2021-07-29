import numpy as np
import matplotlib.pyplot as plt

from modpy.special import gamma, gammaln
from modpy.illustration.illustration_util import SPECIAL_PATH
from modpy.plot.plot_util import default_color


def _plot_gamma():
    x = np.linspace(-5., 5., 1000)
    fig = plt.figure()
    ax = fig.gca()

    ax.plot(x, gamma(x), c=default_color(0), label=r'$\Gamma(x)$')

    ax.grid(True)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$\Gamma(x)$')
    ax.set_ylim((-5., 5.))
    ax.legend()
    fig.savefig(SPECIAL_PATH + 'gamma.png')


def _plot_gammaln():
    x = np.linspace(1e-3, 10., 1000)
    fig = plt.figure()
    ax = fig.gca()

    ax.plot(x, gammaln(x), c=default_color(0), label=r'$\ln(\Gamma(x))$')

    ax.grid(True)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$\ln(\Gamma(x))$')
    ax.legend()
    fig.savefig(SPECIAL_PATH + 'gammaln.png')


if __name__ == '__main__':
    _plot_gamma()
    _plot_gammaln()
