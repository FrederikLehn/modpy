import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from modpy.special import beta, betainc
from modpy.illustration.illustration_util import SPECIAL_PATH
from modpy.plot.plot_util import default_color, cm_parula


def _plot_beta():
    n = 80
    a = np.linspace(-3., 3., n)
    b = np.linspace(-3., 3., n)
    A, B = np.meshgrid(a, b)
    Y = beta(A.flatten(), B.flatten())

    # remove values that cannot be plotted
    Y = np.where(np.isnan(Y), 0., Y)
    Y = np.where(np.isinf(Y), 0., Y)
    Y = np.real(Y)

    fig = plt.figure()
    ax = fig.gca()

    norm = mpl.colors.SymLogNorm(linthresh=0.01, linscale=0.01, vmin=Y.min(), vmax=Y.max())
    cont = ax.contourf(A, B, np.reshape(Y, (n, n)), 200, cmap=cm_parula, norm=norm)
    fig.colorbar(cont, ax=ax, extend='max')
    #plt.colorbar(cont)

    ax.set_xlabel(r'$a$')
    ax.set_ylabel(r'$b$')

    fig.savefig(SPECIAL_PATH + 'beta.png')


def _plot_betainc():
    x = np.linspace(0., 10., 1000)
    fig = plt.figure()
    ax = fig.gca()

    ax.plot(x, betainc(x), c=default_color(0), label=r'$\ln(\Gamma(x))$')

    ax.grid(True)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$\ln(\Gamma(x))$')
    ax.legend()
    fig.savefig(SPECIAL_PATH + 'lngamma.png')


if __name__ == '__main__':
    _plot_beta()
    #_plot_betainc()
