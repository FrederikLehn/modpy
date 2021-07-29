import numpy as np
import matplotlib.pyplot as plt

from modpy.random import NormalDist, LogNormalDist, RootNormalDist, logn2n_par, sqrtn2n_par
from modpy.special import log, sqrt
from modpy.random._transform import linspace, logspace, rootspace, log2lin, root2lin
from modpy.illustration.illustration_util import RANDOM_PATH
from modpy.plot.plot_util import default_color, set_font_sizes


if __name__ == '__main__':

    n = 10
    a = 0.1
    b = 10.

    lin = linspace(a, b, n)
    root = rootspace(a, b, n, 2)
    ln = logspace(a, b, n, np.exp(1.))
    log10 = logspace(a, b, n, 10.)

    y = np.zeros(n)

    fig, axes = plt.subplots(1, 2, figsize=(20, 14))
    ax1, ax2 = axes.flatten()

    ax1.plot(lin, y, 'o', color=default_color(0), label='Lin')
    ax1.plot(root, y + 0.5, 'o', color=default_color(1), label='Sqrt')
    ax1.plot(ln, y + 1., 'o', color=default_color(2), label='Ln')
    ax1.plot(log10, y + 1.5, 'o', color=default_color(3), label='Log10')

    #ax.set_xlim([a, b])
    ax1.set_xlabel(r'$x$')

    ax1.grid(True)
    ax1.legend(loc='best')
    ax1.yaxis.set_visible(False)
    ax1.yaxis.set_ticks([])
    set_font_sizes(ax1, 13)

    # plot transform CDF of normal-distribution
    n_ = 1000
    mu = 5.
    sigma = 1.5

    norm = NormalDist(mu, sigma)
    rootnorm = RootNormalDist(mu, sigma)
    lnnorm = LogNormalDist(mu, sigma, base=np.exp(1.))
    log10norm = LogNormalDist(mu, sigma, base=10.)

    lin = linspace(a, b, n_)
    root = rootspace(a, b, n_, 2)
    ln = logspace(a, b, n_, np.exp(1.))
    log10 = logspace(a, b, n_, 10.)

    ax2.plot(lin, norm.cdf(lin), color=default_color(0), lw=2., label='Lin')
    ax2.plot(root, norm.cdf(root2lin(root, a, b, 2)), color=default_color(1), lw=2., label='Sqrt')
    ax2.plot(ln, norm.cdf(log2lin(ln, a, b, np.exp(1.))), color=default_color(2), lw=2., label='Ln')
    ax2.plot(log10, norm.cdf(log2lin(log10, a, b, 10.)), color=default_color(3), lw=2., label='Log10')

    ax2.set_xlabel(r'$x$')
    ax2.set_ylabel(r'CDF, $Pr[X\leq x]$')
    ax2.set_ylim([0., 1.])

    ax2.grid(True)
    ax2.legend(loc='best')
    set_font_sizes(ax2, 13)

    fig.savefig(RANDOM_PATH + 'normal_transforms.png')
