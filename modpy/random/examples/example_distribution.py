import numpy as np
import matplotlib.pyplot as plt

from modpy.random import *
from modpy.illustration.illustration_util import RANDOM_PATH
from modpy.plot.plot_util import default_color


if __name__ == '__main__':
    n_ls = 1000
    n_sa = 10000
    bins = 50

    dists = [NormalDist(3., 1.),
             LogNormalDist(3., 1.),
             RootNormalDist(3., 1.),
             TruncatedNormalDist(3., 1.5, a=1., b=6.),
             TruncatedLogNormalDist(3., 1.5, a=1., b=6.),
             TruncatedRootNormalDist(3., 1.5, a=1., b=6.),
             UniformDist(2., 10.),
             LogUniformDist(2., 10.),
             RootUniformDist(2., 10.),
             TriangularDist(1., 6., 2.5),
             LogTriangularDist(1., 6., 2.5),
             RootTriangularDist(1., 6., 2.5),
             ExponentialDist(1.5),
             GammaDist(),
             BetaDist(2., 5., 5., 12.),
             LogBetaDist(2., 5., 5., 12.),
             RootBetaDist(2., 5., 5., 12.),
             PertDist(2., 10., 5.)]

    ranges = [(0., 8.), (0., 8.), (0., 8.), (-3., 8.), (0., 8.), (0., 8.), (2., 10.), (2., 10.), (2., 10.), (1., 6.), (1., 6.), (1., 6.), (0., 3.), (0., 2.), (5., 12.), (5., 12.), (5., 12.), (2., 10.)]
    names = ['normal', 'lognormal', 'sqrt_normal', 'trunc_normal', 'trunc_lognormal', 'trunc_rootnormal', 'uniform', 'loguniform', 'sqrt_uniform', 'triangular', 'logtriangular', 'sqrt_triangular', 'exponential', 'gamma', 'beta', 'logbeta', 'sqrt_beta', 'pert']

    for d, ran, name in zip(dists, ranges, names):
        x = np.linspace(*ran, n_ls)

        fig = plt.figure()
        ax1 = fig.gca()
        ax1.grid(True)

        sample = d.sample(n_sa)
        sample = sample[sample <= 1.2 * ran[1]]
        ax1.hist(sample, bins, density=True, facecolor=default_color(0), edgecolor='k', linewidth=0.2)
        ax1.plot(x, d.pdf(x), color=default_color(1), lw=2.)

        ax2 = ax1.twinx()
        ax2.plot(x, d.cdf(x), color=default_color(2), lw=2.)

        # add P10, P50 and P90
        percs = d.ppf(np.array([.1, .5, .9]))
        for perc, pn in zip(percs, ('P10', 'P50', 'P90')):
            ax2.plot([perc, perc], [0., 1], c='k', linestyle='--')
            ax2.annotate(pn, (perc + 0.03, 0.96))

        ax1.set_xlim(ran)
        ax1.set_xlabel(r'$x$')
        ax1.set_ylabel(r'PDF, $Pr[X=x]$')

        ax2.set_ylim((0., 1.))
        ax2.set_ylabel(r'CDF, $Pr[X\leq x]$')

        fig.tight_layout()
        fig.savefig(RANDOM_PATH + 'distributions_{}.png'.format(name))
