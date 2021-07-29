import numpy as np
from numpy.random import Generator, PCG64
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from modpy.stats import hamiltonian_mc
from modpy.stats._core import auto_correlation, auto_correlation_time
from modpy.plot.plot_util import cm_parula, default_color, set_font_sizes
from modpy.illustration.illustration_util import STATS_PATH

from modpy.optimize._derivatives import approx_difference


def _plot_MH_1D():
    # example from: http://www.mit.edu/~ilkery/papers/MetropolisHastingsSampling.pdf

    seed = 1234
    gen = Generator(PCG64(seed))

    n = 150
    samples = 100000

    mu = np.array([0., 0.])
    rho = 0.45
    sigma = np.array([(1., rho),
                      [rho, 1.]])

    x1, x2 = gen.multivariate_normal(mu, sigma, n).T
    rho_emp = np.corrcoef(x1, x2)[0, 1]

    # bi-variate normal distribution with mu1=mu2=0 and sigma1=sigma2=1
    def log_like(rho_):
        p = 1. / (2. * np.pi * np.sqrt(1. - rho_ ** 2.)) * np.exp(-1. / (2. * (1. - rho_ ** 2.)) * (x1 ** 2 - 2. * rho_ * x1 * x2 + x2 ** 2.))
        return np.sum(np.log(p))

    # Jeffreys prior
    def log_prior(rho_):
        return np.log((1. / (1. - rho_ ** 2.)) ** 1.5)

    def log_posterior(rho_):
        return np.minimum(0., log_like(rho_) + log_prior(rho_))

    # def log_like_grad(rho_):
    #     p = (rho_ * (rho_ + 1.) * (rho_ - 1.) + (x1 - rho_ * x2) * (rho_ * x1 - x2)) / ((rho_ + 1) * (rho_ - 1.) * (-rho ** 2. + 1))
    #     return np.sum(p)

    rho0 = np.array([0.])
    bounds = ((-1., 1.),)
    res = hamiltonian_mc(log_posterior, rho0, samples, df='3-point', burn=1000, bounds=bounds, seed=seed, keep_path=True)
    xp = res.x

    # calculate auto-correlation and determine lag-time until independence.
    lags = 100
    auto_corr = auto_correlation(xp.flatten(), lags)
    tau = auto_correlation_time(xp.flatten())

    # sub-sample only uncorrelated samples
    xp_ind = xp[::tau]
    samples_ind = np.arange(0, samples, tau)
    rho_sam = np.mean(xp_ind)

    # plot problem plots -----------------------------------------------------------------------------------------------
    # plot observations
    # ax1.scatter(x1, x2, s=20, color=default_color(0))
    # ax1.set_xlabel('$x_1$')
    # ax1.set_ylabel('$x_2$')
    # ax1.grid(True)
    # ax1.set_title('Data')
    # set_font_sizes(ax1, 12)


    # plot the log-likelihood over the domain [-1, 1]
    # k = 500
    # rhos = np.linspace(-0.999, 0.999, k)
    # L = np.array([log_like(r) for r in rhos])
    #
    # ax4.plot(rhos, L, color=default_color(0))
    # ax4.set_xlabel('$\\rho$')
    # ax4.set_ylabel('$\log(f(\\rho | x, y))$')
    # ax4.grid(True)
    # ax4.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    # ax4.set_title('Log-Likelihood')
    # set_font_sizes(ax4, 12)
    #
    # # plot the gradient of the log-likelihood
    # dL = np.array([log_like_grad(r) for r in rhos])
    #
    # ax5.plot(rhos, dL, color=default_color(0))
    # ax5.set_xlabel('$\\rho$')
    # ax5.set_ylabel('$\\frac{\partial}{\partial \\rho}\log(f(\\rho))$')
    # ax5.grid(True)
    # ax5.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    # ax5.set_title('Derivative of Log-Likelihood')
    # set_font_sizes(ax5, 12)

    # plot HMC behaviour plots -----------------------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(20, 14))
    ax1, ax2, ax3, ax4, ax5, ax6 = axes.flatten()

    # plot Markov chain
    ax1.plot(np.arange(samples), xp, color=default_color(0), label='Full')
    ax1.plot(samples_ind, xp_ind, color=default_color(1), label='Thinned')

    ax1.plot([0, samples], [rho, rho], 'k', label='True $\\rho$')
    ax1.plot([0, samples], [rho_emp, rho_emp], color='m', label='Empirical $\\rho$')
    ax1.plot([0, samples], [rho_sam, rho_sam], lw=2, color='orange', label='Sampled $\\rho$')

    ax1.set_xlim([0, samples])
    ax1.set_ylim([0.2, 0.7])
    ax1.set_xlabel('Samples')
    ax1.set_ylabel('$\\rho$')
    ax1.legend(loc='upper right')
    ax1.grid(True)
    ax1.set_title('Markov Chain')
    set_font_sizes(ax1, 12)

    # plot histogram
    hist = ax2.hist(xp, 50, facecolor=default_color(0))#, edgecolor='k', linewidth=0.2)
    freq = hist[0]
    max_freq = np.amax(freq) * 1.1

    ax2.plot([rho, rho], [0, max_freq], color='k', label='True $\\rho$')
    ax2.plot([rho_emp, rho_emp], [0, max_freq], color='m', label='Empirical $\\rho$')
    ax2.plot([rho_sam, rho_sam], [0, max_freq], lw=2, color='orange', label='Sampled $\\rho$')

    ax2.set_xlim([0.2, 0.7])
    ax2.set_ylim([0., max_freq])
    ax2.set_xlabel('$\\rho$')
    ax2.set_ylabel('Frequency (ind.)')
    ax2.grid(True)
    ax2.set_title('Posterior Distribution')
    set_font_sizes(ax2, 12)

    ax2_1 = ax2.twinx()
    ax2_1.hist(xp_ind, 50, facecolor=default_color(1), alpha=0.35)  # , edgecolor='k', linewidth=0.2
    ax2_1.set_ylabel('Frequency')
    set_font_sizes(ax2_1, 12)

    ax2.legend(handles=(Patch(color=default_color(0), label='Full'),
                        Patch(color=default_color(1), label='Thinned'),
                        Line2D([], [], color='k', label='True $\\rho$'),
                        Line2D([], [], color='m', label='Empirical $\\rho$'),
                        Line2D([], [], color='orange', label='Sampled $\\rho$')))

    # plot the autocorrelation
    ax3.plot(np.arange(lags), auto_corr, color=default_color(0), label='Auto-correlation')
    ax3.plot([tau, tau], [-1., 1.], 'k--', label='Lag-time, $\\tau$')
    ax3.set_xlim([0., lags])
    ax3.set_ylim([-0.1, 1.])
    ax3.set_xlabel('Lag')
    ax3.set_ylabel('Auto-Correlation')
    ax3.legend()
    ax3.grid(True)
    ax3.set_title('Auto-Correlation')
    set_font_sizes(ax3, 12)

    # plot the acceptance probability
    ax4.plot(np.arange(res.path.accept.size), res.path.accept, color=default_color(0), label='$\delta$')
    ax4.plot([0, res.path.accept.size], [0.65, 0.65], 'k--', label='$\delta_{target}$')

    ax4.set_xlim([0, res.path.accept.size])
    ax4.set_ylim([0., 1.])
    ax4.set_xlabel('Samples (incl. burn-in)')
    ax4.set_ylabel('Acceptance Ratio, $\delta$')
    ax4.grid(True)
    ax4.legend()
    ax4.set_title('Acceptance Ratio')
    set_font_sizes(ax4, 12)

    # plot the step-size
    ax5.plot(np.arange(res.path.step_size.size), res.path.step_size, color=default_color(0))

    ax5.set_xlim([0, res.path.step_size.size])
    ax5.set_ylim([0., None])
    ax5.set_xlabel('Burn-in Samples')
    ax5.set_ylabel('Step-size, $\\bar{\epsilon}$')
    ax5.grid(True)
    ax5.set_title('Step-size Adaptation')
    set_font_sizes(ax5, 12)

    # plot the integration steps
    ax6.plot(np.arange(res.path.steps.size), res.path.steps, color=default_color(0))

    ax6.set_xlim([0, res.path.steps.size])
    ax6.set_ylim([0., None])
    ax6.set_xlabel('Samples (incl. burn-in)')
    ax6.set_ylabel('Integration Steps, $L$')
    ax6.grid(True)
    ax6.set_title('Integration Steps')
    set_font_sizes(ax6, 12)

    fig.savefig(STATS_PATH + '1D_performance_hamiltonian_MC.png')


def _plot_MH_2D():
    seed = 1234
    gen = Generator(PCG64(seed))

    n = 150
    samples = 100000
    burn = 1000

    mu = np.array([0., 0.])
    sigma1 = 3.
    sigma2 = 2.
    rho = 0.9
    cov = rho * sigma1 * sigma2
    sigma = np.array([(sigma1 ** 2., cov),
                      [cov, sigma2 ** 2.]])

    x1, x2 = gen.multivariate_normal(mu, sigma, n).T

    s1_emp = np.std(x1)
    s2_emp = np.std(x2)

    # bi-variate normal distribution with mu1=mu2=0, known rho and unknown sigma1 and sigma2
    def log_like(sigma_):
        s1, s2 = sigma_
        p = 1. / (2. * np.pi * s1 * s2 * np.sqrt(1. - rho ** 2.)) * np.exp(-1. / (2. * (1. - rho ** 2.)) * ((x1 / s1) ** 2 - 2. * rho * (x1 / s1) * (x2 / s2) + (x2 / s2) ** 2.))
        return np.sum(np.log(p))

    def log_prior(sigma_):
        s1, s2 = sigma_
        p = 1. / (2. * np.pi * s1 * s2) * np.exp(-1. / 2 * ((x1 / s1) ** 2 + (x2 / s2) ** 2.))
        return np.sum(np.log(p))

    def log_posterior(sigma_):
        return np.minimum(0., log_like(sigma_) + log_prior(sigma_))

    sigma0 = np.array([1.1, 1.1])
    bounds = ((1., None), (1., None))
    res = hamiltonian_mc(log_posterior, sigma0, samples, df='3-point', burn=burn, bounds=bounds, seed=seed, keep_path=True)
    xp = res.x

    # calculate auto-correlation and determine lag-time until independence.
    lags = 100
    auto_corr1 = auto_correlation(xp[:, 0], lags)
    auto_corr2 = auto_correlation(xp[:, 1], lags)

    tau1 = auto_correlation_time(xp[:, 0])
    tau2 = auto_correlation_time(xp[:, 1])
    tau = np.maximum(tau1, tau2)

    # sub-sample only uncorrelated samples
    xp_ind = xp[::tau, :]

    s1_sam = np.mean(xp_ind[:, 0])
    s2_sam = np.mean(xp_ind[:, 1])

    # plot problem plots -----------------------------------------------------------------------------------------------
    # # plot observations
    # ax1 = fig.add_subplot(r, c, 1)
    # ax1.scatter(x1, x2, s=20, color=default_color(0))
    # ax1.set_xlabel('$x_1$')
    # ax1.set_ylabel('$x_2$')
    # ax1.grid(True)
    # ax1.set_title('Data')
    # set_font_sizes(ax1, 12)

    # # plot the likelihood over the domain
    # ng = 250
    # ng2 = ng ** 2
    # s1_ = np.linspace(1., 5., ng)
    # s2_ = np.linspace(1., 5., ng)
    # S1, S2 = np.meshgrid(s1_, s2_)
    # S = [S1.flatten(), S2.flatten()]
    #
    # # calculate likelihood
    # L = np.zeros((ng2,))
    # for i in range(ng2):
    #     L[i] = log_like([S[0][i], S[1][i]])
    #
    # L = np.reshape(L, (ng, ng))
    #
    # ax4 = fig.add_subplot(r, c, 4, projection='3d')
    # ax4.plot_surface(S1, S2, L, cmap=cm_parula, edgecolors='k', lw=0.2)
    #
    # ax4.set_xlabel('$\sigma_1$')
    # ax4.set_ylabel('$\sigma_2$')
    # ax4.set_zlabel('$\log(f(\\rho | x, y))$')
    # ax4.grid(True)
    # ax4.set_title('Log-Likelihood')
    # ax4.ticklabel_format(axis="z", style="sci", scilimits=(0, 0))
    # set_font_sizes(ax4, 12)

    # # calculate prior probability
    # pri = np.zeros((ng2,))
    # for i in range(ng2):
    #     pri[i] = log_like_grad(np.array([S[0][i], S[1][i]]))
    #
    # pri = np.reshape(pri, (ng, ng))
    #
    # ax5 = fig.add_subplot(r, c, 5, projection='3d')
    # ax5.plot_surface(S1, S2, pri, cmap=cm_parula, edgecolors='k', lw=0.2)
    #
    # ax5.set_xlabel('$\sigma_1$')
    # ax5.set_ylabel('$\sigma_2$')
    # ax5.set_zlabel('$\log(f(\\rho))$')
    # ax5.grid(True)
    # ax5.set_title('Log-Prior Probability')
    # ax5.ticklabel_format(axis="z", style="sci", scilimits=(0, 0))
    # set_font_sizes(ax5, 12)

    # plot HMC behaviour plots -----------------------------------------------------------------------------------------
    r, c = 2, 3
    fig = plt.figure(figsize=(20, 14))

    # plot markov chain
    ax1 = fig.add_subplot(r, c, 1)
    ax1.plot(xp[:, 0], xp[:, 1], color=default_color(0), label='Full')
    ax1.plot(xp_ind[:, 0], xp_ind[:, 1], color=default_color(1), label='Thinned')

    ax1.plot(sigma1, sigma2, color='k', marker='o', ls='', ms=8, label='True $(\sigma_1, \sigma_2)$')
    ax1.plot(s1_emp, s2_emp, color='m', marker='o', ls='', ms=8, label='Empirical $(\sigma_1, \sigma_2)$')
    ax1.plot(s1_sam, s2_sam, color='g', marker='o', ls='', ms=8, label='Sampled $(\sigma_1, \sigma_2)$')

    ax1.set_xlim([2., 4.5])
    ax1.set_ylim([1.4, 2.7])
    ax1.set_xlabel('$\sigma_1$')
    ax1.set_ylabel('$\sigma_2$')
    ax1.grid(True)
    ax1.legend()
    ax1.set_title('Markov Chain')
    set_font_sizes(ax1, 12)

    # plot histogram
    cmap = cm_parula
    ax2 = fig.add_subplot(r, c, 2)
    ax2.hist2d(xp[:, 0], xp[:, 1], 100, cmap=cmap, range=[[2., 4.5], [1.4, 2.7]])

    ax2.plot(sigma1, sigma2, color='k', marker='o', ls='', ms=8)
    ax2.plot(s1_emp, s2_emp, color='m', marker='o', ls='', ms=8)
    ax2.plot(s1_sam, s2_sam, color='g', marker='o', ls='', ms=8)

    ax2.set_xlim([2., 4.5])
    ax2.set_ylim([1.4, 2.7])
    ax2.set_xlabel('$\sigma_1$')
    ax2.set_ylabel('$\sigma_2$')
    ax2.grid(True)
    ax2.set_title('Posterior Distribution')
    set_font_sizes(ax2, 12)

    # plot the autocorrelation
    ax3 = fig.add_subplot(r, c, 3)
    ax3.plot(np.arange(lags), auto_corr1, color=default_color(0), label='Auto-corr, $\sigma_1$')
    ax3.plot(np.arange(lags), auto_corr2, color=default_color(1), label='Auto-corr, $\sigma_2$')
    ax3.plot([tau, tau], [-1., 1.], 'k--', label='Lag-time, $\\tau$')
    ax3.set_xlim([0, lags])
    ax3.set_ylim([-0.1, 1.])
    ax3.set_xlabel('Lag')
    ax3.set_ylabel('Auto-Correlation')
    ax3.grid(True)
    ax3.legend()
    ax3.set_title('Auto-Correlation')
    set_font_sizes(ax3, 12)

    # plot the acceptance probability
    ax4 = fig.add_subplot(r, c, 4)
    ax4.plot(np.arange(res.path.accept.size), res.path.accept, color=default_color(0), label='$\delta$')
    ax4.plot([0, res.path.accept.size], [0.65, 0.65], 'k--', label='$\delta_{target}$')

    ax4.set_xlim([0, res.path.accept.size])
    ax4.set_ylim([0., 1.])
    ax4.set_xlabel('Samples (incl. burn-in)')
    ax4.set_ylabel('Acceptance Ratio, $\delta$')
    ax4.grid(True)
    ax4.legend()
    ax4.set_title('Acceptance Ratio')
    set_font_sizes(ax4, 12)

    # plot the step-size
    ax5 = fig.add_subplot(r, c, 5)
    ax5.plot(np.arange(res.path.step_size.size), res.path.step_size, color=default_color(0))

    ax5.set_xlim([0, res.path.step_size.size])
    ax5.set_ylim([0., None])
    ax5.set_xlabel('Burn-in Samples')
    ax5.set_ylabel('Step-size, $\\bar{\epsilon}$')
    ax5.grid(True)
    ax5.set_title('Step-size Adaptation')
    set_font_sizes(ax5, 12)

    # plot the integration steps
    ax6 = fig.add_subplot(r, c, 6)
    ax6.plot(np.arange(res.path.steps.size), res.path.steps, color=default_color(0))

    ax6.set_xlim([0, res.path.steps.size])
    ax6.set_ylim([0., None])
    ax6.set_xlabel('Samples (incl. burn-in)')
    ax6.set_ylabel('Integration Steps, $L$')
    ax6.grid(True)
    ax6.set_title('Integration Steps')
    set_font_sizes(ax6, 12)

    fig.savefig(STATS_PATH + '2D_performance_hamiltonian_MC.png')


if __name__ == '__main__':
    _plot_MH_1D()
    _plot_MH_2D()
