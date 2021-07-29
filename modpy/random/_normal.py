import numpy as np

from modpy.special import EXP, ROOT_2PI, log, erf, erfinv
from modpy.random._random_util import _chk_dist_inp, _chk_invdist_inp, _chk_normal_inp, _chk_prob_inp, _chk_mmm_inp


def _split_finite(x):
    mm = x == -np.inf   # minus infinity mask
    mp = x == np.inf    # plus infinity mask
    mf = ~(mm | mp)     # finite mask
    return mf, mm, mp


def normal_pdf(x, mu=0., sigma=1., bounds=(-np.inf, np.inf)):
    """
    Calculates the probability density function of the normal distribution, i.e.::

        f(x; \mu, \sigma) = 1 / (\sigma \sqrt(2 \pi)) e^{-1 / 2 ((x- \mu) / \sigma) ^ 2}

    Parameters
    ----------
    x : float or array_like, shape (n,)
        Realization.
    mu : float
        Mean.
    sigma : float
        Standard deviation.
    bounds : tuple
        Tuple of minimum and maximum attainable realizations

    Returns
    -------
    p : float or array_like, shape (n,)
        Probability.
    """

    _chk_normal_inp(mu, sigma)
    _chk_dist_inp(x, bounds)
    return 1. / (sigma * ROOT_2PI) * np.exp(-.5 * ((x - mu) / sigma) ** 2.)


def normal_cdf(x, mu=0., sigma=1., bounds=(-np.inf, np.inf)):
    """
    Calculates the cumulative density function of the normal distribution, i.e.::

        F(x; \mu, \sigma) = 1 / 2 (1 + erf((x - \mu) / (\sigma \sqrt(2)))

    Parameters
    ----------
    x : float or array_like, shape (n,)
        Realization.
    mu : float
        Mean.
    sigma : float
        Standard deviation.
    bounds : tuple
        Tuple of minimum and maximum attainable realizations

    Returns
    -------
    p : float or array_like, shape (n,)
        Cumulative probability.
    """

    _chk_normal_inp(mu, sigma)
    _chk_dist_inp(x, bounds)
    return .5 * (1. + erf((x - mu) / (sigma * np.sqrt(2.))))


def normal_ppf(p, mu=0., sigma=1.):
    """
    Calculates the inverse of the cumulative density function of the normal distribution, i.e.::

        x = F^{-1}(p; \mu, \sigma) = \mu + \sigma\sqrt{2} erf^{-1}(2 p - 1)

    Parameters
    ----------
    p : float or array_like, shape (n,)
        Cumulative probability.
    mu : float
        Mean.
    sigma : float
        Standard deviation.

    Returns
    -------
    x : float or array_like, shape (n,)
        Realization.
    """

    _chk_normal_inp(mu, sigma)
    _chk_invdist_inp(p)
    return mu + sigma * np.sqrt(2.) * erfinv(2. * p - 1.)


def lognormal_pdf(x, mu=0., sigma=1., bounds=(-np.inf, np.inf), base=EXP):
    """
    Calculates the probability density function of the lognormal distribution, i.e.::

        f(x; \mu, \sigma) = 1 / (x \ln(n) \sigma \sqrt(2 \pi)) e^{-1 / 2 ((\log_n(x)- \mu) / \sigma) ^ 2}

    where `n` is the logarithmic base, which defaults to base=exp(1), i.e. the natural logarithm, ln.

    Parameters
    ----------
    x : float or array_like, shape (n,)
        Realization.
    mu : float
        Mean of underlying normal distribution.
    sigma : float
        Standard deviation of underlying normal distribution.
    bounds : tuple
        Tuple of minimum and maximum attainable realizations
    base : float
        Logarithmic base

    Returns
    -------
    p : float or array_like, shape (n,)
        Probability.
    """

    # checks are conducted in the method `normal_pdf`
    #return 1. / (x * log(base, base=EXP) * sigma * ROOT_2PI) * np.exp(-.5 * ((log(x, base=base) - mu) / sigma) ** 2.)
    return 1. / (x * np.log(base) * sigma) * normal_pdf((log(x, base=base) - mu) / sigma)


def lognormal_cdf(x, mu=0., sigma=1., bounds=(-np.inf, np.inf), base=EXP):
    """
    Calculates the cumulative density function of the lognormal distribution, i.e.::

        F(x; \mu, \sigma) = 1 / 2 (1 + \erf((\log_n(x) - \mu) / (\sigma \sqrt(2)))

    where `n` is the logarithmic base, which defaults to base=exp(1), i.e. the natural logarithm, ln.

    Parameters
    ----------
    x : float or array_like, shape (n,)
        Realization.
    mu : float
        Mean of underlying normal distribution.
    sigma : float
        Standard deviation of underlying normal distribution.
    bounds : tuple
        Tuple of minimum and maximum attainable realizations
    base : float
        Logarithmic base

    Returns
    -------
    p : float or array_like, shape (n,)
        Cumulative probability.
    """

    # checks are conducted in the method `normal_cdf`
    #return .5 * (1. + erf((log(x, base=base) - mu) / (sigma * np.sqrt(2.))))
    return normal_cdf((log(x, base=base) - mu) / sigma)


def lognormal_ppf(p, mu=0., sigma=1., base=EXP):
    """
    Calculates the inverse of the cumulative density function of the lognormal distribution, i.e.::

        x = F^{-1}(p; \mu, \sigma) = n^{\mu + \sigma\sqrt{2} erf^{-1}(2 p - 1)}

    where `n` is the logarithmic base, which defaults to base=exp(1), i.e. the natural logarithm, ln.

    Parameters
    ----------
    p : float or array_like, shape (n,)
        Cumulative probability.
    mu : float
        Mean of underlying normal distribution.
    sigma : float
        Standard deviation of underlying normal distribution.
    base : float
        Logarithmic base

    Returns
    -------
    x : float or array_like, shape (n,)
        Realization.
    """

    # checks are conducted in the method `normal_ppf`
    #return base ** (mu + sigma * np.sqrt(2.) * erfinv(2. * p - 1.))
    return base ** normal_ppf(p, mu, sigma)


def rootnormal_pdf(x, mu=0., sigma=1., bounds=(-np.inf, np.inf), root=2.):
    """
    Calculates the probability density function of the root-normal distribution, i.e.::

        f(x; \mu, \sigma) = 1 / n x^(1/n-1) 1 / (\sigma \sqrt(2 \pi)) e^{-1 / 2 ((\sqrt[n]{x}- \mu) / \sigma) ^ 2}

    where `n` is the root of the distribution, which defaults to 2, i.e. the square-root.

    Parameters
    ----------
    x : float or array_like, shape (n,)
        Realization.
    mu : float
        Mean of underlying normal distribution.
    sigma : float
        Standard deviation of underlying normal distribution.
    bounds : tuple
        Tuple of minimum and maximum attainable realizations
    root : float
        Root.

    Returns
    -------
    p : float or array_like, shape (n,)
        Probability.
    """

    # checks are conducted in the method `normal_pdf`
    #return x ** (1. / root - 1.) * 1. / (root * sigma * ROOT_2PI) * np.exp(-.5 * ((x ** (1. / root) - mu) / sigma) ** 2.)
    return x ** (1. / root - 1.) / (root * sigma) * normal_pdf((x ** (1. / root) - mu) / sigma)


def rootnormal_cdf(x, mu=0., sigma=1., bounds=(-np.inf, np.inf), root=2.):
    """
    Calculates the cumulative density function of the root-normal distribution, i.e.::

        F(x; \mu, \sigma) = 1 / 2 (1 + \erf((\sqrt[n]{x} - \mu) / (\sigma \sqrt(2)))

    where `n` is the root of the distribution, which defaults to 2, i.e. the square-root.

    Parameters
    ----------
    x : float or array_like, shape (n,)
        Realization.
    mu : float
        Mean of underlying normal distribution.
    sigma : float
        Standard deviation of underlying normal distribution.
    bounds : tuple
        Tuple of minimum and maximum attainable realizations
    root : float
        Root.

    Returns
    -------
    p : float or array_like, shape (n,)
        Cumulative probability.
    """

    # checks are conducted in the method `normal_cdf`
    #return .5 * (1. + erf((x ** (1. / root) - mu) / (sigma * np.sqrt(2.))))
    return normal_cdf((x ** (1. / root) - mu) / sigma)


def rootnormal_ppf(p, mu=0., sigma=1., root=2.):
    """
    Calculates the inverse of the cumulative density function of the root-normal distribution, i.e.::

        x = F^{-1}(p; \mu, \sigma) = (1 / 2 (1 + \erf((\sqrt[n]{x} - \mu) / (\sigma \sqrt(2))))^n

    where `n` is the root of the distribution, which defaults to 2, i.e. the square-root.

    Parameters
    ----------
    p : float or array_like, shape (n,)
        Cumulative probability.
    mu : float
        Mean of underlying normal distribution.
    sigma : float
        Standard deviation of underlying normal distribution.
    root : float
        Root.

    Returns
    -------
    x : float or array_like, shape (n,)
        Realization.
    """

    # checks are conducted in the method `normal_ppf`
    #return (mu + sigma * np.sqrt(2.) * erfinv(2. * p - 1.)) ** root
    return normal_ppf(p, mu, sigma) ** root


def mm2n_par(a, b, sdm=3.):
    """
    Calculates the mean and standard deviation of a lognormal distribution from an assumed min and max, i.e.::

        mu = (a + b) / 2
        sigma = (b - a) / (2 * sdm)

    sdm=1 for 68 % confidence, sdm=2 for 95.4 % confidence and sdm=3 for 99.7 % confidence.
    TODO: change `sdm` input to probability [0, 1] and calculate sdm by normal_ppf(p)

    Parameters
    ----------
    a : float
        Minimum.
    b : float
        Maximum.
    sdm : float
        Standard deviation multiplier (for confidence interval).

    Returns
    -------
    mu : float
        Mean of normal distribution.
    sigma : float
        Standard deviation of normal distribution.
    """

    if a >= b:
        raise ValueError('`a` and `b` must satisfy a<b.')

    if sdm <= 0.:
        raise ValueError('`sdm` must satisfy sdm>0.')

    mu = (a + b) / 2.
    sigma = (b - a) / (2 * sdm)
    return mu, sigma


def n2logn_par(mu, sigma, base=EXP):

    if sigma <= 0:
        raise ValueError('`sigma` must satisfy sigma>0.')

    sigma2 = sigma ** 2.
    m = base ** (mu + sigma2 / 2.)
    sd = np.sqrt(base ** (2. * mu + sigma2) * (base ** sigma2 - 1.))

    return m, sd


def logn2n_par(mu, sigma, base=EXP):
    if mu <= 0:
        raise ValueError('`mu` must satisfy mu>0.')

    if sigma <= 0:
        raise ValueError('`sigma` must satisfy sigma>0.')

    mu2 = mu ** 2.
    sigma2 = sigma ** 2.

    m = log(mu2 / np.sqrt(mu2 + sigma2), base)
    sd = np.sqrt(log(1. + sigma2 / mu2, base))

    return m, sd


def n2sqrtn_par(mu, sigma):

    if sigma <= 0:
        raise ValueError('`sigma` must satisfy sigma>0.')

    mu2 = mu ** 2.
    sigma2 = sigma ** 2.

    m = mu2 + sigma2
    sd = np.sqrt(2. * sigma ** 4. + 4. * mu2 * sigma2)
    return m, sd


def sqrtn2n_par(mu, sigma):
    if mu < 0:
        raise ValueError('`mu` must satisfy mu>=0.')

    if mu < np.sqrt(sigma / 2.):
        raise ValueError('`mu` must satisfy mu>=sqrt(sigma/2).')

    ms = np.sqrt(mu ** 2. - (sigma ** 2.) / 2.)
    m = np.sqrt(ms)
    sd = np.sqrt(mu - ms)

    return m, sd


def pv2par_normal(p1, v1, p2, v2):
    """
    Calculates the mean and standard deviation of a normal distribution given the probability/value sets
    (p1, v1) and (p2, v2).

    Parameters
    ----------
    p1 : float
        Cumulative probability of `v1`.
    v1 : float
        Value at probability `p1`.
    p2 : float
        Cumulative probability of `v2`.
    v2 : float
        Value at probability `p2`.

    Returns
    -------
    mu : float
        Mean.
    sigma : float
        Standard deviation.
    """

    _chk_prob_inp(p1, v1, p2, v2)
    u1 = normal_ppf(p1, 0., 1.)
    u2 = normal_ppf(p2, 0., 1.)

    mu = (v1 * u2 - v2 * u1) / (u2 - u1)
    sigma = (v2 - mu) / u2

    return mu, sigma


def trunc_normal_pdf(x, mu=0., sigma=1., a=-np.inf, b=np.inf, bounds=(-np.inf, np.inf)):
    """
    Calculates the probability density function of the truncated normal distribution, i.e.::

        f(x; \mu, \sigma, a, b) = 1 / \sigma \phi((x-mu)/sigma) / (\Phi((b-\mu)/\sigma) - \Phi((a-\mu)/\sigma))

    where \phi is the PDF of the standard normal distribution and \Phi is the CDF of the standard normal distribution.

    Parameters
    ----------
    x : float or array_like, shape (n,)
        Realization.
    mu : float
        Mean.
    sigma : float
        Standard deviation.
    a : float
        Minimum.
    b : float
        Maximum.
    bounds : tuple
        Tuple of minimum and maximum attainable realizations

    Returns
    -------
    p : float or array_like, shape (n,)
        Probability.
    """

    _chk_mmm_inp(a, b)
    _chk_normal_inp(mu, sigma)

    if not bounds:
        bounds = (a, b)

    _chk_dist_inp(x, bounds)

    alpha, beta_ = _tn2tstdn_par(mu, sigma, a, b)
    p = normal_pdf((x - mu) / sigma) / (sigma * (normal_cdf(beta_) - normal_cdf(alpha)))

    return np.where(x <= a, 0., np.where(x >= b, 0., p))


def trunc_normal_cdf(x, mu=0., sigma=1., a=-np.inf, b=np.inf, bounds=(-np.inf, np.inf)):
    """
    Calculates the cumulative density function of the normal distribution, i.e.::

        F(x; \mu, \sigma, a, b) = (\Phi((x - mu) / sigma) - \Phi((a - mu) / sigma)) / ( \Phi((b - mu) / sigma) - \Phi((a - mu) / sigma))

    where \Phi is the CDF of the standard normal distribution.

    Parameters
    ----------
    x : float or array_like, shape (n,)
        Realization.
    mu : float
        Mean.
    sigma : float
        Standard deviation.
    a : float
        Minimum.
    b : float
        Maximum.
    bounds : tuple
        Tuple of minimum and maximum attainable realizations

    Returns
    -------
    p : float or array_like, shape (n,)
        Cumulative probability.
    """

    _chk_mmm_inp(a, b)
    _chk_normal_inp(mu, sigma)

    if not bounds:
        bounds = (a, b)

    _chk_dist_inp(x, bounds)

    alpha, beta_ = _tn2tstdn_par(mu, sigma, a, b)
    phi_alpha = normal_cdf(alpha)
    p = (normal_cdf((x - mu) / sigma) - phi_alpha) / (normal_cdf(beta_) - phi_alpha)

    return np.where(x <= a, 0., np.where(x >= b, 1., p))


def trunc_normal_ppf(p, mu=0., sigma=1., a=-np.inf, b=np.inf):
    """
    Calculates the inverse of the cumulative density function of the normal distribution, i.e.::

        x = F^{-1}(x; \mu, \sigma, a, b) = \mu + \sigma * (\Phi((a - mu) / sigma) + p * (\Phi((b - mu) / sigma) - \Phi((a - mu) / sigma)))

    Parameters
    ----------
    p : float or array_like, shape (n,)
        Cumulative probability.
    mu : float
        Mean.
    sigma : float
        Standard deviation.
    a : float
        Minimum.
    b : float
        Maximum.

    Returns
    -------
    x : float or array_like, shape (n,)
        Realization.
    """

    _chk_mmm_inp(a, b)
    _chk_invdist_inp(p)

    alpha, beta_ = _tn2tstdn_par(mu, sigma, a, b)
    phi_alpha = normal_cdf(alpha)

    return mu + sigma * normal_ppf(phi_alpha + p * (normal_cdf(beta_) - phi_alpha))


def trunc_normal_sample(mu, sigma, a, b, size, gen):
    """
    Sample from a truncated normal distribution.

    References
    ----------
    [1] Robert, C. P. (2009). Simulation of truncated normal variables. Université Pierre et Marie Curie.
        Link: https://arxiv.org/pdf/0907.4010.pdf

    Parameters
    ----------
    mu : float
        Mean.
    sigma : float
        Standard deviation.
    a : float
        Minimum.
    b : float
        Maximum.
    size : int or tuple
        Size of sampled output.
    gen : np.random.Generator
        Generator used in the random sampling.
    Returns
    -------
    x : float or array_like, shape (n,)
        Realization.
    """

    _chk_mmm_inp(a, b)
    _chk_normal_inp(mu, sigma)

    a, b = _tn2tstdn_par(mu, sigma, a, b)

    if (a != -np.inf) and (b != np.inf):  # two-sided truncation

        x = mu + sigma * _trunc_normal_sample_two_sided(a, b, size, gen)

    else:

        if a != -np.inf:  # lower truncation

            x = mu + sigma * _trunc_normal_sample_one_sided(a, size, gen)

        else:  # upper truncation

            x = -(-mu + sigma * _trunc_normal_sample_one_sided(-b, size, gen))

    return x


def _trunc_normal_sample_one_sided(ab, size, gen):
    """
    Sample from a one-sided lower truncated normal distribution.

    Notice modelling of truncation of the upper tail can be done by calling::

        _trunc_normal_sample_one_sided(-ab, ize, gen)

    References
    ----------
    [1] Robert, C. P. (2009). Simulation of truncated normal variables. Université Pierre et Marie Curie.
        Link: https://arxiv.org/pdf/0907.4010.pdf

    Parameters
    ----------
    ab : float
        Normalized minimum or maximum.
    size : int or tuple
        Size of sampled output.
    gen : np.random.Generator
        Generator used in the random sampling.
    Returns
    -------
    x : float or array_like, shape (n,)
        Realization.
    """

    n = np.prod(size)
    x = np.zeros((n,))

    alpha = (ab + np.sqrt(ab ** 2. + 4.)) / 2.

    def rho(z_):
        return np.exp(-(z_ - alpha) ** 2. / 2.)

    # sample
    i = 0
    while i < n:
        z = gen.exponential(1. / alpha) + ab  # translated exponential distribution
        u = gen.uniform(0., 1.)

        if u <= rho(z):
            x[i] = z
            i += 1

    return np.reshape(x, size)


def _trunc_normal_sample_two_sided(a, b, size, gen):
    """
    Sample from a two-sided truncated normal distribution.

    References
    ----------
    [1] Robert, C. P. (2009). Simulation of truncated normal variables. Université Pierre et Marie Curie.
        Link: https://arxiv.org/pdf/0907.4010.pdf

    Parameters
    ----------
    a : float
        Normalized minimum.
    b : float
        Normalized maximum.
    size : int or tuple
        Size of sampled output.
    gen : np.random.Generator
        Generator used in the random sampling.
    Returns
    -------
    x : float or array_like, shape (n,)
        Realization.
    """

    n = np.prod(size)
    x = np.zeros((n,))

    # prepare acceptance criteria based on `a` and `b`
    if a <= 0. <= b:

        def rho(z_):
            return np.exp(-z_ ** 2. / 2.)

    elif b < 0.:

        def rho(z_):
            return np.exp((b ** 2. - z_ ** 2.) / 2.)

    else:

        def rho(z_):
            return np.exp((a ** 2. - z_ ** 2.) / 2.)

    # sample
    i = 0
    while i < n:
        z = gen.uniform(a, b)
        u = gen.uniform(0., 1.)

        if u <= rho(z):
            x[i] = z
            i += 1

    return np.reshape(x, size)


def _tn2tstdn_par(mu, sigma, a, b):
    """
    Convert `a` and `b` of a truncated normal distribution to a truncated standard normal distribution.

    Parameters
    ----------
    mu : float
        Mean.
    sigma : float
        Standard deviation.
    a : float
        Minimum.
    b : float
        Maximum.
    Returns
    -------
    na : float
        Normalized minimum.
    nb : float
        Normalized maximum.
    """

    return (a - mu) / sigma, (b - mu) / sigma


def trunc_lognormal_pdf(x, mu, sigma, a, b, bounds=(-np.inf, np.inf), base=EXP):
    """
    Calculates the probability density function of the truncated log-normal distribution, i.e.::

        f(x; \mu, \sigma, a, b) = 1 / (x ln(n)\sigma) \phi((\log_n(x)-mu)/sigma) / (\Phi((\log_n(b)-\mu)/\sigma) - \Phi((\log_n(a)-\mu)/\sigma))

    where \phi is the PDF of the standard normal distribution and \Phi is the CDF of the standard normal distribution.
    `n` is the logarithmic base, which defaults to base=exp(1), i.e. the natural logarithm, ln.

    Parameters
    ----------
    x : float or array_like, shape (n,)
        Realization.
    mu : float
        Mean of underlying normal distribution.
    sigma : float
        Standard deviation of underlying normal distribution.
    a : float
        Minimum.
    b : float
        Maximum.
    bounds : tuple
        Tuple of minimum and maximum attainable realizations.
    base : float
        Logarithmic base.

    Returns
    -------
    p : float or array_like, shape (n,)
        Probability.
    """

    # checks are conducted in the method `trunc_normal_pdf`
    p = 1. / (x * np.log(base)) * trunc_normal_pdf(log(x, base=base), mu, sigma, a=log(a, base=base), b=log(b, base=base), bounds=bounds)
    return np.where(x <= a, 0., np.where(x >= b, 0., p))


def trunc_lognormal_cdf(x, mu, sigma, a, b, bounds=(-np.inf, np.inf), base=EXP):
    """
    Calculates the cumulative density function of the normal distribution, i.e.::

        F(x; \mu, \sigma, a, b) = (\Phi((\log_n(x) - mu) / sigma) - \Phi((\log_n(a) - mu) / sigma)) / ( \Phi((log_n(b) - mu) / sigma) - \Phi((\log_n(a) - mu) / sigma))

    where \Phi is the CDF of the standard normal distribution.
    `n` is the logarithmic base, which defaults to base=exp(1), i.e. the natural logarithm, ln.

    Parameters
    ----------
    x : float or array_like, shape (n,)
        Realization.
    mu : float
        Mean of underlying normal distribution.
    sigma : float
        Standard deviation of underlying normal distribution.
    a : float
        Minimum.
    b : float
        Maximum.
    bounds : tuple
        Tuple of minimum and maximum attainable realizations.
    base : float
        Logarithmic base.

    Returns
    -------
    p : float or array_like, shape (n,)
        Cumulative probability.
    """

    # checks are conducted in the method `trunc_normal_cdf`
    p = trunc_normal_cdf(log(x, base=base), mu, sigma, a=log(a, base=base), b=log(b, base=base), bounds=bounds)
    return np.where(x <= a, 0., np.where(x >= b, 1., p))


def trunc_lognormal_ppf(p, mu, sigma, a, b, base=EXP):
    """
    Calculates the inverse of the cumulative density function of the log-normal distribution, i.e.::

        x = F^{-1}(x; \mu, \sigma, a, b) = n^{\mu + \sigma * (\Phi((\log_n(a) - mu) / sigma) + p * (\Phi((\log_n(b) - mu) / sigma) - \Phi((\log_n(a) - mu) / sigma)))}

    Parameters
    ----------
    p : float or array_like, shape (n,)
        Cumulative probability.
    mu : float
        Mean of underlying normal distribution.
    sigma : float
        Standard deviation of underlying normal distribution.
    a : float
        Minimum.
    b : float
        Maximum.
    base : float
        Logarithmic base.

    Returns
    -------
    x : float or array_like, shape (n,)
        Realization.
    """

    # checks are conducted in the method `trunc_normal_ppf`
    return base ** trunc_normal_ppf(p, mu, sigma, a=log(a, base=base), b=log(b, base=base))


def trunc_rootnormal_pdf(x, mu, sigma, a, b, bounds=(-np.inf, np.inf), root=2.):
    """
    Calculates the probability density function of the truncated root-normal distribution, i.e.::

        f(x; \mu, \sigma, a, b) = 1 / \sigma \phi((\sqrt[n]{x}-mu)/sigma) / (\Phi((\sqrt[n]{b}-\mu)/\sigma) - \Phi((\sqrt[n]{a}-\mu)/\sigma))

    where \phi is the PDF of the standard normal distribution and \Phi is the CDF of the standard normal distribution.
    `n` is the root of the distribution, which defaults to 2, i.e. the square-root.

    Parameters
    ----------
    x : float or array_like, shape (n,)
        Realization.
    mu : float
        Mean of underlying normal distribution.
    sigma : float
        Standard deviation of underlying normal distribution.
    a : float
        Minimum.
    b : float
        Maximum.
    bounds : tuple
        Tuple of minimum and maximum attainable realizations.
    root : float
        Root.

    Returns
    -------
    p : float or array_like, shape (n,)
        Probability.
    """

    # checks are conducted in the method `trunc_normal_pdf`
    p = x ** (1. / root - 1.) / root * trunc_normal_pdf(x ** (1. / root), mu, sigma, a=a ** (1. / root), b=b ** (1. / root), bounds=bounds)
    return np.where(x <= a, 0., np.where(x >= b, 0., p))


def trunc_rootnormal_cdf(x, mu, sigma, a, b, bounds=(-np.inf, np.inf), root=2.):
    """
    Calculates the cumulative density function of the normal distribution, i.e.::

        F(x; \mu, \sigma, a, b) = (\Phi((\sqrt[n]{x} - mu) / sigma) - \Phi((\sqrt[n]{a} - mu) / sigma)) / ( \Phi((sqrt[n]{b} - mu) / sigma) - \Phi((\sqrt[n]{a} - mu) / sigma))

    where \Phi is the CDF of the standard normal distribution.
    `n` is the root of the distribution, which defaults to 2, i.e. the square-root.

    Parameters
    ----------
    x : float or array_like, shape (n,)
        Realization.
    mu : float
        Mean of underlying normal distribution.
    sigma : float
        Standard deviation of underlying normal distribution.
    a : float
        Minimum.
    b : float
        Maximum.
    bounds : tuple
        Tuple of minimum and maximum attainable realizations.
    root : float
        Root.

    Returns
    -------
    p : float or array_like, shape (n,)
        Cumulative probability.
    """

    # checks are conducted in the method `trunc_normal_cdf`
    p = trunc_normal_cdf(x ** (1. / root), mu, sigma, a=a ** (1. / root), b=b ** (1. / root), bounds=bounds)
    return np.where(x <= a, 0., np.where(x >= b, 1., p))


def trunc_rootnormal_ppf(p, mu, sigma, a, b, root=2.):
    """
    Calculates the inverse of the cumulative density function of the log-normal distribution, i.e.::

        x = F^{-1}(x; \mu, \sigma, a, b) = (\mu + \sigma * (\Phi((\sqrt[n]{a} - mu) / sigma) + p * (\Phi((\sqrt[n]{b} - mu) / sigma) - \Phi((\sqrt[n]{a} - mu) / sigma))))^n

    Parameters
    ----------
    p : float or array_like, shape (n,)
        Cumulative probability.
    mu : float
        Mean of underlying normal distribution.
    sigma : float
        Standard deviation of underlying normal distribution.
    a : float
        Minimum.
    b : float
        Maximum.
    root : float
        Root.

    Returns
    -------
    x : float or array_like, shape (n,)
        Realization.
    """

    # checks are conducted in the method `trunc_normal_ppf`
    return trunc_normal_ppf(p, mu, sigma, a=a ** (1. / root), b=b ** (1. / root)) ** root
