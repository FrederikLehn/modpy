import numpy as np

from modpy import normalize, scale_translate, range_

from modpy.special import beta, betainc, betaincinv, sqrt
from modpy.random._random_util import _chk_dist_inp, _chk_invdist_inp, _chk_mmm_inp, _chk_log_mmm_inp,\
    _chk_root_mmm_inp, _chk_beta_inp, _chk_normal_inp


def beta_pdf(x, alpha, beta_, a=0., b=1., bounds=()):
    """
    Calculates the probability density function of the beta distribution, i.e.::

        f(z; \alpha, \beta, a, b) = 1 / B(\alpha, \beta) z^{\alpha - 1} (1 - z)^{\beta - 1}) / (b - a)

    where ``z=(x-a)/(b-a)``.

    Parameters
    ----------
    x : float or array_like, shape (n,)
        Realization.
    alpha : float
        Shape parameter 1.
    beta_ : float
        Shape parameter 2.
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

    if not bounds:
        bounds = (a, b)

    _chk_beta_inp(alpha, beta_)
    _chk_dist_inp(x, bounds)
    x_ = normalize(x, a, b)
    return x_ ** (alpha - 1.) * (1. - x_) ** (beta_ - 1.) / beta(alpha, beta_) / range_(a, b)


def beta_cdf(x, alpha, beta_, a=0., b=1., bounds=()):
    """
    Calculates the cumulative density function of the beta distribution, i.e.::

        F(z; a, b) = I_z(a, b)

    where ``z=(x-a)/(b-a)`` and ``I_z(a, b)`` is the regularized incomplete beta function.

    Parameters
    ----------
    x : float or array_like, shape (n,)
        Realization.
    alpha : float
        Shape parameter 1.
    beta_ : float
        Shape parameter 2.
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

    if not bounds:
        bounds = (a, b)

    _chk_beta_inp(alpha, beta_)
    _chk_dist_inp(x, bounds)
    return betainc(normalize(x, a, b), alpha, beta_)


def beta_ppf(p, alpha, beta_, a=0., b=1.):
    """
    Calculates the inverse of the cumulative density function of the beta distribution, i.e.::

        F(z; a, b) = I_z(a, b)

    solved w.r.t. x, where ``z=(x-a)/(b-a)``. and `I_z(a, b)`` is the regularized incomplete beta function.

    Parameters
    ----------
    p : float or array_like, shape (n,)
        Probability.
    alpha : float
        Shape parameter 1.
    beta_ : float
        Shape parameter 2.
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
    _chk_beta_inp(alpha, beta_)
    _chk_invdist_inp(p)
    return scale_translate(betaincinv(p, alpha, beta_), a, b)


def logbeta_pdf(x, alpha, beta_, a, b, bounds=()):
    """
    Calculates the probability density function of the log-beta distribution, i.e.::

        f(z; \alpha, \beta, a, b) = 1 / B(\alpha, \beta) z^{\alpha - 1} (1 - z)^{\beta - 1}) / (x(ln(b) - ln(a))

    where ``z=(ln(x)-ln(a))/(ln(b)-ln(a))``.

    Parameters
    ----------
    x : float or array_like, shape (n,)
        Realization.
    alpha : float
        Shape parameter 1.
    beta_ : float
        Shape parameter 2.
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

    _chk_log_mmm_inp(a, b)

    if not bounds:
        bounds = (a, b)

    _chk_beta_inp(alpha, beta_)
    _chk_dist_inp(x, bounds)

    z = normalize(np.log(x), np.log(a), np.log(b))
    return beta_pdf(z, alpha, beta_, 0., 1.) / range_(np.log(a), np.log(b)) / x


def logbeta_cdf(x, alpha, beta_, a, b, bounds=()):
    """
    Calculates the cumulative density function of the log-beta distribution, i.e.::

        F(z; a, b) = I_z(a, b)

    where ``z=(ln(x)-ln(a))/(ln(b)-ln(a))`` and ``I_z(a, b)`` is the regularized incomplete beta function.

    Parameters
    ----------
    x : float or array_like, shape (n,)
        Realization.
    alpha : float
        Shape parameter 1.
    beta_ : float
        Shape parameter 2.
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

    _chk_log_mmm_inp(a, b)

    if not bounds:
        bounds = (a, b)

    _chk_beta_inp(alpha, beta_)
    _chk_dist_inp(x, bounds)
    return beta_cdf(np.log(x), alpha, beta_, np.log(a), np.log(b))


def logbeta_ppf(p, alpha, beta_, a, b):
    """
    Calculates the inverse of the cumulative density function of the log-beta distribution, i.e.::

        F(z; a, b) = I_z(a, b)

    solved w.r.t. x, where ``z=(ln(x)-ln(a))/(ln(b)-ln(a))``. and `I_z(a, b)`` is the regularized incomplete beta function.

    Parameters
    ----------
    p : float or array_like, shape (n,)
        Probability.
    alpha : float
        Shape parameter 1.
    beta_ : float
        Shape parameter 2.
    a : float
        Minimum.
    b : float
        Maximum.

    Returns
    -------
    x : float or array_like, shape (n,)
        Realization.
    """

    _chk_log_mmm_inp(a, b)
    _chk_beta_inp(alpha, beta_)
    _chk_invdist_inp(p)
    return np.exp(beta_ppf(p, alpha, beta_, np.log(a), np.log(b)))


def rootbeta_pdf(x, alpha, beta_, a, b, bounds=(), root=2.):
    """
    Calculates the probability density function of the log-beta distribution, i.e.::

        f(z; \alpha, \beta, a, b) = 1 / B(\alpha, \beta) z^{\alpha - 1} (1 - z)^{\beta - 1}) / (x(ln(b) - ln(a))

    where ``z=(ln(x)-ln(a))/(ln(b)-ln(a))``.

    Parameters
    ----------
    x : float or array_like, shape (n,)
        Realization.
    alpha : float
        Shape parameter 1.
    beta_ : float
        Shape parameter 2.
    a : float
        Minimum.
    b : float
        Maximum.
    bounds : tuple
        Tuple of minimum and maximum attainable realizations
    root : float
        Root.

    Returns
    -------
    p : float or array_like, shape (n,)
        Probability.
    """

    _chk_root_mmm_inp(a, b)

    if not bounds:
        bounds = (a, b)

    _chk_beta_inp(alpha, beta_)
    _chk_dist_inp(x, bounds)

    z = normalize(sqrt(x, root), sqrt(a, root), sqrt(b, root))
    return beta_pdf(z, alpha, beta_, 0., 1.) / range_(sqrt(a, root), sqrt(b, root)) * x ** (1. / root - 1) / root


def rootbeta_cdf(x, alpha, beta_, a, b, bounds=(), root=2.):
    """
    Calculates the cumulative density function of the log-beta distribution, i.e.::

        F(z; a, b) = I_z(a, b)

    where ``z=(ln(x)-ln(a))/(ln(b)-ln(a))`` and ``I_z(a, b)`` is the regularized incomplete beta function.

    Parameters
    ----------
    x : float or array_like, shape (n,)
        Realization.
    alpha : float
        Shape parameter 1.
    beta_ : float
        Shape parameter 2.
    a : float
        Minimum.
    b : float
        Maximum.
    bounds : tuple
        Tuple of minimum and maximum attainable realizations
    root : float
        Root.

    Returns
    -------
    p : float or array_like, shape (n,)
        Probability.
    """

    _chk_root_mmm_inp(a, b)

    if not bounds:
        bounds = (a, b)

    _chk_beta_inp(alpha, beta_)
    _chk_dist_inp(x, bounds)
    return beta_cdf(sqrt(x, root), alpha, beta_, sqrt(a, root), sqrt(b, root))


def rootbeta_ppf(p, alpha, beta_, a, b, root=2.):
    """
    Calculates the inverse of the cumulative density function of the log-beta distribution, i.e.::

        F(z; a, b) = I_z(a, b)

    solved w.r.t. x, where ``z=(ln(x)-ln(a))/(ln(b)-ln(a))``. and `I_z(a, b)`` is the regularized incomplete beta function.

    Parameters
    ----------
    p : float or array_like, shape (n,)
        Probability.
    alpha : float
        Shape parameter 1.
    beta_ : float
        Shape parameter 2.
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

    _chk_root_mmm_inp(a, b)
    _chk_beta_inp(alpha, beta_)
    _chk_invdist_inp(p)

    return (beta_ppf(p, alpha, beta_, sqrt(a, root), sqrt(b, root))) ** root


def ms2par_beta(mu, sigma, a, b):
    """
    Converts a mean, standard deviation, minimum and maximum to `alpha` and `beta` parameters of a beta distribution.

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
    a : float
        Minimum.
    b : float
        Maximum.
    """

    _chk_mmm_inp(a, b, c=mu)
    _chk_normal_inp(mu, sigma)

    max_mu = b - mu
    mu_min = mu - a
    max_min = b - a
    v = sigma ** 2.

    alpha = (max_mu * mu_min / v - 1.) * mu_min / max_min
    beta_ = (max_mu * mu_min / v - 1.) * max_mu / max_min

    return alpha, beta_


def par2ms_beta(alpha, beta_, a, b):
    """
    Converts a mean, standard deviation, minimum and maximum to `alpha` and `beta` parameters of a beta distribution.

    Parameters
    ----------
    alpha : float
        Shape parameter 1.
    beta_ : float
        Shape parameter 2.
    a : float
        Minimum.
    b : float
        Maximum.

    Returns
    -------
    a : float
        Minimum.
    b : float
        Maximum.
    """

    _chk_mmm_inp(a, b)
    _chk_beta_inp(alpha, beta_)

    m = alpha / (alpha + beta_)
    var = alpha * beta_ / ((alpha + beta_) ** 2. * (alpha + beta_ + 1.)) * (b - a) ** 2.

    return m, np.sqrt(var)
