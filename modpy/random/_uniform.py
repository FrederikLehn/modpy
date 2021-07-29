import numpy as np

from modpy.special import sqrt
from modpy.random._random_util import _chk_dist_inp, _chk_invdist_inp, _chk_mmm_inp, _chk_log_mmm_inp,\
    _chk_root_mmm_inp, _chk_prob_inp


def uniform_pdf(x, a, b, bounds=()):
    """
    Calculates the probability density function of the uniform distribution, i.e.::

        f(x; a, b) =
        \begin{cases}
            1 / (b - a), for x\in[a, b]
            0,           otherwise
        \end{cases}

    Parameters
    ----------
    x : float or array_like, shape (n,)
        Realization.
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

    _chk_dist_inp(x, bounds)

    p = np.zeros_like(x)
    return np.where((x >= a) & (x <= b), 1. / (b - a), p)


def uniform_cdf(x, a, b, bounds=()):
    """
    Calculates the cumulative density function of the uniform distribution, i.e.::

        F(x; a, b) =
        \begin{cases}
            0,           for x < a
            1 / (b - a), for x\in[a, b]
            1,           for x > b
        \end{cases}

    Parameters
    ----------
    x : float or array_like, shape (n,)
        Realization.
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

    _chk_dist_inp(x, bounds)

    p = np.zeros_like(x)
    p = np.where((x >= a) & (x <= b), (x - a) / (b - a), p)
    return np.where(x > b, 1., p)


def uniform_ppf(p, a, b):
    """
    Calculates the inverse of the cumulative density function of the uniform distribution, i.e.::

        x = F^{-1}(y; a, b) = a + y * (b - a)

    Parameters
    ----------
    p : float or array_like, shape (n,)
        Cumulative probability.
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
    return a + p * (b - a)


def loguniform_pdf(x, a, b, bounds=()):
    """
    Calculates the probability density function of the log-uniform distribution (reciprocal distribution), i.e.::

        f(x; a, b) =
        \begin{cases}
            1 / (x\ln(b/a)), for x\in[a, b]
            0,               otherwise
        \end{cases}

    The log-uniform distribution is unaffected by choice of logarithmic base, so the natural logarithm
    is used in order to simplify expression and reduce computational cost.

    Parameters
    ----------
    x : float or array_like, shape (n,)
        Realization.
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

    _chk_dist_inp(x, bounds)
    return uniform_pdf(np.log(x), np.log(a), np.log(b)) / x


def loguniform_cdf(x, a, b, bounds=()):
    """
    Calculates the cumulative density function of the log-uniform distribution (reciprocal distribution), i.e.::

        F(x; a, b) =
        \begin{cases}
            0,                for x < a
            log_{b/a}(x / a), for x\in[a, b]
            1,                for x > b
        \end{cases}

    The log-uniform distribution is unaffected by choice of logarithmic base, so the natural logarithm
    is used in order to simplify expression and reduce computational cost.

    Parameters
    ----------
    x : float or array_like, shape (n,)
        Realization.
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

    _chk_dist_inp(x, bounds)
    return uniform_cdf(np.log(x), np.log(a), np.log(b))


def loguniform_ppf(p, a, b):
    """
    Calculates the inverse of the cumulative density function of the log-uniform distribution
    (reciprocal distribution), i.e.::

        x = F^{-1}(y; a, b) = e^{ln(b / a) * p + ln(a)}

    The log-uniform distribution is unaffected by choice of logarithmic base, so the natural logarithm
    is used in order to simplify expression and reduce computational cost.

    Parameters
    ----------
    p : float or array_like, shape (n,)
        Cumulative probability.
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
    _chk_invdist_inp(p)
    return np.exp(uniform_ppf(p, np.log(a), np.log(b)))


def rootuniform_pdf(x, a, b, bounds=(), root=2.):
    """
    Calculates the probability density function of the root-uniform distribution, i.e.::

        f(x; a, b) =
        \begin{cases}
            1 / (n (b^{1/n} - a^{1/n}) * x^{1/n-1}, for x\in[a, b]
            0,                                      otherwise
        \end{cases}

    where `n` is the root of the function.

    Parameters
    ----------
    x : float or array_like, shape (n,)
        Realization.
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

    _chk_root_mmm_inp(a, b)

    if not bounds:
        bounds = (a, b)

    _chk_dist_inp(x, bounds)
    return uniform_pdf(sqrt(x, root), sqrt(a, root), sqrt(b, root)) * x ** (1. / root - 1.) / root


def rootuniform_cdf(x, a, b, bounds=(), root=2.):
    """
    Calculates the cumulative density function of the root-uniform distribution, i.e.::

        F(x; a, b) =
        \begin{cases}
            0,                                       for x < a
            (x^{1/n}-a^{1/n}) / (b^{1/n} - a^{1/n}), for x\in[a, b]
            1,                                       for x > b
        \end{cases}

    where `n` is the root of the function.

    Parameters
    ----------
    x : float or array_like, shape (n,)
        Realization.
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

    _chk_dist_inp(x, bounds)
    return uniform_cdf(sqrt(x, root), sqrt(a, root), sqrt(b, root))


def rootuniform_ppf(p, a, b, root=2.):
    """
    Calculates the inverse of the cumulative density function of the root-uniform distribution, i.e.::

        x = F^{-1}(y; a, b) = (y (b^{1/n} - a^{1/n}) + a^{1/n})^n

    where `n` is the root of the function.

    Parameters
    ----------
    p : float or array_like, shape (n,)
        Cumulative probability.
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
    _chk_invdist_inp(p)
    return uniform_ppf(p, sqrt(a, root), sqrt(b, root)) ** root


def pv2par_uniform(p1, v1, p2, v2):
    """
    Calculates the minimum and the maximum value of a uniform distribution given the probability/value sets
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
    a : float
        Minimum.
    b : float
        Maximum.
    """

    _chk_prob_inp(p1, v1, p2, v2)

    a = (p2 - p1) / (v2 - v1)
    b = p1 - (a * v1)

    return -b / a, (1. - b) / a
