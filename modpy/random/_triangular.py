import numpy as np

from modpy.special import sqrt
from modpy.random._random_util import _chk_dist_inp, _chk_invdist_inp, _chk_mmm_inp, _chk_log_mmm_inp, _chk_root_mmm_inp


def triangular_pdf(x, a, c, b, bounds=()):
    """
    Calculates the probability density function of the triangular distribution, i.e.::

        f(x; a, c, b) =
        \begin{cases}
            0,                             for x < a
            2 (x - a) / ((b - a) (c - a)), for a <= x < c
            2 / (b - a),                   for x = c
            2 (b - x) / ((b - a) (b - c)), for c < x < b
            0,                             for x > b
        \end{cases}

    Parameters
    ----------
    x : float or array_like, shape (n,)
        Realization.
    a : float
        Minimum.
    c : float
        Mode.
    b : float
        Maximum.
    bounds : tuple
        Tuple of minimum and maximum attainable realizations

    Returns
    -------
    p : float or array_like, shape (n,)
        Probability.
    """

    _chk_mmm_inp(a, b, c)

    if not bounds:
        bounds = (a, b)

    _chk_dist_inp(x, bounds)

    p = np.zeros_like(x)
    p = np.where((x >= a) & (x < c), 2. * (x - a) / ((b - a) * (c - a)), p)
    p = np.where(x == c, 2. / (b - a), p)
    p = np.where(((x > c) & (x < b)), 2. * (b - x) / ((b - a) * (b - c)), p)

    return p


def triangular_cdf(x, a, c, b, bounds=()):
    """
    Calculates the cumulative density function of the triangular distribution, i.e.::

        F(x; a, c, b) =
        \begin{cases}
            0,                                   for x <= a
            (x - a) ^ 2 / ((b - a) (c - a)),     for a < x <= c
            1 - (b - x) ^ 2 / ((b - a) (b - c)), for c < x < b
            1,                                   for x > b
        \end{cases}

    Parameters
    ----------
    x : float or array_like, shape (n,)
        Realization.
    a : float
        Minimum.
    c : float
        Mode.
    b : float
        Maximum.
    bounds : tuple
        Tuple of minimum and maximum attainable realizations

    Returns
    -------
    p : float or array_like, shape (n,)
        Probability.
    """

    _chk_mmm_inp(a, b, c)

    if not bounds:
        bounds = (a, b)

    _chk_dist_inp(x, bounds)

    p = np.zeros_like(x)
    p = np.where((x > a) & (x <= c), (x - a) ** 2. / ((b - a) * (c - a)), p)
    p = np.where(((x > c) & (x < b)), 1. - (b - x) ** 2. / ((b - a) * (b - c)), p)
    p = np.where(x >= b, 1., p)
    return p


def triangular_ppf(p, a, c, b):
    """
    Calculates the inverse of the cumulative density function of the triangular distribution, i.e.::

        F(x; a, c, b) =
        \begin{cases}
            0,                                   for x <= a
            (x - a) ^ 2 / ((b - a) (c - a)),     for a < x <= c
            1 - (b - x) ^ 2 / ((b - a) (b - c)), for c < x < b
            1,                                   for x > b
        \end{cases}

    solved w.r.t. x

    Parameters
    ----------
    p : float or array_like, shape (n,)
        Cumulative probability.
    a : float
        Minimum.
    c : float
        Mode.
    b : float
        Maximum.

    Returns
    -------
    x : float or array_like, shape (n,)
        Realization.
    """

    _chk_mmm_inp(a, b, c)
    _chk_invdist_inp(p)
    f = np.where(b > a, (c - a) / (b - a), 0.)
    return np.where(p < f, a + np.sqrt(p * (b - a) * (c - a)), b - np.sqrt((1. - p) * (b - a) * (b - c)))


def logtriangular_pdf(x, a, c, b, bounds=()):
    """
    Calculates the probability density function of the log-triangular distribution, i.e.::

        f(x; a, c, b) =
        \begin{cases}
            0,                                       for x < a
            2 log(x / a) / (x log(b / a)log(c / a)), for a <= x < c
            2 / log(b / a),                          for x = c
            2 log(b / x) / (x log(b / a)log(b / c)), for c < x < b
            0,                                       for x > b
        \end{cases}

    The log-triangular distribution is unaffected by choice of logarithmic base, so the natural logarithm
    is used in order to simplify expression and reduce computational cost.

    Parameters
    ----------
    x : float or array_like, shape (n,)
        Realization.
    a : float
        Minimum.
    c : float
        Mode.
    b : float
        Maximum.
    bounds : tuple
        Tuple of minimum and maximum attainable realizations

    Returns
    -------
    p : float or array_like, shape (n,)
        Probability.
    """

    _chk_log_mmm_inp(a, b, c)

    if not bounds:
        bounds = (a, b)

    _chk_dist_inp(x, bounds)
    return triangular_pdf(np.log(x), np.log(a), np.log(c), np.log(b)) / x


def logtriangular_cdf(x, a, c, b, bounds=()):
    """
    Calculates the cumulative density function of the log-triangular distribution, i.e.::

        F(x; a, c, b) =
        \begin{cases}
            0,                                            for x <= a
            log(x / a) ^ 2 / (log(b / a) log(c / a)),     for a < x <= c
            1 - log(b / x) ^ 2 / (log(b / a) log(b / c)), for c < x < b
            1,                                            for x > b
        \end{cases}

    The log-triangular distribution is unaffected by choice of logarithmic base, so the natural logarithm
    is used in order to simplify expression and reduce computational cost.

    Parameters
    ----------
    x : float or array_like, shape (n,)
        Realization.
    a : float
        Minimum.
    c : float
        Mode.
    b : float
        Maximum.
    bounds : tuple
        Tuple of minimum and maximum attainable realizations

    Returns
    -------
    p : float or array_like, shape (n,)
        Probability.
    """

    _chk_log_mmm_inp(a, b, c)

    if not bounds:
        bounds = (a, b)

    _chk_dist_inp(x, bounds)
    return triangular_cdf(np.log(x), np.log(a), np.log(c), np.log(b))


def logtriangular_ppf(p, a, c, b):
    """
    Calculates the inverse of the cumulative density function of the log-triangular distribution, i.e.::

        F(x; a, c, b) =
        \begin{cases}
            0,                                   for x <= a
            (x - a) ^ 2 / ((b - a) (c - a)),     for a < x <= c
            1 - (b - x) ^ 2 / ((b - a) (b - c)), for c < x < b
            1,                                   for x > b
        \end{cases}

    The log-triangular distribution is unaffected by choice of logarithmic base, so the natural logarithm
    is used in order to simplify expression and reduce computational cost.

    Parameters
    ----------
    p : float or array_like, shape (n,)
        Cumulative probability.
    a : float
        Minimum.
    c : float
        Mode.
    b : float
        Maximum.

    Returns
    -------
    x : float or array_like, shape (n,)
        Realization.
    """

    _chk_log_mmm_inp(a, b, c)
    _chk_invdist_inp(p)
    return np.exp(triangular_ppf(p, np.log(a), np.log(c), np.log(b)))


def roottriangular_pdf(x, a, c, b, bounds=(), root=2.):
    """
    Calculates the probability density function of the root-triangular distribution, i.e.::

        f(x; a, c, b) =
        \begin{cases}
            0,                                       for x < a
            2 log(x / a) / (x log(b / a)log(c / a)), for a <= x < c
            2 / log(b / a),                          for x = c
            2 log(b / x) / (x log(b / a)log(b / c)), for c < x < b
            0,                                       for x > b
        \end{cases}

    Parameters
    ----------
    x : float or array_like, shape (n,)
        Realization.
    a : float
        Minimum.
    c : float
        Mode.
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

    _chk_root_mmm_inp(a, b, c)

    if not bounds:
        bounds = (a, b)

    _chk_dist_inp(x, bounds)
    return triangular_pdf(sqrt(x, root), sqrt(a, root), sqrt(c, root), sqrt(b, root)) * x ** (1. / root - 1.) / root


def roottriangular_cdf(x, a, c, b, bounds=(), root=2.):
    """
    Calculates the cumulative density function of the log-triangular distribution, i.e.::

        F(x; a, c, b) =
        \begin{cases}
            0,                                            for x <= a
            log(x / a) ^ 2 / (log(b / a) log(c / a)),     for a < x <= c
            1 - log(b / x) ^ 2 / (log(b / a) log(b / c)), for c < x < b
            1,                                            for x > b
        \end{cases}

    Parameters
    ----------
    x : float or array_like, shape (n,)
        Realization.
    a : float
        Minimum.
    c : float
        Mode.
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

    _chk_root_mmm_inp(a, b, c)

    if not bounds:
        bounds = (a, b)

    _chk_dist_inp(x, bounds)
    return triangular_cdf(sqrt(x, root), sqrt(a, root), sqrt(c, root), sqrt(b, root))


def roottriangular_ppf(p, a, c, b, root=2.):
    """
    Calculates the inverse of the cumulative density function of the log-triangular distribution, i.e.::

        F(x; a, c, b) =
        \begin{cases}
            0,                                   for x <= a
            (x - a) ^ 2 / ((b - a) (c - a)),     for a < x <= c
            1 - (b - x) ^ 2 / ((b - a) (b - c)), for c < x < b
            1,                                   for x > b
        \end{cases}

    solved w.r.t. x

    Parameters
    ----------
    p : float or array_like, shape (n,)
        Cumulative probability.
    a : float
        Minimum.
    c : float
        Mode.
    b : float
        Maximum.
    root : float
        Root.

    Returns
    -------
    x : float or array_like, shape (n,)
        Realization.
    """

    _chk_root_mmm_inp(a, b, c)
    _chk_invdist_inp(p)
    return triangular_ppf(p, sqrt(a, root), sqrt(c, root), sqrt(b, root)) ** root
