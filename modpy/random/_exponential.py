import numpy as np

from modpy.random._random_util import _chk_dist_inp, _chk_invdist_inp, _chk_exp_inp


def exponential_pdf(x, lam, bounds=(0., np.inf)):
    """
    Calculates the probability density function of the exponential distribution, i.e.::

        f(x; \lambda) =
        \begin{cases}
            \lambda e^{-\lambda x},  x >= 0
            0,                       x < 0
        \end{cases}

    Parameters
    ----------
    x : float or array_like, shape (n,)
        Realization.
    lam : float
        Rate parameter.
    bounds : tuple
        Tuple of minimum and maximum attainable realizations

    Returns
    -------
    p : float or array_like, shape (n,)
        Probability.
    """

    _chk_dist_inp(x, bounds)
    _chk_exp_inp(lam)

    m = x >= 0
    p = np.zeros_like(x)
    p[m] = lam * np.exp(-lam * x[m])

    return p


def exponential_cdf(x, lam, bounds=(0., np.inf)):
    """
    Calculates the cumulative density function of the exponential distribution, i.e.::

        F(x; \lambda) =
        \begin{cases}
            1 - e^{-\lambda x},  x >= 0
            0,                   x < 0
        \end{cases}

    Parameters
    ----------
    x : float or array_like, shape (n,)
        Realization.
    lam : float
        Rate parameter.
    bounds : tuple
        Tuple of minimum and maximum attainable realizations

    Returns
    -------
    p : float or array_like, shape (n,)
        Probability.
    """

    _chk_dist_inp(x, bounds)
    _chk_exp_inp(lam)

    m = x >= 0
    p = np.zeros_like(x)
    p[m] = 1. - np.exp(-lam * x[m])

    return p


def exponential_ppf(p, lam):
    """
    Calculates the inverse of the cumulative density function of the exponential distribution, i.e.::

        F^{-1}(p; a, b) = -\ln(1 - p) / \lambda

    solved w.r.t. x, where ``\gamma(bx, a)`` is the lower incomplete gamma function.

    Parameters
    ----------
    p : float or array_like, shape (n,)
        Cumulative probability.
    lam : float
        Rate parameter.

    Returns
    -------
    x : float or array_like, shape (n,)
        Realization.
    """

    _chk_invdist_inp(p)
    _chk_exp_inp(lam)

    return -np.log(1. - p) / lam
