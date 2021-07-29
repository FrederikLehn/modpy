import numpy as np

from modpy.special import gamma, gammainc, gammaincinv
from modpy.random._random_util import _chk_dist_inp, _chk_invdist_inp


def gamma_pdf(x, alpha, beta_, bounds=(0., np.inf)):
    """
    Calculates the probability density function of the gamma distribution, i.e.::

        f(x; a, b) = b^a / \Gamma(a) x^{a - 1} e^(-bx}

    Parameters
    ----------
    x : float or array_like, shape (n,)
        Realization.
    alpha : float
        Shape parameter.
    beta_ : float
        Rate parameter.
    bounds : tuple
        Tuple of minimum and maximum attainable realizations

    Returns
    -------
    p : float or array_like, shape (n,)
        Probability.
    """

    _chk_dist_inp(x, bounds)
    return beta_ ** alpha / gamma(alpha) * x ** (alpha - 1.) * np.exp(-beta_ * x)


def gamma_cdf(x, alpha, beta_, bounds=(0., np.inf)):
    """
    Calculates the cumulative density function of the gamma distribution, i.e.::

        F(x; a, b) = 1 / \Gamma(a) \gamma(bx, a)

    where ``\gamma(bx, a)`` is the lower incomplete gamma function.

    Parameters
    ----------
    x : float or array_like, shape (n,)
        Realization.
    alpha : float
        Shape parameter.
    beta_ : float
        Rate parameter.
    bounds : tuple
        Tuple of minimum and maximum attainable realizations

    Returns
    -------
    p : float or array_like, shape (n,)
        Probability.
    """

    _chk_dist_inp(x, bounds)
    return 1. / gamma(alpha) * gammainc(beta_ * x, alpha)


def gamma_ppf(p, alpha, beta_):
    """
    Calculates the inverse of the cumulative density function of the gamma distribution, i.e.::

        F(x; a, b) = 1 / \Gamma(a) \gamma(bx, a)

    solved w.r.t. x, where ``\gamma(bx, a)`` is the lower incomplete gamma function.

    Parameters
    ----------
    p : float or array_like, shape (n,)
        Cumulative probability.
    alpha : float
        Shape parameter.
    beta_ : float
        Rate parameter.

    Returns
    -------
    x : float or array_like, shape (n,)
        Realization.
    """

    _chk_invdist_inp(p)
    return gammaincinv(p, alpha) / beta_
