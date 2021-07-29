import numpy as np


MACHEP = 1.11022302462515654042E-16
PI = np.pi
LOG_PI = np.log(PI)
ROOT_2PI = np.sqrt(2. * PI)
EXP = np.exp(1.)
EXP_NEG2 = np.exp(-2.)


def _poly_eval(x, c):
    """
    Evaluate an arbitrary polynomial of the form::

        y = c[0] * x ** n + c[1] * x ** (n-1) + ...

    NOTE:
    # inspiration from https://github.com/dougthor42/PyErf/blob/master/pyerf/pyerf.py
    # re-written to interact with numpy

    Parameters
    ----------
    x : array_like, shape (n,)
        Variable to be evaluated
    c : array_like, shape (n,)
        Coefficients

    Returns
    -------
    y : float
        Value of evaluated polynomial
    """

    if not (0 < x.ndim <= 1):
        raise ValueError('`x` must be a 1D array.')

    if not (0 < x.ndim <= 1):
        raise ValueError('`c` must be a 1D array.')

    y = c * np.reshape(np.repeat(x, c.size), (-1, c.size)) ** np.arange(c.size - 1, -1, -1)

    return y.sum().sum()


def _poly_eval1(x, c):
    """
    Evaluate an arbitrary polynomial of the form::

        y = x ** n + c[0] * x ** (n-1) + c[1] * x ** (n-2) + ...

    NOTE:
    # inspiration from https://github.com/dougthor42/PyErf/blob/master/pyerf/pyerf.py
    # re-written to interact with numpy

    Parameters
    ----------
    x : array_like, shape (n,)
        Variable to be evaluated
    c : array_like, shape (m,)
        Coefficients

    Returns
    -------
    y : float
        Value of evaluated polynomial
    """

    if isinstance(c, list) or isinstance(c, tuple):
        c = [1] + list(c)
    else:
        c = np.insert(c, 0, 1.)

    return _poly_eval(x, c)
