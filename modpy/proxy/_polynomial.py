import numpy as np

from modpy.optimize import lsq_linear
from modpy.optimize._optim_util import _ensure_matrix


class PolynomialModel:
    def __init__(self, x, y, method='linear'):

        self.method = method
        self.x = x
        self.y = y

        self.b = None  # parameters

    def initialize(self):
        A = _polynomial_system_matrix(self.x, self.method)
        self.b = _polynomial_fit(A, self.y)

    def eval(self, x):
        A = _polynomial_system_matrix(x, self.method)
        return A @ self.b


def _allocate_system_matrix(shape):
    A = np.zeros(shape)
    A[:, 0] = 1.
    return A


def _assign_linear_terms(A, x):
    _, m = x.shape
    A[:, 1:(m+1)] = x


def _assign_quadratic_terms(A, x):
    _, m = x.shape
    A[:, (m+1):(2*m+1)] = x ** 2.


def _assign_cubic_terms(A, x):
    _, m = x.shape
    A[:, (2*m+1):(3*m+1)] = x ** 3.


def _assign_interaction_terms(A, x, k):
    """
    Add 2-way linear interaction terms to a polynomial system matrix

    Parameters
    ----------
    A : array_like, shape (n, nc)
        System matrix on which to add the interaction terms
    x : array_like, shape (n, m)
        Explanatory variable of n data points and m variables
    k : int
        Starting index for placement of interaction terms
    """

    _, m = x.shape

    for i in range(m):
        for j in range(i+1, m):
            A[:, k] = x[:, i] * x[:, j]
            k += 1


def _linear_system_matrix(x):
    """
    Create system matrix of a model with linear terms and 1st-order 2-way interaction terms

    Parameters
    ----------
    x : array_like, shape (n, m)
        Explanatory variable of n data points and m variables

    Returns
    -------
    A : ndarray, shape (n, 1 + 0.5 m + 0.5 m ** 2)
        System matrix
    """

    n, m = x.shape
    nc = int(1. + .5 * m + .5 * m ** 2.)  # number of coefficients

    if n < nc:
        raise ValueError('Insufficient data points to fit full linear model.')

    A = _allocate_system_matrix((n, nc))
    _assign_linear_terms(A, x)
    _assign_interaction_terms(A, x, m + 1)

    return A


def _quadratic_system_matrix(x):
    """
    Create system matrix of a model with linear, quadratic and 1st-order 2-way interaction terms

    Parameters
    ----------
    x : array_like, shape (n, m)
        Explanatory variable of n data points and m variables

    Returns
    -------
    A : ndarray, shape (n, 1 + 1.5 m + 0.5 m ** 2)
        System matrix
    """

    n, m = x.shape
    nc = int(1. + 1.5 * m + .5 * m ** 2.)  # number of coefficients

    if n < nc:
        raise ValueError('Insufficient data points to fit full quadratic model.')

    A = _allocate_system_matrix((n, nc))
    _assign_linear_terms(A, x)
    _assign_quadratic_terms(A, x)
    _assign_interaction_terms(A, x, 2 * m + 1)

    return A


def _cubic_system_matrix(x):
    """
    Create system matrix of a model with linear, quadratic, cubic and 1st-order 2-way interaction terms

    Parameters
    ----------
    x : array_like, shape (n, m)
        Explanatory variable of n data points and m variables

    Returns
    -------
    A : ndarray, shape (n, 1 + 2.5 m + 0.5 m ** 2)
        System matrix
    """

    n, m = x.shape
    nc = int(1. + 2.5 * m + .5 * m ** 2.)  # number of coefficients

    if n < nc:
        raise ValueError('Insufficient data points to fit full cubic model.')

    A = _allocate_system_matrix((n, nc))
    _assign_linear_terms(A, x)
    _assign_quadratic_terms(A, x)
    _assign_cubic_terms(A, x)
    _assign_interaction_terms(A, x, 3 * m + 1)

    return A


def _polynomial_system_matrix(x, method='linear'):
    """
    Construct the system matrix of a polynomial model including 1st-order 2-way interaction terms::

        Ab=y

    Parameters
    ----------
    x : array_like, shape (n, m)
        Explanatory variable of n data points and m variables
    method : {'linear', 'quadratic', 'cubic'}, optional
        Order of polynomial to fit

    Returns
    -------
    A : ndarray, shape (n, nc)
        System matrix, size varies based on number of included terms
    """

    if not (0 < x.ndim <= 2):
        raise ValueError('`x` must be either a 1D or 2D array.')

    x = _ensure_matrix(x)

    if method == 'linear':
        A = _linear_system_matrix(x)
    elif method == 'quadratic':
        A = _quadratic_system_matrix(x)
    elif method == 'cubic':
        A = _cubic_system_matrix(x)
    else:
        raise ValueError("`method` must be 'linear', 'quadratic' or 'cubic'.")

    return A


def _polynomial_fit(A, y):
    """
    Fit a polynomial model to data including 1st-order 2-way interaction terms::

        Ab=y

    Parameters
    ----------
    A : array_like, shape (n, nc)
        System matrix, size varies based on number of included terms
    y : array_like, shape (n,)
        Response vector

    Returns
    -------
    b : array_like, shape (m,)
        Coefficient vector
    """

    n, nc = A.shape

    if (y.ndim != 1) or (y.size != n):
        raise ValueError('Inconsistent size between `A` and `y`.')

    res = lsq_linear(A, y)
    return np.reshape(res.x, (nc,))
