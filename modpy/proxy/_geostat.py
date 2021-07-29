from functools import partial
import numpy as np
import numpy.linalg as la

from modpy._util import point_distance
from modpy.optimize import lsq_linear


def kriging(s, kernel, method='simple'):
    """
    Estimate a response hyper-surface based on kriging

    Parameters
    ----------
    s : array_like, shape (p, q)
        Points at which to estimate the response surface
    kernel : KrigingKernel
        Class KrigingKernel
    method : {'simple', 'ordinary'}, optional TODO: implement ordinary

    Returns
    -------
    z : ndarray, shape (p,)
        Estimated response surface
    """

    return _simple_kriging(s, kernel)


def _simple_kriging(s, kernel):
    """
    Estimate a response hyper-surface based on simple kriging

    Parameters
    ----------
    s : array_like, shape (p, q)
        Points at which to estimate the response surface
    kernel : KrigingKernel
        Class KrigingKernel

    Returns
    -------
    z : ndarray, shape (p,)
        Estimated response surface
    """

    Q, R = la.qr(kernel.cov(kernel.h))

    n, _ = s.shape
    z = np.zeros((n, 1))

    for i in range(z.size):
        w = la.solve(R, Q.T @ kernel.cov(la.norm(kernel.s-s[i, :], axis=1, ord=kernel.ord)))
        z[i] = w.T @ kernel.z.T

    return z


class KrigingKernel:
    def __init__(self, s, z, method, nugget, ord_, bins):

        self.method = method    # {'linear, 'cubic', 'spherical', 'circular', 'exponential', 'gaussian', 'thin plate'}
        self.ord = ord_         # norm-order,       {non-zero int, inf, -inf, ‘fro’, ‘nuc’}
        self.s = s              # spatial points,   ndarray, shape (n, m)
        self.z = z              # responses,        ndarray, shape (n,)
        self.h = None           # spatial distance, ndarray, shape (n, n)

        self.bins = bins        # bins for fitting, int
        self.nugget = nugget    # include nugget,   boolean
        self.r = 1.             # range,            float
        self.b = None           # fit parameters,   ndarray, shape(p,)
        self.cov = None         # co-variogram,     callable

    def calculate_distance(self):
        self.h = point_distance(self.s, self.ord)

    def initialize_covariogram(self):
        h, v, c, r = empirical_variogram(self.h, self.z, self.bins)
        self.b = co_variogram_fit(h, v, r, method=self.method, nugget=self.nugget)
        self.r = r
        self.cov = co_variogram_callable(self.r, self.method, self.b)

    def update(self, s, z):
        self.s = np.vstack(self.s, s)
        self.z = np.vstack(self.z, z)
        self.calculate_distance()
        self.initialize_covariogram()


def kriging_kernel(s, z, method=None, nugget=True, ord_=None, bins=10):

    if not (0 < s.ndim <= 2):
        raise ValueError('`s` must be either a 1D or 2D array')

    s = np.atleast_2d(s)
    n, _ = s.shape

    if z.size != n:
        raise ValueError('Inconsistent size between `s` and `z`.')

    if method is None:
        pass
        # TODO: fitting

    kernel = KrigingKernel(s, z, method, nugget, ord_, bins)
    kernel.calculate_distance()
    kernel.initialize_covariogram()

    return kernel


def _linear_semi_var(h, r, b):
    return np.where(h == 0., 0., np.where(h > r, np.sum(b), b[0] + b[1] * h / r))


def _cubic_semi_var(h, r, b):
    return np.where(h == 0., 0., np.where(h > r, np.sum(b), b[0] + b[1] * (h / r) ** 3.))


def _spherical_semi_var(h, r, b):
    hr = h / r
    return np.where(h == 0., 0., np.where(h > r, np.sum(b), b[0] + b[1] * (1.5 * hr - .5 * (h / r) ** 3.)))


def _circular_semi_var(h, r, b):
    hr = h / r
    return np.where(h == 0., 0., np.where(h > r, np.sum(b),
                    b[0] + b[1] * (1. - 2. / np.pi * np.arccos(hr) + np.sqrt(1. - hr ** 2.))))


def _exponential_semi_var(h, r, b):
    return np.where(h == 0., 0., b[0] + b[1] * (1. - np.exp(-h / r)))


def _gaussian_semi_var(h, r, b):
    return np.where(h == 0., 0., b[0] + b[1] * (1. - np.exp(-(h / r) ** 2.)))


def _thinplate_semi_var(h, r, b):
    hr = h/r
    return np.where(h == 0., 0., np.where(h > r, np.sum(b), b[0] + b[1] * (hr ** 2. * np.log(hr))))


def empirical_variogram(H, z, nb):
    """
    Creates an empirical semi-variogram and co-variogram.

    Parameters
    ----------
    H : array_like, shape (n, n)
        Distance matrix
    z : array_like, shape (n,)
        Response vector
    nb : int
        Number of bins in the empirical variogram

    Returns
    -------
    h : ndarray, shape (nb,)
        Distance vector
    v : ndarray, shape(nb,)
        Semi-variogram vector
    c : ndarray, shape(nb,)
        Co-variogram vector
    r : float
        Range
    """

    if H.ndim != 2:
        raise ValueError('`X` must be a 2D array.')

    n, _ = H.shape

    if n != H.shape[1]:
        raise ValueError('`H` must be a square matrix.')

    if (z.ndim != 1) or (z.size != n):
        raise ValueError('Inconsistent size between `H` and `z`.')

    if nb < 1:
        raise ValueError('Number of bins must be larger than 0.')

    freq, edges = np.histogram(H, nb)
    _norm = np.where(freq > 0, freq, 1)
    idx = np.digitize(H, edges[:-1]) - 1  # for zero indexing
    v = np.zeros(nb)
    c = np.zeros(nb)
    m = np.zeros(nb)

    for j in range(n):
        for k in range(j+1, n):
            i = idx[j, k]
            v[i] += (z[j] - z[k]) ** 2.
            m[i] += z[j] + z[k]

    m /= _norm

    for j in range(n):
        for k in range(j, n):
            i = idx[j, k]
            c[i] += (z[j] - m[i]) * (z[k] - m[i])

    h = edges[:-1] + .5 * np.diff(edges)
    v /= _norm
    c /= _norm
    r = edges[-1]

    # return non-zero bins to avoid skewing curve-fits
    nz = freq != 0
    return h[nz], v[nz], c[nz], r


def semi_variogram_callable(r, method, b=(0, 1)):
    """
    Return a callable function of a pre-defined semi-variogram method

    Parameters
    ----------
    r : float
        Range (distance) at which data has no impact on variance
    method : {'linear', 'cubic', 'spherical', 'exponential', 'gaussian', 'thin plate'}, optional
        Method to fit coefficients to
    b : array_like, shape (2,), optional
        Model coefficients b=[slope, nugget]

    Returns
    -------
    f : callable
        Callable (function of h) semi-variogram
    """

    if method == 'linear':
        f = partial(_linear_semi_var, r=r, b=b)
    elif method == 'cubic':
        f = partial(_cubic_semi_var, r=r, b=b)
    elif method == 'spherical':
        f = partial(_spherical_semi_var, r=r, b=b)
    elif method == 'circular':
        f = partial(_circular_semi_var, r=r, b=b)
    elif method == 'exponential':
        f = partial(_exponential_semi_var, r=r, b=b)
    elif method == 'gaussian':
        f = partial(_gaussian_semi_var, r=r, b=b)
    elif method == 'thin plate':
        f = partial(_thinplate_semi_var, r=r, b=b)
    else:
        raise ValueError("`method` must be 'linear', 'cubic', 'spherical', 'circular', 'exponential', 'gaussian'"
                         "or 'thin plate'.")

    return f


def _co_variogram_fun(h, f, b=(0., 1.)):
    return b[1] * (1. - f(h)) + b[0]


def co_variogram_callable(r, method, b=(1., 1)):
    """
    Return a callable function of a pre-defined co-variogram method

    Parameters
    ----------
    r : float
        Range (distance) at which data has no impact on variance
    method : {'linear', 'cubic', 'spherical', 'exponential', 'gaussian', 'thin plate'}, optional
        Method to fit coefficients to
    b : array_like, shape (2,), optional
        Model coefficients b=[slope, nugget]

    Returns
    -------
    f : callable
        Callable (function of h) semi-variogram
    """

    semi = semi_variogram_callable(r, method)
    return partial(_co_variogram_fun, f=semi, b=b)


def _semi_variogram_system_matrix(h, r, method, nugget=True):
    """
    Return the system matrix used in fitting coefficients to a pre-defined semi-variogram method

    Parameters
    ----------
    h : array_like, shape (n,)
        Distance vector
    r : float
        Range (distance) at which data has no impact on variance
    method : {'linear', 'cubic', 'spherical', 'exponential', 'gaussian', 'thin plate'}, optional
        Method to fit coefficients to
    nugget : bool, optional
        True if a nugget should be fitted to the data

    Returns
    -------
    A : ndarray, shape (n,)
        System matrix
    """

    f = semi_variogram_callable(r, method)
    A = f(h)
    if nugget:
        A = np.hstack((np.ones_like(A), A))

    return np.atleast_2d(A).T


def semi_variogram_fit(h, v, r, method='gaussian', nugget=True):
    """
    Fit coefficients to a pre-defined semi-variogram model.

    Parameters
    ----------
    h : array_like, shape (n,)
        Distance vector
    v : array_like, shape (n,)
        Variogram vector
    r : float
        Range (distance) at which data has no impact on variance
    method : {'linear', 'cubic', 'spherical', 'exponential', 'gaussian', 'thin plate'}, optional
        Method to fit coefficients to
    nugget : bool, optional
        True if a nugget should be fitted to the data

    Returns
    -------
    b : ndarray, shape (2,)
        Fitted coefficients, b = [nugget, a]
    """

    A = _semi_variogram_system_matrix(h, r, method, nugget)
    res = lsq_linear(A, v)
    b = res.x

    if not nugget:
        b = (0., b[0])

    return b


def co_variogram_fit(h, v, r, method='gaussian', nugget=True):
    """
    Fit coefficients to a pre-defined co-variogram model.

    Parameters
    ----------
    h : array_like, shape (n,)
        Distance vector
    v : array_like, shape (n,)
        Variogram vector
    r : float
        Range (distance) at which data has no impact on variance
    method : {'linear', 'cubic', 'spherical', 'exponential', 'gaussian', 'thin plate'}, optional
        Method to fit coefficients to
    nugget : bool, optional
        True if a nugget should be fitted to the data

    Returns
    -------
    b : ndarray, shape (2,)
        Fitted coefficients, b = [a, nugget]
    """

    A = 1. - _semi_variogram_system_matrix(h, r, method, nugget)
    res = lsq_linear(A, v)
    b = res.x

    if not nugget:
        b = (1., b[0])

    return b
