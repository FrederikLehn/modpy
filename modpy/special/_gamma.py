import numpy as np

from modpy._util import split_set
from modpy.special._special_util import MACHEP, ROOT_2PI, LOG_PI
from modpy.special._erf import _ndtri


def gamma(z):
    """
    Gamma function valid in the entire complex plane. Accuracy is 15 significant digits along the real axis.
    This routine uses a Lanczos series approximation for the complex Gamma function.

    NOTE:
    Inspired from: https://en.wikipedia.org/wiki/Lanczos_approximation

    Parameters
    ----------
    z : float or array_like
        Gamma function parameter

    Returns
    -------
    y : array_like, shape
        Resulting value

    """

    z = np.atleast_1d(z)
    shape = z.shape
    z = z.flatten()
    p = np.argwhere(np.real(z) >= .5)
    p = np.reshape(p, (p.size,))
    p_ = np.setdiff1d(np.arange(z.size), p)

    c = np.array([676.5203681218851,
                  -1259.1392167224028,
                  771.32342877765313,
                  -176.61502916214059,
                  12.507343278686905,
                  -0.13857109526572012,
                  9.9843695780195716e-6,
                  1.5056327351493116e-7])

    y = np.zeros_like(z) + 0j  # cast to complex

    if p.size:
        z[p] -= 1.
        x = np.repeat(0.99999999999980993, p.size).astype(np.complex128)
        for (i, c_) in enumerate(c):
            x += c_ / (z[p] + i + (1. + 0j))
        t = z[p] + len(c) - 0.5
        y[p] = ROOT_2PI * t ** (z[p] + 0.5) * np.exp(-t) * x

    if p_.size:
        y[p_] = np.pi / (np.sin(np.pi * z[p_]) * gamma(1. - z[p_]))

    if np.all(np.imag(y) < 1e-7):
        y = np.real(y)

    return np.reshape(y, shape)


def gammaln(z):
    """
    Logarithm of the gamma function valid in both the negative and positive real plane.
    It is precise to a tolerance of 3 significant digits.

    NOTE:
    Inspired from: http://www.ee.ic.ac.uk/hp/staff/dmb/voicebox/mdoc/v_mfiles/v_gammalns.html

    Parameters
    ----------
    z : float or array_like
        Gamma function parameter

    Returns
    -------
    y : array_like
        Resulting value
    """

    z = np.atleast_1d(z)
    shape = z.shape
    z = z.flatten() + 0j  # cast to complex
    p = np.argwhere(z > 0.)
    p = np.reshape(p, (p.size,))
    p_ = np.setdiff1d(np.arange(z.size), p)

    y = np.zeros_like(z)

    if p.size:  # non-negative x-values
        y[p] = np.log(gamma(z[p]))

    f = np.intersect1d(p_, np.argwhere(z == np.fix(np.real(z))))  # find non-positive integers
    if f.size:
        y[f] = np.inf  # set output to infinity
        p_ = np.setdiff1d(p_, f)  # do not consider these values further

    if p_.size:  # negative x-values
        t = np.sin(np.pi * z[p_])
        y[p_] = LOG_PI - gammaln(1. - z[p_]) - np.log(t)

    if np.all(np.imag(y) < 1e-7):
        y = np.real(y)

    return np.reshape(y, shape)


def _chk_gammainc_inp(x, s):

    if isinstance(x, float):
        if isinstance(s, np.ndarray):
            x = np.repeat(x, s.size)
        else:
            x = np.atleast_1d(x)

    if np.any(x < 0.):
        raise ValueError('All values of `x` must satisfy in 0 <= x.')

    if isinstance(s, float):
        s = np.repeat(s, x.size)

    if np.any(s <= 0):
        raise ValueError('All values of `a` must be strictly positive numbers.')

    if np.any(x.shape != s.shape):
        raise ValueError('`x`, `s` must have the same shape.')

    return x.flatten(), s.flatten()


def gammainc(x, s):
    """
    Calculates the regularized lower incomplete gamma function, i.e.::

        \gamma(x; s) = 1/\Gamma(s)\int_0^x t^{s-1}e^{-t} \mathrm{d}t

    NOTE:
    Inspired by: https://people.sc.fsu.edu/~jburkardt/m_src/asa032/asa032.html

    Parameters
    ----------
    x : float or array_like, shape (n,)
        Variable to be evaluated
    s : float or array_like, shape (n,)
        Gamma function parameter

    Returns
    -------
    y : array_like, shape (n,)
        Resulting value
    """

    tol = 1.0E-08

    # Check the input.
    x, s = _chk_gammainc_inp(x, s)
    y = np.zeros_like(x)

    g = gammaln(s)
    arg = s * np.log(x) - x - g
    factor = np.exp(arg)

    i, i_ = split_set((x <= 1.) | (x < s))

    # Calculation by series expansion.
    y[i] = _gamma_series_exp(x[i], s[i], factor[i], tol=tol)

    # Calculation by continued fraction.
    y[i_] = _gamma_cont_frac(x[i_], s[i_], factor[i_], tol=tol)

    return y


def _gamma_series_exp(x, s, f, tol=1e-7):
    gin = 1.
    term = 1.
    rn = s.copy()

    while True:

        rn += 1.
        term = term * x / rn
        gin = gin + term

        if np.all(term <= tol):
            break

    return gin * f / s


def _gamma_cont_frac(x, s, f, tol=1e-7):
    oflo = 1.0E+37  # overflow value

    x = np.atleast_2d(x).T
    s = np.atleast_2d(s).T

    a = 1. - s
    b = a + x + 1.
    term = 0.

    pn = np.hstack((np.ones_like(x), x, x + 1., x * b, np.zeros_like(x), np.zeros_like(x)))
    gin = pn[:, 2] / pn[:, 3]

    while True:
        a += 1.
        b += 2.
        term += 1.
        an = a * term
        pn[:, 4:6] = b * pn[:, 2:4] - an * pn[:, 0:2]

        i = np.argwhere(pn[:, 5] != 0)
        rn = np.reshape(pn[i, 4] / pn[i, 5], (x.size,))
        dif = np.abs(gin[i] - rn[i])

        # absolute error tolerance satisfied?
        if np.all((dif <= tol) & (dif <= (tol * rn))):
            y = 1. - f * gin
            break

        gin[i] = rn[i]

        pn[i, 0:4] = pn[i, 2:6]
        i_ = np.argwhere(oflo <= np.abs(pn[i, 4]))
        pn[i_, 0:4] /= oflo

    return y


def gammaincc(x, s):
    """
    Calculates the upper part of the incomplete gamma function, i.e.::

        \gamma(x; s) = \int_x^{\inf} t^{s-1}e^{-t} \mathrm{d}t

    Parameters
    ----------
    x : float or array_like, shape (n,)
        Variable to be evaluated
    s : float or array_like, shape (n,)
        Gamma function parameter

    Returns
    -------
    y : array_like, shape (n,)
        Resulting value
    """

    return 1. - gammainc(x, s)


def gammaincinv(y, s):
    """
    Calculates the inverse of the regularized lower incomplete gamma function, i.e.::

        \gamma(x; s) = 1/\Gamma(s)\int_0^x t^{s-1}e^{-t} \mathrm{d}t

    NOTE:
    Inspired by: https://github.com/minrk/scipy-1/blob/master/scipy/special/c_misc/gammaincinv.c

    Parameters
    ----------
    y : float or array_like, shape (n,)
        Variable to be evaluated
    s : float or array_like, shape (n,)
        Gamma function parameter

    Returns
    -------
    y : array_like, shape (n,)
        Resulting value
    """

    return gammainccinv(1. - y, s)


def gammainccinv(y, s):
    """
    Calculates the inverse of the regularized upper incomplete gamma function, i.e.::

        \Gamma(x; s) = 1/\Gamma(s)\int_x^{\inf} t^{s-1}e^{-t} \mathrm{d}t

    NOTE:
    Inspired by: https://github.com/nearform/node-cephes/blob/master/cephes/igami.c

    Parameters
    ----------
    y : float or array_like, shape (n,)
        Variable to be evaluated
    s : float or array_like, shape (n,)
        Gamma function parameter

    Returns
    -------
    y : array_like, shape (n,)
        Resulting value
    """

    shape = y.shape
    y, s = _chk_gammainc_inp(y, s)

    tol = 1e-7
    max_iter = 10

    # Split to ignore calculations of y=0 and y=1
    i0, i1, ii = split_set((y == 0., y == 1.))

    y0 = y[ii]
    a = s[ii]
    x0 = np.repeat(np.inf, ii.size)
    y1 = np.zeros_like(ii, dtype=np.float64)
    x1 = np.zeros_like(ii, dtype=np.float64)
    yh = np.ones_like(ii, dtype=np.float64)

    # simple approximation of inverse
    d = 1. / 9. * a
    yy = (1. - d - _ndtri(y0) * np.sqrt(d))
    x = a * yy ** 3.

    gammaln_ = gammaln(a)

    # Newton iterations
    it = 0
    i_ = np.arange(ii.size)

    while it < max_iter:
        # if diverged outside bounding x-interval go to bisection
        i, i_ = split_set((x[i_] > x0[i_]) | (x[i_] < x1[i_]), idx=i_)
        x[i] = _gammainccinv_bisec(y0[i], a[i], x[i], x0[i], x1[i], y1[i], yh[i])

        yy[i_] = gammaincc(x[i_], a[i_])

        # if diverged outside bounding y-interval go to bisection
        i, i_ = split_set((yy[i_] < y1[i_]) | (yy[i_] > yh[i_]), idx=i_)
        x[i] = _gammainccinv_bisec(y0[i], a[i], x[i], x0[i], x1[i], y1[i], yh[i])

        # split according to direction relative to start-guess
        is1, is2 = split_set(yy[i_] < y0[i_], idx=i_)

        x0[is1] = x[is1]
        y1[is1] = yy[is1]

        x1[is2] = x[is2]
        yh[is2] = yy[is2]

        # calculate derivative
        d[i_] = (a[i_] - 1.) * np.log(x[i_]) - x[i_] - gammaln_[i_]
        d[i_] = -np.exp(d[i_])

        # calculate the next step
        d[i_] = (yy[i_] - y0[i_]) / d[i_]

        if np.all(np.abs(d[i_] / x[i_]) < tol):
            break

        x[i_] -= d[i_]

        it += 1

    x_ = np.zeros_like(y)
    x_[i0] = np.inf
    x_[ii] = x
    return np.reshape(x_, shape)


def _gammainccinv_bisec(y, s, x, x0, x1, y1, yh):
    """
    Performs a bisection interval halving as part of solving the inverse of the upper incomplete gamma function

    NOTE:
    Inspired by: https://github.com/nearform/node-cephes/blob/master/cephes/igami.c

    Parameters
    ----------
    y : float or array_like, shape (n,)
        Variable to be evaluated
    s : float or array_like, shape (n,)
        Gamma function parameter

    Returns
    -------
    y : array_like, shape (n,)
        Resulting value
    """

    # if input is empty, return
    if not y.size:
        return x

    dithresh = 5. * MACHEP

    max_iter = 400
    it = 0

    yy = np.zeros_like(y, dtype=np.float64)
    i, _ = split_set(x0 == np.inf)
    x[i] = np.where(x[i] <= 0., 1., x[i])
    d = np.repeat(.0625, i.size)

    while np.any(x0[i] == np.inf):
        x[i] *= (1. + d[i])
        yy[i] = gammaincc(x[i], s[i])

        i, i_ = split_set(yy[i] >= y[i])
        x0[i_] = x[i_]
        y1[i_] = yy[i_]

        d[i] *= 2.

    d = np.repeat(.5, y.size)
    di = np.zeros_like(y, dtype=np.int64)

    ii = np.arange(y.size)

    while it < max_iter:
        x[ii] = x1[ii] + d[ii] * (x0[ii] - x1[ii])
        yy[ii] = gammaincc(x[ii], s[ii])

        # remove converged elements
        lgm = (x0[ii] - x1[ii]) / (x1[ii] + x0[ii])
        ii, _ = split_set(np.abs(lgm) >= dithresh, idx=ii)
        if not len(ii):
            break

        lgm = (yy[ii] - y[ii]) / y[ii]
        ii, _ = split_set(np.abs(lgm) >= dithresh, idx=ii)
        if not len(ii):
            break

        ii, _ = split_set(x[ii] > 0., idx=ii)
        if not len(ii):
            break

        # split yy > y and below
        i, i_ = split_set(yy[ii] >= y[ii], idx=ii)

        x1[i] = x[i]
        yh[i] = yy[i]

        x0[i_] = x[i_]
        y1[i_] = yy[i_]

        # yy > y --------------------------------------
        # within yy > y, split di < 0, di > 1 and else
        id1, id2, id3 = split_set((di[i] < 0, di[i] > 1), idx=i)

        # yy > y & di < 0
        di[id1] = 0
        d[id1] = .5

        # yy > y & di < 1
        d[id2] = .5 * d[id2] + .5

        # yy > y else
        d[id3] = (y[id3] - y1[id3]) / (yh[id3] - y1[id3])

        d[i] += 1

        # yy <= y -------------------------------------
        # within yy <= y, split di > 0, di < -1 and else
        id1, id2, id3 = split_set((di[i_] > 0, di[i_] < -1), idx=i_)

        # yy <= y & di > 0
        di[id1] = 0
        d[id1] = .5

        # yy <= y & di < -1
        d[id2] *= .5

        # yy <= y & else
        d[id3] = (y[id3] - y1[id3]) / (yh[id3] - y1[id3])

        di[i_] -= 1

        it += 1

    return x




