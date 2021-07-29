import numpy as np

from modpy.special._gamma import gammaln


def beta(a, b):
    return np.exp(gammaln(a) + gammaln(b) - gammaln(a + b))


def _chk_betainc_inp(x, a, b):

    if isinstance(x, float):
        if isinstance(a, np.ndarray):
            x = np.repeat(x, a.size)
        elif isinstance(b, np.ndarray):
            x = np.repeat(x, b.size)
        else:
            x = np.atleast_1d(x)

    if np.any((x < 0.) & (1. < x)):
        raise ValueError('All values of `x` must satisfy in 0 <= x <= 1.')

    if isinstance(a, float):
        a = np.repeat(a, x.size)

    if np.any(a <= 0):
        raise ValueError('All values of `a` must be strictly positive numbers.')

    if isinstance(b, float):
        b = np.repeat(b, x.size)

    if np.any(b <= 0):
        raise ValueError('All values of `b` must be strictly positive numbers.')

    if not ((x.shape == a.shape) and (x.shape == b.shape)):
        raise ValueError('`x`, `a` and `b` must have the same shape.')

    return x, a, b


def betainc(x, a, b):
    """
    Calculates the incomplete beta function, i.e.::

        B(x; a, b) = 1 / B(a, b) * \int_0^x t^{a-1}(1-t)^{b-1} dt

    NOTE:
    Inspired by: https://malishoaib.wordpress.com/2014/04/15/the-beautiful-beta-functions-in-raw-python/

    Parameters
    ----------
    x : float or array_like, shape (n,)
        Variable to be evaluated
    a : float or array_like, shape (n,)
        First parameter of beta function
    b : float or array_like, shape (n,)
        Second parameter of beta function

    Returns
    -------
    y : array_like, shape (n,)
        Resulting value
    """

    x, a, b = _chk_betainc_inp(x, a, b)

    y = np.zeros_like(x)

    i = np.argwhere(x < (a + 1.) / (a + b + 2.))
    i_ = np.setdiff1d(np.arange(x.size), i)

    y[i] = np.exp(betaincln(x[i], a[i], b[i])) * _beta_cont_frac(x[i], a[i], b[i]) / a[i]
    y[i_] = 1. - np.exp(betaincln(x[i_], a[i_], b[i_])) * _beta_cont_frac(1. - x[i_], b[i_], a[i_]) / b[i_]

    y = np.where(x == 1., 1., y)

    return y


def betaln(a, b):
    """
    Calculates the logarithm of the beta function.

    Parameters
    ----------
    a : float or array_like, shape (n,)
        First parameter of beta function
    b : float or array_like, shape (n,)
        Second parameter of beta function

    Returns
    -------
    y : array_like, shape (n,)
        Resulting value
    """

    return np.log(np.abs(beta(a, b)))


def betaincln(x, a, b):
    """
    Calculates the logarithm of the incomplete beta function, i.e.::

        ln(B(x; a, b)) = ln(1 / B(a, b) * \int_0^x t^{a-1}(1-t)^{b-1} dt)

    NOTE:
    Inspired by: https://malishoaib.wordpress.com/2014/04/15/the-beautiful-beta-functions-in-raw-python/

    Parameters
    ----------
    x : float or array_like, shape (n,)
        Variable to be evaluated
    a : float or array_like, shape (n,)
        First parameter of beta function
    b : float or array_like, shape (n,)
        Second parameter of beta function

    Returns
    -------
    y : array_like, shape (n,)
        Resulting value
    """

    return gammaln(a + b) - gammaln(a) - gammaln(b) + a * np.log(x) + b * np.log(1. - x)


def _beta_cont_frac(x, a, b, tol=1e-6, max_iter=200):
    """
    Calculates continued fraction of the incomplete beta function.

    NOTE:
    Inspired by: https://malishoaib.wordpress.com/2014/04/15/the-beautiful-beta-functions-in-raw-python/

    Parameters
    ----------
    x : array_like, shape (n,)
        Variable to be evaluated
    a : float
        First parameter of beta function
    b : float
        Second parameter of beta function

    Returns
    -------
    y : array_like, shape (n,)
        Resulting value
    """

    bm = az = am = np.ones_like(x)
    qab = a + b
    qap = a + 1.
    qam = a - 1.
    bz = 1. - qab * x / qap

    i = 0
    while i < max_iter:
        em = float(i + 1)
        tem = em + em
        d = em * (b - em) * x / ((qam + tem) * (a + tem))
        ap = az + d * am
        bp = bz + d * bm
        d = -(a + em) * (qab + em) * x / ((qap + tem) * (a + tem))
        app = ap + d * az
        bpp = bp + d * bz
        aold = az
        am = ap / bpp
        bm = bp / bpp
        az = app / bpp
        bz = 1.

        if np.all(np.abs(az - aold) < (tol * np.abs(az))):
            return az

        i += 1

    raise ValueError('Unable to converge in maximum number of iterations.')


def betaincinv(y, a, b):
    """
    Calculates the inverse of the incomplete beta function, i.e.::

        B(x a, b) = 1 / B(a, b) * \int_0^x t^{a-1}(1-t)^{b-1} dt

    solving for x as the upper bound of the integral using a Newton method

    NOTE:
    Inspired from: https://people.sc.fsu.edu/~jburkardt/f_src/asa109/asa109.html

    Parameters
    ----------
    y : float or array_like, shape (n,)
        Variable to be evaluated
    a : float or array_like, shape (n,)
        First parameter of beta function
    b : float or array_like, shape (n,)
        Second parameter of beta function

    Returns
    -------
    x : array_like, shape (n,)
        Resulting value
    """

    y, a, b = _chk_betainc_inp(y, a, b)

    sae = -30.
    fpu = 10. ** sae

    shape = y.shape
    x = y.flatten()
    yy = y.flatten()
    aa = a.flatten()
    bb = b.flatten()

    # change tail
    it = np.argwhere(y > .5)  # index of tail
    yy[it] = 1. - y[it]
    aa[it] = b[it]
    bb[it] = a[it]

    #  Calculate the initial approximation
    r = np.sqrt(-np.log(yy ** 2.))
    z = r - (2.30753 + 0.27061 * r) / (1. + (0.99229 + 0.04481 * r) * r)

    i = np.argwhere((aa > 1.) & (bb > 1.))
    i_ = np.setdiff1d(np.arange(y.size), i)
    s = t = h = w = np.zeros_like(y)

    r[i] = (z[i] ** 2. - 3.) / 6.
    s[i] = 1. / (2. * aa[i] - 1.)
    t[i] = 1. / (2. * bb[i] - 1.)
    h[i] = 2. / (s[i] + t[i])
    w[i] = z[i] * np.sqrt(h[i] + r[i]) / h[i] - (t[i] - s[i]) * (r[i] + 5. / 6. - 2. / (3. * h[i]))
    x[i] = aa[i] / (aa[i] + bb[i] * np.exp(2. * w[i]))

    r[i_] = 2. * bb[i_]
    t[i_] = 1. / (9. * bb[i_])
    t[i_] = r[i_] * (1. - t[i_] + z[i_] * np.sqrt(t[i_])) ** 3.

    betaln_ = betaln(a, b)
    t[i_] = np.where(t[i_] <= 0., t[i_], (4. * aa[i_] + r[i_] - 2.) / t[i_])
    x[i_] = np.where(t[i_] <= 0., 1. - np.exp((np.log((1. - yy[i_]) * bb[i_]) + betaln_[i_]) / bb[i_]),
            np.where(t[i_] <= 1., np.exp((np.log(yy[i_] * aa[i_]) + betaln_[i_]) / aa[i_]),
                     1. - 2. / (t[i_] + 1.)))

    #  solve for x by a modified Newton-Raphson method, using the function betainc.
    r = 1. - aa
    t = 1. - bb
    zprev = np.zeros_like(y)
    sq = prev = np.ones_like(y)

    # truncate initial value away from boundaries
    x = np.clip(x, 0.0001, 0.9999)

    iex = np.maximum(-5. / aa / aa - 1. / yy ** 0.2 - 13., sae)
    acu = 10. ** iex

    # iteration loop.
    while True:

        z = betainc(x, aa, bb)
        xin = x
        z = (z - yy) * np.exp(betaln_ + r * np.log(xin) + t * np.log(1. - xin))

        prev = np.where(z * zprev <= 0., np.maximum(sq, fpu), prev)

        g = 1.

        while True:

            while True:

                adj = g * z
                sq = adj ** 2.

                tx = np.where(sq < prev, x - adj, x)

                if np.all((0. <= tx) & (tx <= 1.)):
                    break

                g /= 3.

            #  check whether current estimate is acceptable.
            #  the change "x = tx" was suggested by Ivan Ukhov.
            x = np.where((prev <= acu) & (z ** 2. <= acu), tx, x)

            if np.all((tx != 0.) & (tx != 1.0)):
                break

            g /= 3.

        if np.all(tx == x):
            break

        x = tx
        zprev = z

    x[it] = 1. - x[it]

    return np.reshape(x, shape)
