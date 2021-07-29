import numpy as np
from numpy.fft import fft, ifft


def auto_correlation(x, lags):
    """
    Calculates the auto-correlation of `x`.

    References
    ----------
    [1] https://stackoverflow.com/questions/643699/how-can-i-use-numpy-correlate-to-do-autocorrelation

    Parameters
    ----------
    x : array_like, shape (n,)
        Set of variables to calculate the autocorrelation for.
    lags : int
        Number of lags for which to calculate the auto-correlation

    Returns
    -------
    corr : array_like, shape (m,)
        Autocorrelation.
    """

    n = len(x)
    es = 2 * n - 1
    fs = 2 ** np.ceil(np.log2(es)).astype(np.int64)

    xp = x - np.mean(x)
    var = np.var(x)

    # do fft and ifft
    cf = fft(xp, fs)
    corr = ifft(cf.conjugate() * cf).real
    corr = corr / var / n

    return corr[:lags]


def auto_correlation_time(x):
    """
    Calculates the auto-correlation lag time of a process which auto-correlation asymptotically reduces from 1 to 0,
    as is the case for MCMC sampling processes. It uses the method of 'Match Means' which is simple and performs
    almost as well as the much more advanced auto-regression fit method.

    References
    ----------
    [1] Thompson, M. B. (2010). "A Comparison of Methods for Computing Autocorrelation time". University of Toronto.
        https://arxiv.org/pdf/1011.0175.pdf

    Parameters
    ----------
    x : array_like, shape (n,)
        Set of variables to calculate the autocorrelation for.

    Returns
    -------
    tau : int
        Time at which the auto-correlation has asymptotically reduced to 0.
    """

    n = x.size
    b = int(n ** (1. / 3.))
    m = int(n ** (2. / 3.))

    means = [np.mean(x[(i * m): ((i + 1) * m)]) for i in range(b - 1)]
    means.append(np.mean(x[b * m:]))

    s2 = np.var(x)
    s2m = np.var(means)

    return max(1, int(m * s2m / s2))


def ess(x):
    """
    Calculate the effective sample size (ESS) of a Markov chain `x`.

    Parameters
    ----------
    x : array_like, shape (n, m)
        Markov chain.

    Returns
    -------
    ESS : array_like, shape (m,)
        Effective sample size per dimension.
    """

    n, m = x.shape
    return [n / auto_correlation_time(x[:, i]) for i in range(m)]


def moving_average(x, m=None):
    """
    Calculates the moving average of an array `x` with the period `m`.

    Parameters
    ----------
    x : array_like, shape (n,)
        Vector of results.
    m : int
        Period over which to average values. If `m` is None, then m=x.size.

    Returns
    -------
    ma : array_like, shape (n,)
        Moving average of `x` with period `m`.
    """

    n = x.size

    if m is None:
        m = n
    else:
        m = int(m)

    # boundary effects are handled by using as many points as possible until complete overlap is reached.
    ma = np.array([np.mean(x[:i]) for i in range(m)])

    # simplify remaining computation by np.convolve
    return np.append(ma, np.convolve(x, np.ones((m,)), mode='valid') / m)
