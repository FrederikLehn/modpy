import numpy as np
import numpy.linalg as la


def sign(x):
    return (x >= 0).astype(float) * 2 - 1


def range_(a, b):
    return b - a


def scale_translate(x, a, b):
    return x * range_(a, b) + a


def normalize(x, a=None, b=None):
    if a is None:
        a = x.min()

    if b is None:
        b = x.max()

    return (x - a) / range_(a, b)


def point_distance(x, ord_=None):
    """
    Creates an array of distances between all points in x.

    Given a point vector of length n, the function returns an
    n-by-n array h of distances such that h[i,j] = norm(x[j]-x[i])

    Parameters
    ----------
    x : array_like, shape (n, m)
        Point vector
    ord_ : {non-zero int, inf, -inf, ‘fro’, ‘nuc’}, optional
        Input to np.linalg.norm

    Returns
    -------
    H : ndarray, shape (n, n)
        Distance matrix
    """

    if not (0 < x.ndim <= 2):
        raise ValueError('`x` must be either a 1D or 2D array.')

    x = np.atleast_2d(x)
    n, _ = x.shape
    H = np.zeros((n, n))

    for i in range(n):
        H[:, i] = la.norm(x-x[i, :], axis=1, ord=ord_)

    return H


def split_set(conds, idx=None):
    """
    Split a set of indices into multiple sub-sets based on a list of conditions

    Parameters
    ----------
    conds : list, tuple or array_like
        A set of conditions (array_like boolean arrays of same length)
    idx : list, tuple or array_like, optional
        Existing set of indices to sub-index

    Returns
    -------
    sets : tuple
        A tuple of array_like with integers which are sub-indexes to another array
    """

    if isinstance(conds, np.ndarray):
        conds = (conds,)

    shape = conds[0].shape
    ndim = conds[0].ndim
    n = conds[0].size
    sets = [np.empty(0, dtype=np.int64) for _ in conds]

    if idx is None:
        fs = np.arange(n)
    else:
        if (idx.ndim != ndim) or (np.any(idx.shape != shape)):
            raise ValueError('`idx` must have the same shape as the conditions.')

        fs = idx

    for i, cond in enumerate(conds):
        if (cond.ndim != ndim) or np.any(cond.shape != shape):
            raise ValueError('All conditions must have the same shape.')

        a = np.argwhere(cond)
        fs = np.setdiff1d(fs, a)

        sets[i] = a

    sets.append(np.asarray(fs, dtype=np.int64))  # accommodate remaining elements
    return (s.flatten() for s in sets)


def where(mask, x, f, g, fargs=(), gargs=()):
    """
    An alternative to `np.where` which does not evaluate the function `f` at `x` which does not satisfy the condition.

    Parameters
    ----------
    mask : array_like
        A mask of `x`, where True values are passed to `f` and False values are passed to `g`.
    x : array_like
        Value at which to evaluate `f` or `g`.
    f : callable
        Function used for `x` values where mask is True.
    g : callable
        Function used for `x` values where mask is False.
    fargs : tuple
        Additional arguments to `f`.
    gargs : tuple
        Additional arguments to `g`.
    Returns
    -------
    y : array_like
        The resulting combination of calls to `f` and `g`.
    """

    # prepare additional arguments
    fargs_ = [fa[mask] if isinstance(fa, np.ndarray) else fa for fa in fargs]
    gargs_ = [ga[~mask] if isinstance(ga, np.ndarray) else ga for ga in gargs]

    y = np.empty_like(x)
    y[mask] = f(x[mask], *fargs_)
    y[~mask] = g(x[~mask], *gargs_)

    return y


def convert_input(x):
    """
    Converts the input `x` to an array if it is a float/int, otherwise returns `x`.

    Parameters
    ----------
    x : int, float or array_like
        Input to a function which operates on numpy arrays.

    Returns
    -------
    x : array_like
        Input as a numpy array
    """

    if isinstance(x, float):

        return np.array([x], dtype=np.float64)

    elif isinstance(x, int):

        return np.array([x], dtype=np.int64)

    else:

        return x


def convert_output(xi, xu):
    """
    Converts the output `xu` to the same type as `xi`.

    Parameters
    ----------
    xi : int, float or array_like
        Input to a function which operates on numpy arrays.
    xu : array_like
        Output to a function which operates on numpy arrays.

    Returns
    -------
    xu : int, float or array_like
        Output in the same type as `xi`.
    """

    if isinstance(xi, float):

        return float(xu[0])

    elif isinstance(xi, int):

        return int(xu[0])

    else:

        return xu
