import numpy as np

from modpy.special import log, sqrt


def _chk_lin(a, b):
    if b <= a:
        raise ValueError('`b` ({}) must be greater than `a` ({}).'.format(b, a))


def _chk_log(a, base):
    if a <= 0.:
        raise ValueError('`a` ({}) must be greater than 0.'.format(a))

    if base <= 0:
        raise ValueError('`base` ({}) must be greater than 0.'.format(base))


def _chk_root(root):
    if root <= 0:
        raise ValueError('`root` ({}) must be greater than 0.'.format(root))


def _get_bounds(x, a=None, b=None):
    if a is None:
        a = np.amin(x)

    if b is None:
        b = np.amax(x)

    return a, b


def linspace(a, b, n):
    _chk_lin(a, b)
    return np.linspace(a, b, n)


def logspace(a, b, n, base=10.):
    _chk_log(a, base)
    return base ** linspace(log(a, base), log(b, base), n)


def rootspace(a, b, n, root=2.):
    _chk_root(root)
    return linspace(sqrt(a, root), sqrt(b, root), n) ** root


def lin2log(x, a=None, b=None, base=10.):
    a, b = _get_bounds(x, a, b)
    _chk_lin(a, b)

    loga = log(a, base)
    logb = log(b, base)

    return base ** (loga + (x - a) / (b - a) * (logb - loga))


def log2lin(x, a=None, b=None, base=10.):
    a, b, = _get_bounds(x, a, b)
    _chk_lin(a, b)
    _chk_log(a, base)

    loga = log(a, base)
    logb = log(b, base)

    return (log(x, base) - loga) * (b - a) / (logb - loga) + a


def lin2root(x, a=None, b=None, root=2.):
    a, b, = _get_bounds(x, a, b)
    _chk_lin(a, b)

    roota = sqrt(a, root)
    rootb = sqrt(b, root)

    return (roota + (x - a) / (b - a) * (rootb - roota)) ** root


def root2lin(x, a=None, b=None, root=2.):
    a, b, = _get_bounds(x, a, b)
    _chk_lin(a, b)
    _chk_root(root)

    roota = sqrt(a, root)
    rootb = sqrt(b, root)

    return (sqrt(x, root) - roota) * (b - a) / (rootb - roota) + a
