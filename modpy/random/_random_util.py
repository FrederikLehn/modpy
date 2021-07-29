import numpy as np

from modpy.special import EXP, log


def _chk_dist_inp(x, bounds=(-np.inf, np.inf)):
    x_ = np.atleast_1d(x)

    if not (x_.ndim == 1):
        raise ValueError('`x` must be a 1D array.')

    if np.any((x_ < bounds[0]) | (x_ > bounds[1])):
        raise ValueError('All values of `x` must be in the interval {}.'.format(bounds))


def _chk_invdist_inp(p):
    if np.any((1. < p) | (p < 0.)):
        raise ValueError('All values of `p` must satisfy in 0 <= p <= 1.')


def _chk_mmm_inp(a, b, c=None):
    if c is None:
        if not (a < b):
            raise ValueError('`a` and `b` must satisfy a<b ({}<{}).'.format(a, b))
    else:
        if not (a < c < b):
            raise ValueError('`a`, `b` and `c` must satisfy a<c<b ({}<{}<{}).'.format(a, c, b))


def _chk_log_mmm_inp(a, b, c=None):
    if (a <= 0.) or (b <= 0.):
        raise ValueError('`a` and `b` must satisfy 0>a and 0>b.')

    _chk_mmm_inp(a, b, c)


def _chk_root_mmm_inp(a, b, c=None):
    if (a < 0.) or (b < 0.):
        raise ValueError('`a` and `b` must satisfy 0>=a and 0>=b.')

    _chk_mmm_inp(a, b, c)


def _chk_normal_inp(mu, sigma):
    # if mu < 0.:
    #     raise ValueError('`mu` must satisfy mu>=0.')

    if sigma <= 0.:
        raise ValueError('`sigma` must satisfy sigma>0.')


def _chk_beta_inp(alpha, beta_):
    if (alpha <= 0.) or (beta_ <= 0.):
        raise ValueError('`alpha` and `beta` must satisfy alpha>0 and beta>0.')


def _chk_prob_inp(p1, v1, p2, v2):
    if (p1 < 0.) or (p2 < 0.):
        raise ValueError('Probabilities `p1` and `p2` must satisfy p>=0.')

    if p1 >= p2:
        raise ValueError('Probabilities must satisfy p1<p2.')

    if v1 == v2:
        raise ValueError('Values must satisfy v1!=v2.')


def _chk_exp_inp(lam):
    if lam <= 0.:
        raise ValueError('`lam` must satisfy lam>0 ({})'.format(lam))