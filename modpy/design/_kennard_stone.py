from modpy._util import point_distance
from modpy.design._design_util import _scale_sfd_design, _initial_sfd_samples, _max_min_dist_split


def kennard_stone_design(joint, k, m, scale=False, ord_=None, seed=None):
    """
    Create a Kennard-Stone space-filling design with `k` samples from an initial sample pool of size `m`.

    Parameters
    ----------
    joint : JointDistribution
        Class JointDistribution.
    k : int
        Number of final samples in design.
    m : int
        Number of initial samples to draw from.
    scale : boolean
        If true, scale design to the support of the distributions.
    ord_ : int
        Order of the distance norm, default to Euclidian.
    seed : int
        Seed of the random generator.

    Returns
    -------
    D : array_like, shape (samples, n)
        Design matrix of the form 'number of experiments' x 'number of variables'.
    """

    if k > m:
        raise ValueError('`k` has to satisfy k<=m.')

    n = len(joint.marginals)
    D, _ = ksd(_initial_sfd_samples(n, m, seed=seed), k, ord_=ord_)

    if scale:
        D = _scale_sfd_design((D + 1.) / 2., joint)

    return D


def ksd(X, k, ord_=None):
    """
    Create a Kennard-Stone space-filling design with `k` samples drawn from a sample pool `X`

    NOTE: Based on https://hxhc.github.io/post/kennardstone-spxy/

    Parameters
    ----------
    X : array_like, shape (n, m)
        Initial sample pool.
    k : int
        Number of final samples in design.
    ord_ : int
        Order of the distance norm, default to Euclidian.

    Returns
    -------
    design : array_like, shape (k, m)
        Design matrix of the form 'number of experiments' x 'number of variables'.
    idx : array_like, shape (k,)
        Indices of samples to pick from `X`.
    """

    idx, _ = _max_min_dist_split(point_distance(X, ord_=ord_), k)
    return X[idx, :], idx
