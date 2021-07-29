import numpy as np

from modpy.design._design_util import _scale_factorial_design


def full_factorial_design(joint, level=2, scale=False):
    """
    Create a full factorial design sampling 2 ** n values, one for each vertex of the hypercube.

    Parameters
    ----------
    joint : JointDistribution
        Class JointDistribution.
    level : int
        Level of investigation, 2: min & max, 3: min, mode & max
    scale : boolean
        If true, scale design to the support (and possibly reference value) of the distributions

    Returns
    -------
    design : array_like, shape (2 ** n, n)
        Design matrix of the form 'number of runs' x 'number of variables'.
    """

    if level not in (2, 3):
        raise ValueError('`level` must be either 2 or 3.')

    D = ff2n(len(joint.marginals))

    if scale:
        D = _scale_factorial_design(D, joint, level=level)

    return D


def fullfact(levels):
    """
    Create a full factorial design sampling an arbitrary number of levels per factor.

    NOTE: Inspired from https://github.com/tirthajyoti/Design-of-experiment-Python/blob/master/doe_factorial.py

    Parameters
    ----------
    levels : array_like
        Array of levels at which to sample each factor

    Returns
    -------
    D : array_like
        Design matrix
    """

    k = len(levels)  # number of factors
    ne = np.prod(levels)  # number of experiments
    D = np.zeros((ne, k))

    lvl_rep = 1
    ran_rep = ne
    for i in range(k):

        ran_rep //= levels[i]  # integer floor division
        lvl = []

        for j in range(levels[i]):
            lvl += [j] * lvl_rep

        D[:, i] = lvl * ran_rep
        lvl_rep *= levels[i]

    return D


def ff2n(n):
    """
    Create a full factorial design with 2 levels sampling `n` variables.

    Parameters
    ----------
    n : int
        Number of variables in the experiment

    Returns
    -------
    D : array_like
        Design matrix
    """

    return 2 * fullfact([2] * n) - 1
