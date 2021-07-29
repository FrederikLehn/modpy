import numpy as np

from modpy.design._design_util import _scale_factorial_design


def mono_sensitivity_design(joint, scale=False):
    """
    Create a mono-sensitivity design sampling 2n+1 values.
    1 run for the reference and 2n for the min/max of each variable.

    Parameters
    ----------
    joint : JointDistribution
        Class JointDistribution.
    scale : boolean
        If true, scale design to the support (and possibly reference value) of the distributions

    Returns
    -------
    design : array_like, shape (2n+1, n)
        Design matrix of the orm 'number of runs' x 'number of variables'.
    """

    D = mono(len(joint.marginals))

    if scale:
        D = _scale_factorial_design(D, joint, level=2)

    return D


def mono(n):
    """
    Create a mono-sensitivity design sampling 2n+1 values.
    1 run for the reference and 2n for the min/max of each variable.

    Parameters
    ----------
    n : int
        Number of variables in the experiment

    Returns
    -------
    D : array_like
        Design matrix
    """

    D = np.zeros((2 * n + 1, n))

    i = 0
    j = 1
    for _ in range(n):
        D[j, i] = -1
        D[j + 1, i] = 1
        i += 1
        j += 2

    return D
