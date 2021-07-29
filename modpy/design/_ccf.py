import numpy as np

from modpy.design._mono_sens import mono
from modpy.design._full_factorial import ff2n
from modpy.design._design_util import _scale_factorial_design


def ccf_design(joint, scale=False):
    """
    Create a cubic centered face design sampling 2 ** n + 2n + 1 values. It is a combination of the full factorial and
    mono-sensitivity design, sampling all vertices as well as cell faces and a reference point.

    Parameters
    ----------
    joint : JointDistribution
        Class JointDistribution.
    scale : boolean
        If true, scale design to the support (and possibly reference value) of the distributions

    Returns
    -------
    design : array_like, shape (2 ** n + 2n + 1, n)
        Design matrix of the form 'number of runs' x 'number of variables'.

    """

    D = ccf(len(joint.marginals))

    if scale:
        D = _scale_factorial_design(D, joint)

    return D


def ccf(n):
    """
    Create a cubic centered face design sampling 2 ** n + 2n + 1 values. It is a combination of the full factorial and
    mono-sensitivity design, sampling all vertices as well as cell faces and a reference point.

    Parameters
    ----------
    n : int
        Number of variables in the experiment

    Returns
    -------
    D : array_like, shape (2 ** n + 2n + 1, n)
        Design matrix of the form 'number of runs' x 'number of variables'.
    """

    return np.vstack((mono(n), ff2n(n)))
