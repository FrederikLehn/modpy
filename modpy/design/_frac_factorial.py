import numpy as np


def full_factorial_design(dists, resolution, norm=False):
    """
    Create a 2-level fractional factorial design with a given resolution. Assumption made that the number of required
    coefficients for the design matrix are linear terms and 1st-order 2-way interaction terms.

    Parameters
    ----------
    dists : tuple or list
        Tuple or list of class Distribution.
    resolution : int
        Resolution of the design
    norm : boolean
        If true, normalize design to min=-1, reference=0, maximum=1

    Returns
    -------
    design : array_like, shape (??, n)
        Design matrix of the form 'number of runs' x 'number of variables'.
    """

    n = len(dists)  # factors
    m = 2 ** n  # number of experiments for a full factorial
    ne = int(1. + .5 * n + .5 * n ** 2.)  # number of experiments (TODO: reduce based on resolution?)
