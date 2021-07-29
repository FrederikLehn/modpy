import numpy as np
from numpy.random import Generator, PCG64

from modpy.design._design_util import _normalize_sfd_design


def latin_hypercube_design(joint, samples, scale=False, smooth=True, seed=None):
    """
    Create a latin-hypercube design with a given amount of samples.

    NOTE: Based on the description of https://mathieu.fenniak.net/latin-hypercube-sampling/

    Parameters
    ----------
    joint : JointDistribution
        Class JointDistribution.
    samples : int
        Number of experiments
    scale : boolean
        If true, scale design to the support (and possibly reference value) of the distributions
    smooth : boolean
        If true, sample the inverse cum. function at equal spaces, otherwise use uniform sampling in intervals.
    seed : int
        Seed of the random number generator.

    Returns
    -------
    design : array_like, shape (samples, n)
        Design matrix of the form 'number of experiments' x 'number of variables'.
    """

    # prepare random generator
    gen = Generator(PCG64(seed))

    n = len(joint.marginals)
    seg_size = 1. / samples

    # stratified sampling
    S = np.zeros((samples, n))
    min_range = np.arange(0., 1., seg_size)
    max_range = min_range + seg_size

    for i, d in enumerate(joint.marginals):
        if smooth:
            p = (min_range + max_range) / 2.
        else:
            p = gen.uniform(min_range, max_range)

        S[:, i] = d.ppf(p)

    # grouping
    D = np.zeros_like(S)
    for i, d in enumerate(joint.marginals):
        D[:, i] = gen.permutation(S[:, i])

    if not scale:
        D = _normalize_sfd_design(D, joint)

    return D
