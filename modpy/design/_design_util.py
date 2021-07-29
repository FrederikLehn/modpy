import numpy as np
from numpy.random import Generator, PCG64

from modpy._util import split_set, scale_translate, normalize
from modpy.special import sqrt


def _scale_factorial_design(D, joint, level=2):
    """
    Scale a normalized factorial design to the actual sample values based on the supplied distributions.

    Parameters
    ----------
    D : array_like
        Design matrix
    joint : JointDistribution
        Class JointDistribution.
    level : int
        Level of investigation, 2: min & max, 3: min, mode & max

    Returns
    -------
    D : array_like
        Design matrix scaled to the bounds of the distribution
    """

    if level not in (2, 3):
        raise ValueError('`level` must be either 2 or 3.')

    for i, d in enumerate(joint.marginals):
        a = d.minimum()
        b = d.maximum()

        if (a in (-np.inf, np.inf)) or (b in (-np.inf, np.inf)):
            raise ValueError('Unable to create a design without finite support.')

        if level == 2:
            ia, ib = split_set(D[:, i] == -1)

        else:
            ia, ib, ic = split_set((D[:, i] == -1, D[:, i] == 1))
            D[ic, :] = d.reference()

        D[ia, i] = a
        D[ib, i] = b

    return D


def _scale_sfd_design(D, joint):
    """
    Scale a normalized space-filling design to the actual sample values based on the supplied distributions.

    Parameters
    ----------
    D : array_like
        Design matrix
    joint : JointDistribution
        Class JointDistribution.

    Returns
    -------
    D : array_like
        Design matrix scaled to the bounds of the distribution
    """

    for i, d in enumerate(joint.marginals):
        a = d.minimum()
        b = d.maximum()

        if (a in (-np.inf, np.inf)) or (b in (-np.inf, np.inf)):
            raise ValueError('Unable to create a design without finite support.')

        D[:, i] = scale_translate(D[:, i], a, b)

    return D


def _normalize_sfd_design(D, joint):
    """
    Normalize a scaled space-filling design to fit within the unit hypercube [-1, 1]^n.

    Parameters
    ----------
    D : array_like
        Design matrix
    joint : JointDistribution
        Class JointDistribution.

    Returns
    -------
    D : array_like
        Design matrix scaled to the bounds of the distribution
    """

    for i, d in enumerate(joint.marginals):
        a = d.minimum()
        b = d.maximum()

        if (a in (-np.inf, np.inf)) or (b in (-np.inf, np.inf)):
            raise ValueError('Unable to create a design without finite support.')

        D[:, i] = 2. * normalize(D[:, i], a, b) - 1.

    return D


def _initial_sfd_samples(n, m, seed=None):
    """
    Generate an initial ensemble of `m` points in the unit-hypercube [-1, 1]^n for a space-filling design.

    Parameters
    ----------
    n : int
        Dimension of hypercube
    m : int
        Number of samples
    seed : int
        Seed of the random generator.
    Returns
    -------
    X : array_like, shape (m, n)
        Ensemble of initial points for a space-filling design
    """

    return Generator(PCG64(seed)).uniform(-1., 1., size=(m, n))  #np.random.uniform(-1., 1., size=(m, n))


def _max_min_dist_split(D, k):
    """
    Split a set of points into two groups. A sub-set and a remaining set of points based on a max-min-distance approach.

    NOTE: Based on https://hxhc.github.io/post/kennardstone-spxy/

    Parameters
    ----------
    D : array_like
        Matrix of distances between points
    k : int
        Number of samples to draw from the matrix

    Returns
    -------
    sub_set : array_like
        Sub-set of points for a space-filling design
    rem_set : array_like
        Remaining set of points
    """

    n = D.shape[0]

    sub_pts = []
    rem_pts = list(range(n))

    # first select 2 farthest points
    ini_pts = np.unravel_index(np.argmax(D), D.shape)
    sub_pts.append(ini_pts[0])
    sub_pts.append(ini_pts[1])

    # remove the first 2 points from the remaining list
    rem_pts.remove(ini_pts[0])
    rem_pts.remove(ini_pts[1])

    for i in range(k - 2):
        # find the maximum minimum distance
        dist = D[sub_pts, :]
        min_dist = dist[:, rem_pts]
        min_dist = np.min(min_dist, axis=0)
        max_min_dist = np.max(min_dist)

        # select the first point (in case that several distances are the same, choose the first one)
        points = np.argwhere(dist == max_min_dist)[:, 1].tolist()
        for point in points:
            if point in sub_pts:
                pass
            else:
                sub_pts.append(point)
                rem_pts.remove(point)
                break

    return sub_pts, rem_pts
