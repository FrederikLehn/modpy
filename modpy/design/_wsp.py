import numpy as np
import numpy.linalg as la

from modpy._util import point_distance
from modpy.design._design_util import _scale_sfd_design, _initial_sfd_samples


def wsp_design(joint, k, m, scale=False, method='iterative', ord_=None, tol=0.1, maxit=10, seed=None):
    """
    Create a WSP space-filling design with `k` samples from an initial sample pool of size `m`.

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
    method : str
        Method {'simple', 'iterative'} approach to sampling the k points.
    ord_ : int
        Order of the distance norm, default to Euclidian.
    tol : float
        Tolerance for the iterative method. Allowable tolerance on points is int(k * tol).
    maxit : int
        Maximum number of iterations for iterative method.
    seed : int
        Seed of the random generator.

    Returns
    -------
    D : array_like, shape (samples, dim)
        Design matrix of the form 'number of experiments' x 'number of variables'.
    """

    if k > m:
        raise ValueError('`k` has to satisfy k<=m.')

    dim = len(joint.marginals)

    # generate candidate pool
    pool = _initial_sfd_samples(dim, m, seed=seed)

    # if required, iteratively attempt to find a design with size as close to `k` as possible
    if method == 'simple':
        d = la.norm([2.] * dim) / k
        D = wsp(pool, d, ord_=ord_)

    elif method == 'iterative':
        D = _wsp_root_finding(pool, dim, k, ord_=ord_, tol=tol, maxit=maxit)

    else:
        raise ValueError("`method` must be either 'simple' or 'iterative'.")

    if scale:
        D = _scale_sfd_design(D, joint)

    return D


def _wsp_root_finding(pool, dim, k, ord_=None, tol=0.1, maxit=10):
    """
    Apply a root-finding iterative algorithm to try to achieve a design with as close to `k` points as possible.

    Parameters
    ----------
    pool : array_like
        Pool of candidate points.
    k : int
        Number of final samples in design.
    dim : int
        Number of dimensions
    ord_ : int
        Order of the distance norm, default to Euclidian.
    tol : float
        Tolerance for the iterative method. Allowable tolerance on points is int(k * tol).
    maxit : int
        Maximum number of iterations for iterative method.

    Returns
    -------
    D : array_like, shape (samples, n)
        Design matrix of the form 'number of experiments' x 'number of variables'.
    """

    if maxit < 1:
        raise ValueError('Maximum number of iterations must be larger than 0.')

    tol_num = int(k * tol)
    it = 0

    # initial interval
    a = 2. / k  # minimum initial distance (high k)
    b = la.norm([2.] * dim)  # maximum initial distance (low k)
    Da, _ = wsp(pool, a, ord_=ord_)
    Db, _ = wsp(pool, b, ord_=ord_)
    ka, _ = Da.shape
    kb, _ = Db.shape

    if abs(k - ka) < tol_num:
        return Da

    if abs(k - kb) < tol_num:
        return Db

    del Da
    del Db

    while it < maxit:

        c = (a + b) / 2.  # start guess
        D, _ = wsp(pool, c, ord_=ord_)
        kc, _ = D.shape

        if abs(k - kc) <= tol_num:
            return D

        if k - kc < 0:
            a = c
        else:
            b = c

        it += 1

    return D


def wsp(X, d, ord_=None):
    """
    Create a WSP space-filling design by drawing samples from an initial pool `X` with a minimum distance `d`.

    NOTE: Based on https://hal-amu.archives-ouvertes.fr/hal-02079830/document

    Parameters
    ----------
    X : array_like
        Initial sample pool.
    d : float
        Minimum distance used in point selection.
    ord_ : int
        Order of the distance norm, default to Euclidian.

    Returns
    -------
    design : array_like, shape (k, n)
        Design matrix of the form 'number of experiments' x 'number of variables'.
    idx : array_like, shape (k,)
        Indices of samples to pick from `X`.
    """

    # find initial point near the center of gravity.
    m, n = X.shape
    idx0 = int(m / 2.)

    idx, _ = _wsp_split(idx0, point_distance(X, ord_=ord_), d)
    return X[idx, :], idx


def _wsp_split(idx0, D, d):
    """
    Split a set of points into two groups. A sub-set and a remaining set of points based on a minimum-distance approach.
    This minimum distance approach follows a WSP algorithm.

    Parameters
    ----------
    idx0 : int
        Index of first point
    D : array_like
        Matrix of distances between points
    d : float
        Minimum distance used in point selection.

    Returns
    -------
    sub_set : array_like
        Sub-set of points for a space-filling design
    rem_set : array_like
        Remaining set of points
    """

    n = D.shape[0]

    sub_pts = []
    rem_pts = []
    ini_pts = list(range(n))

    sub_pts.append(idx0)
    ini_pts.remove(idx0)

    while True:
        p = sub_pts[-1]  # active point from which to eliminate points in a circle around
        p_in = [ini_pts[int(p_)] for p_ in np.argwhere(D[p, ini_pts] < d)]  # points inside the circle of minimum distance

        # add points to the remaining list and remove points inside the circle from available list
        for p_ in p_in:
            rem_pts.append(p_)
            ini_pts.remove(p_)

        # stop looking when candidate pool is empty
        if not ini_pts:
            break

        # jump to the closest next point
        idx = np.argwhere(D[p, :] == np.min(D[p, ini_pts]))[0][0]

        # add point to final set of points and remove from candidate pool
        sub_pts.append(idx)
        ini_pts.remove(idx)

    return sub_pts, rem_pts
