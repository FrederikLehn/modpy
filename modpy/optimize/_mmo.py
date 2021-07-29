import numpy as np
import numpy.linalg as la

from modpy.special import sqrt
from modpy.optimize._optim_util import OptimizeMMOResult, _function
from modpy.optimize._constraints import Bounds, Constraints, prepare_bounds


TERMINATION_MESSAGES = {
    0: 'maximum number of iterations reached with no result.',
    1: 'maximum number of iterations reached with result(s).',
    2: 'number of pre-determined peaks reached.',
}


def mmo(obj, opt, pop, dim, bounds=None, constraints=(), method='hill', lbound=np.inf, peaks=None, maxiter=100, V=None,
        sigma0=1., args=(), kwargs={}):
    """
    Multi-Modal Optimization algorithm for finding several local and global optimums simultaneously.
    All methods are based on 'niching'.

    Parameters
    ----------
    obj : callable
        Objective function
    opt : callable
        Optimization algorithm for finding optimums in each niche. The algorithm should be callable as follows:
            `opt(obj, x0, bounds, constraints, sigma0=1., C0=None)`
        and should return a class OptimizeResult. The input is:

            obj: objective function
            x0: start-guess
            bounds: bound constraints
            constraints: linear and non-linear constraints
            sigma0: standard deviation (CMA-ES only)
            C0: correlation matrix (CMA-ES only)

    pop : callable
        Method for sampling the population in each loop of the evolution algorithm. Should be callable as follows:
            `pop(N)`
        resulting in an array_like (N, dim) population matrix.
    dim : int
        Problem dimension.
    bounds : 2-tuple of array_like, optional
        Bounds on the solution vector, should be (array_like (n,), array_like (n,)). If None no bounds are applied.
    constraints : tuple or Constraint
        Tuple of class Constraints.
    method : {'hill', }.
        Method used for the optimization.
    lbound : float, optional
        Analytical or empirical bound for the function value to at least obtain in order to have converged.
    peaks : int
        Number of peaks in the objective function If known a priori.
    maxiter : int, optional
        Maximum number of allowed iterations
    V : float
        Volume of the solution space. Only used for method 'hill'.
    sigma0 : float
        Default start-guess of standard deviation if `opt` uses a CMA-ES algorithm.
    args : tuple
        Additional arguments to `obj`.
    kwargs : dict
        Additional key-word arguments to `obj`.

    Returns
    -------
    Tuple of multiple OptimizeResult with the following fields:
    x : array_like, shape (n,)
        Solution vector.
    success : bool,
        True if algorithm converged within its optimality conditions.
    status : int
        Reason for algorithm termination:

            * -1 : LinAlgError due to indeterminate system matrix
            *  0 : Maximum number of iterations reached
            *  1 : All tolerances are satisfied

    message : str
        Description of the termination reason.
    nit : int
        Number of iterations used to converge.
    """

    # wrap function call with args and kwargs
    f = _function(obj, args, kwargs)

    if peaks is None:
        peaks = np.inf

    if V is None:
        if bounds is None:
            raise ValueError('If `bounds` is None, then `V` has to be given.')
        else:
            V = _volume_from_bounds(bounds)

    # perform optimization
    if method == 'hill':

        E, track, status, nit = _hill_valley_ea(f, opt, pop, dim, V, bounds=bounds, constraints=constraints,
                                                lbound=lbound, peaks=peaks, maxiter=maxiter, sigma0=sigma0)

    else:

        raise ValueError("`method` must be 'hill'.")

    res = OptimizeMMOResult(E, track, nit=nit)
    res.success = status > 0
    res.message = TERMINATION_MESSAGES[res.status]

    return res


def _volume_from_bounds(bounds):
    V = 1.

    for (a, b) in bounds:

        if (a in (-np.inf, np.inf)) or (b in (-np.inf, np.inf)):

            raise ValueError('The domain has to be bounded to calculate the volume.')

        V *= (b - a)

    return V


def _hill_valley_test(obj, x, xn, Nt):
    """
    Test if a point `x` and its neighbor `xn` belongs to the same niche based on a Hill-Valley test approach

    Parameters
    ----------
    obj : callable
        Objective function.
    x : array_like, shape (n,)
        Point.
    xn : array_like, shape (n,)
        Neighboring point.
    Nt : int
        Number of test points.

    Returns
    -------
    test : bool
        True if `x` and `xn` belongs to the same niche, otherwise false.
    """

    fmax = max(obj(x), obj(xn))

    for i in range(1, Nt + 1):
        xt = xn + i / (Nt + 1) * (x - xn)
        if fmax < obj(xt):
            return False

    return True


def _hill_valley_clustering(obj, S, eel):
    """
    Partition a set of possible solutions into clusters based on a Hill-Valley niching approach.

    Parameters
    ----------
    obj : callable
        Objective function.
    S : array_like, shape (N, d)
        Set of possible solution vectors sorted low to high based on fitness (low = good fitness)
    eel : float
        Expected edge length.

    Returns
    -------
    K : array_like, (C, d)
        Set of clusters.
    """

    # sort S based on the fitness value
    N, dim = S.shape

    K = [[S[0, :]]]  # clusters (first cluster created from best solution)
    x_in_C = [None for _ in range(N)]  # record of which cluster a given x belongs to
    x_in_C[0] = 0

    for i in range(1, N):
        x = S[i, :]  # point
        xns = S[:i, :]  # neighboring points

        # calculate distance between points and sort based on closest neighbor
        d = la.norm(x - xns, axis=1)
        kj = np.argsort(d.flatten())

        # keep a record of checked clusters to reduce function calls
        added = False
        checked = []

        for j in range(np.minimum(i, dim + 1)):
            k = kj[j]

            Nt = int(1 + np.floor(d[k] / eel))

            if k not in checked:

                if _hill_valley_test(obj, x, xns[k, :], Nt):

                    c = x_in_C[k]
                    K[c].append(x)
                    checked.append(k)
                    x_in_C[i] = c
                    added = True
                    break

        if not added:
            K.append([x])
            x_in_C[i] = len(K) - 1

    return K


def _truncation_selection(obj, P, tau):
    """
    Truncate the set of possible solutions by a fraction `tau` of the best fitted options.

    Parameters
    ----------
    obj : callable
        Objective function.
    P : array_like, shape (d, N)
        Population of candidate solutions
    tau : float
        Value 0 < tau <= 1 of the population to consider.

    Returns
    -------
    C : array_like, (d, K)
        Set of clusters.
    """
    N, _ = P.shape

    # sort population based on fitness
    f = np.array([obj(P[i, :]) for i in range(N)])
    idx = np.argsort(f.flatten())
    S = P[idx, :]

    return S[:int(tau * N), :]


# def _already_optimised(C, E):
#     return np.any([np.isclose(c, e) for c in C for e in E])

def _already_optimised(obj, C, E, eel):
    # find the best solution in the niche `C`.
    val = np.array([obj(c) for c in C])
    best = C[np.argmin(val)]

    #print(best, [('e', e, 'Nt', int(1 + np.floor(la.norm(best - e) / eel)), _hill_valley_test(obj, best, e, int(1 + np.floor(la.norm(best - e) / eel)))) for e in E])

    return np.any([_hill_valley_test(obj, best, e, int(1 + np.floor(la.norm(best - e) / eel))) for e in E])


def _hill_valley_ea(obj, opt, pop, dim, V, peaks, bounds=None, constraints=(), lbound=np.inf, maxiter=100, sigma0=1.):
    """
    A Hill-Valley Clustering Evolution Algorithm for Multi-Modal Optimization.

    NOTE: Inspired from: https://arxiv.org/pdf/1810.07085.pdf

    TODO: Does not solve constrained issues correctly.
    TODO: May need to look at the feasibility of niches prior to launching core algorithm.

    Parameters
    ----------
    obj : callable
        Objective function
    opt : callable
        Optimization algorithm for finding optimums in each niche. The algorithm should be callable as follows:
            `opt(obj, x0, bounds=bounds, constraints=constraints)`
    pop : callable
        Method for sampling the population in each loop of the evolution algorithm. Should be callable as follows:
            p = `pop(N)`
        where 'p' is of shape (N, dim).
    dim : int
        Problem dimension.
    V : float
        Volume of the solution space.
    peaks : int
        Number of peaks in the objective function of known a priori.
    bounds : 2-tuple of array_like, optional
        Bounds on the solution vector, should be (array_like (n,), array_like (n,)). If None no bounds are applied.
    constraints : tuple or Constraint
        Tuple of class Constraints.
    lbound : float, optional
        Analytical or empirical bound for the function value to at least obtain in order to have converged.
    maxiter : int, optional
        Maximum number of allowed iterations.
    sigma0 : float
        Default start-guess of standard deviation if `opt` uses a CMA-ES algorithm.

    Returns
    -------
    x_opt : array_like, shape (n,)
        Solution vector.
    status : int
        Reason for algorithm terminating.
    nit : int
        Number of iterations.
    """

    N = 16 * dim
    N_max = 10 * N
    tau = .5
    E = []
    pre_opt = np.empty((0, dim))
    track = []

    status = 0
    count_N = 0     # count of approximate function calls (more precise than 'it')
    count_nn = 0    # number of iterations with no newly added optimums
    it = 0          # number of MMO iterations

    while it < maxiter:

        P = np.vstack((pop(N), pre_opt))        # sample population
        S = _truncation_selection(obj, P, tau)  # truncate to tau best samples

        eel = sqrt(V / int(N * tau), dim)  # expected edge length

        K = _hill_valley_clustering(obj, S, eel)  # cluster population into niches
        added = False

        count_N += N

        for C in K:

            # continue to next niche if the niche has already been optimized
            if _already_optimised(obj, C, pre_opt, eel):  #_already_optimised(C, pre_opt):
                continue

            # run core search optimization within niche
            X = np.vstack(C)
            x0 = np.mean(X, axis=0)

            # initialize the standard deviation and covariance matrix (only for CMA-ES)
            if (len(C) >= 8) and (dim > 1):
                C0 = np.cov(X, rowvar=False)
                sigma0_ = np.amax(np.abs(np.diag(C0)))
                C0 /= sigma0_

            else:
                sigma0_ = sigma0
                C0 = None

            res = opt(obj, x0, bounds=bounds, constraints=constraints, sigma0=sigma0_, C0=C0)

            # add x to the list of optimums
            if res.success and (res.f <= lbound):
                E.append(res)
                pre_opt = np.vstack((pre_opt, res.x))
                added = True

                # track progression
                track.append(count_N)

        # no new solution was found, expand search space
        if not added:
            if N < N_max:
                N *= 2

            count_nn += 1

        else:
            count_nn = 0

        # terminate if all expected peaks are found
        if len(E) == peaks:
            break

        # terminate if no new optimums found in last 5 iterations
        if count_nn > 10:
            break

        it += 1

    # status
    if (len(E) > 0) and (it < maxiter):
        status = 2

    elif len(E) > 0:
        status = 1

    return E, track, status, it
