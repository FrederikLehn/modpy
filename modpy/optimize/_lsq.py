from warnings import warn
import numpy as np
import numpy.linalg as la

from modpy.optimize._optim_util import OptimizeResult
from modpy.optimize._constraints import prepare_bounds


TERMINATION_MESSAGES = {
    -1: 'LinAlgError due to indeterminate system.',
    0: 'maximum number of iterations reached.',
    1: 'not implemented yet.',
    2: 'not implemented yet.',
    3: 'the unconstrained solution is optimal.'
}


def lsq_linear(A, b, W=None, bounds=None, method='', tol=1e-10, max_iter=100):
    """
    Solves a weighted linear least squares problem with bounds on the variables.

    Given an m-by-n system matrix, A, and a target vector with m elements,
    solves the least squares formulation::

        min 1/2 * ||A x - b|| ** 2
        s.t. lb <= x <= ub

    with unconstrained solution (ordinary)::

        x = (A'A) **(-1) A'b

    with unconstrained solution (weighted)::

        x = (A'WA) **(-1) A'Wb

    Parameters
    ----------
    A : array_like, shape (n, m)
        System matrix
    b : array_like, shape (n,)
        Target vector
    W : array_like, shape (n, n), optional
        Weight matrix
    bounds : 2-tuple of array_like, optional
        Lower and upper bounds. Both lower and upper bound must
        have shape (n,) or be a scalar. Use np.inf for no bounds
    method : string, optional
        TODO: allow Trust Region Reflective algorithm
    tol : float, optional
        Tolerance parameter related to convergence of iterative bounded solver
    max_iter : None or int, optional
        Maximum number of iterations during iterative bounded solver

    Returns
    -------
    OptimizeResult with the following fields:
    x : array_like, shape (m,)
        Solution vector.
    success : bool,
        True if algorithm converged within its optimality conditions.
    status : int
        Reason for algorithm termination:

            * -1 : LinAlgError due to indeterminate system matrix (or weights)
            *  0 : Maximum number of iterations reached (TODO: for trust-region)
            *  1 : TODO: first order optimality reached (trust-region)
            *  2 : TODO : relative change of cost function (trust-region)
            *  3 : Unconstrained solution is optimal

    message : str
        Description of the termination reason.
    nit : int
        Number of iterations used to converge. 0 in the unconstrained case (no iterations required).
    """

    n, m = A.shape

    b = np.atleast_1d(b)
    if b.ndim != 1:
        if (b.ndim == 2) and (np.prod(b.shape) == b.size):
            b = np.reshape(b, (b.size,))
        else:
            raise ValueError('`b` must have at most 1 dimension')

    if b.size != n:
        raise ValueError('Inconsistent shapes between `A` and `b`.')

    if (W is not None) and ((W.shape[0] != n) or (W.shape[1] != n)):
        raise ValueError('Inconsistent shapes between `A` and `W`.')

    if bounds is not None:
        lb, ub = prepare_bounds(bounds, n)

    try:

        if W is None:
            x, cond = _ordinary_lsq(A, b)
        else:
            x, cond = _weighted_lsq(A, b, W)

        res = OptimizeResult(x, success=True, status=3, nit=0, cond=cond)

    except la.LinAlgError:
        res = OptimizeResult(None, success=False, status=-1, nit=0)

    res.message = TERMINATION_MESSAGES[res.status]

    return res


def _weighted_lsq(A, b, W):
    ATW = A.T @ W
    Q, R = la.qr(ATW @ A)
    return la.solve(R, Q.T @ (ATW @ b)), _condition(R)


def _ordinary_lsq(A, b):
    Q, R = la.qr(A.T @ A)
    return la.solve(R, Q.T @ (A.T @ b)), _condition(R)


def _condition(R):
    cond = la.cond(R)
    if cond > 1e5:
        warn('The condition number of R in the LSQ solver is: {}'.format(cond))

    return cond
