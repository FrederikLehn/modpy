import numpy as np
import numpy.linalg as la

from modpy.optimize._optim_util import OptimizeResult, _jacobian_function
from modpy.optimize._constraints import prepare_bounds



TERMINATION_MESSAGES = {
    -1: 'LinAlgError due to indeterminate system.',
    0: 'maximum number of iterations reached.',
    1: 'gtol termination condition is satisfied.',
    2: 'ftol termination condition is satisfied.',
    3: 'xtol termination condition is satisfied.',
    4: 'ftol and xtol termination condition is satisfied.'
}


def least_squares(fun, x0, jac='3-point', bounds=None, ftol=1e-8, xtol=1e-8, gtol=1e-8, maxit=1000, args=(), kwargs={}):
    """
    Solves a non-linear least squares problem with potential bounds on the solution.

    Given a function `fun` which calculates the residuals, the algorithm minimizes::

        min F(x) = 1/2 * sum(f(x) ** 2)
        subject to lb <= x <= ub

    Parameters
    ----------
    fun : callable
        Function which computes a vector of residuals with call f(x, *args, **kwargs).
    x0 : array_like with shape (n,) or float
        Initial guess of the dependent variable.
    jac : {'2-point', '3-point', callable}, optional
        Method for calculating the Jacobian, with `2-point` and `3-point` referring to finite difference schemes.
    bounds : 2-tuple of array_like, optional
        Bounds on the solution vector, should be (array_like (n,), array_like (n,)). If None no bounds are applied.
    ftol : float, optional
        Tolerance related to change in the objective function.
    xtol : float, optional
        Tolerance related to change in the solution vector.
    gtol : float, optional
        Tolerance related to the norm of the gradient.
    maxit : int, optional
        Maximum number of allowable iterations.
    args : tuple
        Additional arguments to `fun`.
    kwargs : dict
        Additional key-word arguments to `fun`.

    Returns
    -------
    OptimizeResult with the following fields:
    x : array_like, shape (n,)
        Solution vector.
    success : bool,
        True if algorithm converged within its optimality conditions.
    status : int
        Reason for algorithm termination:

            * -1 : LinAlgError due to indeterminate system matrix
            *  0 : Maximum number of iterations reached
            *  1 : Gradient tolerance is reached
            *  2 : Function tolerance is reached
            *  3 : Solution tolerance is reached
            *  4 : Function and solution tolerance is reached

    message : str
        Description of the termination reason.
    nit : int
        Number of iterations used to converge. 0 in the unconstrained case (no iterations required).
    """

    jac_wrapped = _jacobian_function(fun, jac, x0, args, kwargs)

    try:
        if bounds:
            # TODO constrained solver
            raise NotImplementedError

        else:

            x, status, nit = _nl_lsq(fun, x0, jac_wrapped, ftol, xtol, gtol, maxit, args=args, kwargs=kwargs)

    except la.LinAlgError:

        x = None
        status = -1
        nit = 0

    res = OptimizeResult(x, status=status, nit=nit)
    res.success = status > 0
    res.message = TERMINATION_MESSAGES[res.status]

    return res


def lsq_obj(r):
    return .5 * la.norm(r) ** 2.


def d_lsq_obj(r, J):
    return J.T @ r


def _check_convergence(df, fd, xd, gtol, ftol, xtol):
    status = 0

    # terminate due to gradient tolerance condition met
    if la.norm(df, np.inf) < gtol:
        status = 1

    # terminate due to function and solution tolerance condition met.
    elif (abs(fd) < ftol) and (np.all(np.abs(xd)) < xtol):
        status = 4

    # terminate due to function tolerance condition met
    elif abs(fd) < ftol:
        status = 2

    # terminate due to solution tolerance condition met
    elif np.all(np.abs(xd)) < xtol:
        status = 3

    return status, status > 0


def _nl_lsq(fun, x0, jac, ftol=1e-8, xtol=1e-8, gtol=1e-8, maxit=1000, maxbt=100, rho=0.5, c=1e-4, args=(), kwargs={}):
    """
    Solves an unconstrained non-linear least squares problem, using a backtracking algorithm.

    Given a function `fun` which calculates the residuals, the algorithm minimizes::

        min F(x) = 1/2 * sum(f(x) ** 2)

    Parameters
    ----------
    fun : callable
        Function which computes a vector of residuals with call f(x, *args, **kwargs).
    x0 : array_like with shape (n,) or float
        Initial guess of the dependent variable.
    jac : callable
        Function for calculating the Jacobian.
    ftol : float, optional
        Tolerance related to change in the objective function.
    xtol : float, optional
        Tolerance related to change in the solution vector.
    gtol : float, optional
        Tolerance related to the norm of the gradient.
    maxit : int, optional
        Maximum number of allowable iterations.
    maxbt : int, optional
        Maximum number of allowable iterations for the backtracking.
    rho : float, optional
        Step-length reduction multiplier for backtracking.
    c : float, optional
        Constant used in calculation of first order optimality conditions
    Returns
    -------
    x : array_like, shape (n,)
        Optimal solution vector.
    """

    # initializing loop
    status = None
    it = 0
    x = x0

    r = fun(x, *args, **kwargs)
    J = jac(x, r, *args, **kwargs)
    f = lsq_obj(r)
    df = d_lsq_obj(r, J)

    x_old = x
    f_old = f

    # iterating --------------------------------------------------------------------------------------------------------
    while it < maxit:
        # calculate optimized step
        Q, R = la.qr(J)
        dx = -la.solve(R, Q.T @ r)

        # invoke backtracking
        alpha = 1.
        it_bt = 0

        x_bt = x + dx
        f_bt = lsq_obj(fun(x_bt, *args, **kwargs))
        csdf = -c * np.dot(dx, df)

        while (f_bt >= (f + alpha * csdf)) and (it_bt < maxbt):
            x_bt = x + alpha * dx
            f_bt = lsq_obj(fun(x_bt, *args, **kwargs))

            alpha *= rho
            it_bt += 1

        x = x_bt

        # update parameters and check convergence
        r = fun(x, *args, **kwargs)
        J = jac(x, r, *args, **kwargs)
        f = lsq_obj(r)
        df = d_lsq_obj(r, J)

        # check whether solution has converged
        status, converged = _check_convergence(df, f-f_old, x-x_old, gtol, ftol, xtol)

        if converged:
            break

        x_old = x
        f_old = f
        it += 1

    if it == maxit:
        status = 0

    return x, status, it
