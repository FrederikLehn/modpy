import numpy as np

from modpy.optimize._optim_util import OptimizeResult, OptimizePath


TERMINATION_MESSAGES = {
    -1: 'ZeroDivisionError due to function evaluation difference too low.',
    0: 'maximum number of iterations reached.',
    1: 'xtol termination condition is satisfied.',
    2: 'rtol termination condition is satisfied.',
    3: 'xtol and rtol termination condition is satisfied.',
}


def _check_convergence(x, xd, xtol, rtol):
    status = 0
    ares = abs(xd)
    acon = ares < xtol

    if x != 0:
        rres = abs(xd / x)
        rcon = rres < rtol
    else:
        rres = 0.
        rcon = False  # even if satisfied, it impossible to evaluate in 0.

    if acon and rcon:
        status = 3

    elif acon:
        status = 1

    elif rcon:
        status = 2

    return status, status > 0, ares, rres


def bisection_scalar(f, a, b, xtol=1e-8, rtol=1e-8, maxit=1000, args=(), kwargs={}):
    """
    Solves a root-finding optimization problem using the bisection method.

    Parameters
    ----------
    f : callable
        Function which computes a scalar result f(x, *args, **kwargs).
    a : float
        Lower bound of the solution space.
    b : float
        Upper bound of the solution space.
    xtol : float, optional
        Absolute tolerance related to change in the solution vector.
    rtol : float, optional
        Relative tolerance related to change in the solution vector.
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

            * -1 : ZeroDivisionError due to function evaluation difference being too low
            *  0 : Maximum number of iterations reached
            *  1 : Absolute solution tolerance is reached
            *  2 : Relative solution tolerance is reached
            *  3 : Absolute and relative solution tolerance is reached

    message : str
        Description of the termination reason.
    nit : int
        Number of iterations used to converge. 0 in the unconstrained case (no iterations required).
    """

    status = 0
    it = 0
    fa = f(a, *args, **kwargs)
    c = None
    c_old = a

    while it < maxit:
        c = (a + b) / 2.
        fc = f(c, *args, **kwargs)

        status, converged, _, _ = _check_convergence(c, c - c_old, xtol, rtol)

        if converged:
            break

        if (fa - fc) < 0:
            a = c
        else:
            b = c

        c_old = c
        it += 1

    res = OptimizeResult(c, status=status, nit=it)
    res.success = res.status > 0
    res.message = TERMINATION_MESSAGES[res.status]

    return res


def secant_scalar(f, x0, x1, xtol=1e-8, rtol=1e-8, maxit=1000, args=(), kwargs={}):
    """
    Solves a root-finding optimization problem using the secant method.

    Parameters
    ----------
    f : callable
        Function which computes a scalar result f(x, *args, **kwargs).
    x0 : float
        Initial guess of the dependent variable.
    x1 : float
        Second guess of the dependent variable.
    xtol : float, optional
        Absolute tolerance related to change in the solution vector.
    rtol : float, optional
        Relative tolerance related to change in the solution vector.
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

            * -1 : ZeroDivisionError due to function evaluation difference being too low
            *  0 : Maximum number of iterations reached
            *  1 : Absolute solution tolerance is reached
            *  2 : Relative solution tolerance is reached
            *  3 : Absolute and relative solution tolerance is reached

    message : str
        Description of the termination reason.
    nit : int
        Number of iterations used to converge. 0 in the unconstrained case (no iterations required).
    """

    status = 0
    it = 0
    x = x1
    x_old = x

    while it < maxit:
        f0 = f(x0, *args, **kwargs)
        f1 = f(x1, *args, **kwargs)
        x -= f1 * (x - x0) / (f1 - f0)

        if x in (np.inf, np.nan):
            raise ZeroDivisionError

        status, converged, _, _ = _check_convergence(x, x - x_old, xtol, rtol)

        if converged:
            break

        x0 = x1
        x1 = x
        x_old = x
        it += 1

    res = OptimizeResult(x, status=status, nit=it)
    res.success = res.status > 0
    res.message = TERMINATION_MESSAGES[res.status]

    return res


def newton_scalar(f, df, x0, xtol=1e-8, rtol=1e-8, maxiter=1000, keep_path=False, args=(), kwargs={}):
    """
    Solves a root-finding optimization problem using Newton's method.

    Parameters
    ----------
    f : callable
        Function which computes a scalar result f(x, *args, **kwargs).
    df : callable
        Function which computes the derivative of `f`.
    x0 : float
        Initial guess of the dependent variable.
    xtol : float, optional
        Absolute tolerance related to change in the solution vector.
    rtol : float, optional
        Relative tolerance related to change in the solution vector.
    maxiter : int, optional
        Maximum number of allowable iterations.
    keep_path : bool, optional
        Whether to save path information. Can require substantial memory.
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

            * -1 : ZeroDivisionError due to function evaluation difference being too low
            *  0 : Maximum number of iterations reached
            *  1 : Absolute solution tolerance is reached
            *  2 : Relative solution tolerance is reached
            *  3 : Absolute and relative solution tolerance is reached

    message : str
        Description of the termination reason.
    nit : int
        Number of iterations used to converge. 0 in the unconstrained case (no iterations required).
    """

    path = OptimizePath(keep=keep_path)

    status = 0
    it = 0
    x = x0
    x_prev = x0
    y = f(x0, *args, **kwargs)
    dy = df(x0, *args, **kwargs)

    if keep_path:
        path.append(x0, y, 0.)  # TODO: not zero tol

    while it < maxiter:

        x -= y / dy

        status, converged, ares, rres = _check_convergence(x, x - x_prev, xtol, rtol)

        # update y and dy
        y = f(x, *args, **kwargs)
        dy = df(x, *args, **kwargs)

        if keep_path:
            path.append(x, y, ares, rtol=rres)

        if converged:
            break

        x_prev = x
        it += 1

    res = OptimizeResult(x, y, status=status, nit=it)
    res.success = res.status > 0
    res.message = TERMINATION_MESSAGES[res.status]
    res.path = path

    return res
