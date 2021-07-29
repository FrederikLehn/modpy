import numpy as np

from modpy._util import sign


# ======================================================================================================================
# Finite Difference Approximations
# ======================================================================================================================
def _eps_for_method(method):
    """
    Calculate the relevant eps for a given method of a finite difference scheme.

    NOTE: Inspired from: https://github.com/scipy/scipy/blob/master/scipy/optimize/_numdiff.py

    Parameters
    ----------
    method : {'2-point', '3-point'}
        Method used for the finite difference scheme.

    Returns
    -------
    eps : float
        Relative step-size (eps).
    """

    eps = np.finfo(np.float64).eps

    if method == '2-point':
        return eps ** .5
    elif method == '3-point':
        return eps ** (1. / 3.)
    else:
        raise ValueError("Step-method must be either '2-point' or '3-point'.")


def _compute_absolute_step(rel_step, x0, method):
    """
    Calculates the absolute step-size used to compute an approximation of a numerical Jacobian matrix.

    NOTE: Inspired from: https://github.com/scipy/scipy/blob/master/scipy/optimize/_numdiff.py

    Parameters
    ----------
    rel_step : float
        Relative step-size (eps).
    x0 : array_like
        Point at which to calculate the gradient at.
    method : {'2-point', '3-point'}
        Method used for the finite difference scheme.

    Returns
    -------
    abs_step : array_like
        Absolute step-size (h).
    """

    if rel_step is None:
        rel_step = _eps_for_method(method)

    return rel_step * sign(x0) * np.maximum(1., np.abs(x0))


def approx_difference(fun, x0, method='3-point', rel_step=None, abs_step=None, f0=None, args=(), kwargs={}):
    """
    Calculates an approximation of the Jacobian of `fun`at the point `x0` for a finite difference scheme.

    NOTE: Inspired from: https://github.com/scipy/scipy/blob/master/scipy/optimize/_numdiff.py

    Parameters
    ----------
    fun : callable
        Function which computes a vector of residuals with call f(x, *args, **kwargs).
    x0 : array_like with shape (n,) or float
        Initial guess of the dependent variable.
    method : {'2-point', '3-point'}, optional
        Method used for the finite difference scheme.
    rel_step : float, optional
        Relative step-size (eps).
    abs_step : float, optional
        Absolute step-size (h).
    f0 : array_like
        `fun` evaluated in x0.
    args : tuple
        Additional arguments to `fun`.
    kwargs : dict
        Additional key-word arguments to `fun`.

    Returns
    -------
    J : array_like, shape(m, n)
        Approximation of the Jacobian matrix.
    """

    if f0 is None:
        f0 = fun(x0, *args, **kwargs)

    if abs_step is None:
        h = _compute_absolute_step(rel_step, x0, method)

    else:
        h = abs_step

        # cannot allow zero-step, so replacing zeros by computed alternative.
        dx = ((x0 + h) - x0)
        h = np.where(dx == 0, _compute_absolute_step(rel_step, x0, method), h)

    if method == '2-point':
        one_sided = np.ones_like(h, dtype=bool)

    elif method == '3-point':
        one_sided = np.zeros_like(h, dtype=bool)

    else:
        raise ValueError("Step-method must be either '2-point' or '3-point'.")

    return _dense_difference(fun, x0, f0, h, one_sided, method)


def _dense_difference(fun, x0, f0, h, one_sided, method):
    """
    Calculates an approximation of the Jacobian of `fun`at the point `x0` in dense matrix form.

    NOTE: Inspired from: https://github.com/scipy/scipy/blob/master/scipy/optimize/_numdiff.py

    Parameters
    ----------
    fun : callable
        Function which computes a vector of residuals with call f(x, *args, **kwargs).
    x0 : array_like with shape (n,) or float
        Initial guess of the dependent variable.
    method : {'2-point', '3-point'}, optional
        Method used for the finite difference scheme.

    Returns
    -------
    J : array_like, shape (m, n)
        Approximation of the Jacobian matrix.
    """

    m = f0.size
    n = x0.size
    Jt = np.empty((n, m))
    hv = np.diag(h)

    for i in range(h.size):

        if method == '2-point':

            x = x0 + hv[i]
            dx = x[i] - x0[i]
            df = fun(x) - f0

        elif (method == '3-point') and one_sided[i]:

            x1 = x0 + hv[i]
            x2 = x0 + 2. * hv[i]
            dx = x2[i] - x0[i]
            f1 = fun(x1)
            f2 = fun(x2)
            df = -3. * f0 + 4. * f1 - f2

        elif (method == '3-point') and (not one_sided[i]):

            x1 = x0 - hv[i]
            x2 = x0 + hv[i]
            dx = x2[i] - x1[i]
            f1 = fun(x1)
            f2 = fun(x2)
            df = f2 - f1

        else:

            raise ValueError("Step-method must be either '2-point' or '3-point'.")

        Jt[i, :] = df / dx

    if m == 1:
        Jt = np.ravel(Jt)

    return Jt.T


# ======================================================================================================================
# Hessian Approximations
# ======================================================================================================================
def _damped_BFGS_update(B, s, y):
    """
    Updates the approximate Hessian matrix `B` using a damped BFGS method.

    Reference: Nocedal, J, and S J Wright. 2006. Numerical Optimization. Springer New York.
               Procedure 18.2 (Damped BFGS Updating), page 537.

    Parameters
    ----------
    B : array_like, shape (n, n)
        Approximate Hessian matrix
    s : array_like, shape (n,)
        Search direction vector.
    y : array_like, shape (n,)
        Residual of gradient of Lagrangian.

    Returns
    -------
    B : array_like, shape (n, n)
        Updated approximation of the Hessian matrix
    """

    Bs = B @ s
    sBs = s.T @ Bs
    sy = s.T @ y

    if sy >= .2 * sBs:
        theta = 1.
    else:
        theta = (.8 * sBs) / (sBs - sy)

    r = theta * y + (1. - theta) * Bs
    sr = s.T @ r

    if (sr > 1e-15) and (sBs > 1e-15):
        B += np.outer(r, r) / sr - np.outer(Bs, Bs) / sBs

    return B
