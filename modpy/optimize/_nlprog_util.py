import warnings
import numpy as np

from modpy._util import where
from modpy.optimize._lsq import lsq_linear


def _lagrangian_heuristic(dc, df):
    """
    Calculate the initial guess of Lagrangian multipliers based on least-squares.

    Parameters
    ----------
    dc : array_like, shape (m, n)
        Gradient of the constraint function.
    df : array_like, shape (n,)
        Gradient of the objective function

    Returns
    -------
    y : array_like, shape (m,)
        Lagrangian multiplier.
    """

    res = lsq_linear(dc, df)

    if res.success and (res.cond < 1e5):
        y = np.abs(res.x)
    else:
        # TODO: always occurs if bounds are used. Test prior, rather than after somehow?
        y = np.ones(dc.shape[1])
        warnings.warn('Constraint system not well-defined for calculation of initial Lagrangian multiplier.')

    return y


def _max_step_length(x, dx, tau=0.995):
    """
    Computes the maximum step-length, alpha, that allows::

        x + alpha * dx > 0

    This can be used as an initial/maximum input to a line-search algorithm.

    References
    ----------
    [1] Nocedal, J., and Wright, S. J. (2006). Numerical Optimization. Springer New York.
        Pages: 408-409

    Parameters
    ----------
    x : array_like, shape (n,)
        Current solution vector.
    dx : array_like, shape (n,)
        Current step solution vector.
    tau : float, optional
        Parameter of the 'fraction to boundary rule'.

    Returns
    -------
    alphas : tuple
        Maximum step-length for each pair of (x, dx).
    """

    alphas = where(dx < 0., -x, np.divide, np.ones_like, fargs=(dx,))
    alpha = np.minimum(np.maximum(np.amin(alphas), 0.), 1.) if alphas.size else 1.

    return alpha * tau


def _max_step_lengths(*xdx, tau=0.995):
    """
    Computes the maximum step-length, alpha, for each (x, dx) pair in `xdx` that allows::

        x + alpha * dx > 0

    This can be used as an initial/maximum input to a line-search algorithm.

    References
    ----------
    [1] Nocedal, J, and S J Wright. 2006. Numerical Optimization. Springer New York.
        Equation 19.9, page: 567

    Parameters
    ----------
    xdx : tuples
        Tuples of (x, dx) where x and dx are of shape (n,).
    tau : float, optional
        Parameter of the 'fraction to boundary rule'.

    Returns
    -------
    alphas : tuple
        Maximum step-length for each pair of (x, dx).
    """

    return tuple([_max_step_length(x, dx, tau=tau) for (x, dx) in xdx])


def _calculate_hessian(x, y, z, hess, con):
    """
    Parameters
    ----------
    x : array_like, shape (n,)
        Current solution vector.
    y : array_like, shape (me,)
        Lagrangian multiplier of equality constraint.
    z : array_like, shape (mi,)
        Lagrangian multiplier of inequality constraint.
    hess : class Hessian
        Hessian function.
    con : Constraints
        Class Constraints.

    Returns
    -------
    H : array_like, shape (n, n)
        Hessian of the Lagrangian function.
    """

    d2f = hess.calculate(x)
    d2c_eq_i = [yi * ci for (yi, ci) in zip(y, con.hess_eq(x))]
    d2c_iq_i = [zi * ci for (zi, ci) in zip(z, con.hess_iq(x))]
    d2c_eq = np.sum(d2c_eq_i, axis=0) if d2c_eq_i else 0.
    d2c_iq = np.sum(d2c_iq_i, axis=0) if d2c_iq_i else 0.

    return -(d2f - d2c_eq - d2c_iq)
