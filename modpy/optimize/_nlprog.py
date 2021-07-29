import warnings
import numpy as np
import numpy.linalg as la

from modpy.optimize._constraints import _prepare_constraints
from modpy.optimize._optim_util import OptimizeResult, _function, _jacobian_function, _hessian_function, _chk_callable
from modpy.optimize._nl_sqp import _nl_unc, _nl_sqp
from modpy.optimize._nl_ip import _nl_ip
from modpy._exceptions import InfeasibleProblemError


TERMINATION_MESSAGES = {
    -4: 'NaN or +/- inf values encountered in solution.',
    -3: 'Quadratic sub-problem in SQP failed to converge.',
    -2: 'The NLP has infeasible linear constraints.',
    -1: 'LinAlgError due to indeterminate system.',
    0: 'maximum number of iterations reached.',
    1: 'Tolerance termination condition is satisfied.',
}


def nlprog(obj, x0, jac='3-point', hess='BFGS', bounds=None, constraints=(), method='SQP', merit='lagrangian', tol=1e-6,
           maxiter=1000, keep_path=False, args=(), kwargs={}):
    """
    Solves a non-linear programming problem with equality and inequality constraints as well as bounds, of the form::

        min f(x)
        s.t. ce(x) = 0
        s.t. ci(x) >= 0
        s.t. lb <= x <= ub

    Parameters
    ----------
    obj : callable
        Objective function to optimize. Returns a vector of shape (n,)
    x0 : array_like or int
        If array_like `x0` is used as a start guess, if int it is the problem dimension.
    jac : {'2-point', '3-point', callable}
        Function for calculating the Jacobian of the objective function. If callable the input should be:
            jac(x)
        and return a vector of shape (n,)
    hess : {'BFGS', 'SR1', callable}
        Function for calculating the Hessian of the objective function. If callable the input should be:
            hess(x)
        and a matrix of shape (n, n)
        TODO: implement SR1 (see discussion page 538)
        TODO: additional options at: https://en.wikipedia.org/wiki/Quasi-Newton_method
    bounds : 2-tuple of array_like, optional
        Bounds on the solution vector, should be (array_like (n,), array_like (n,)). If None no bounds are applied.
    constraints : tuple or Constraint
        Tuple of class Constraints.
    method: {'SQP', 'IP'}
        Algorithm used for solving the non-linear programming problem.
    merit : {'lagrangian', 1, 2, np.inf}:
        The merit function used the in linesearch algorithm:

            'lagrangian': Augmented Lagrangian merit function (smooth)
            1: l1 merit norm (non-smooth)
            2: l2 merit norm (non-smooth)
            np.inf: maximum merit norm (non-smooth)

    tol : float, optional
        Tolerance related to change in the Lagrangian objective function.
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

            * -1 : LinAlgError due to indeterminate system matrix
            *  0 : Maximum number of iterations reached
            *  1 : All tolerances are satisfied

    message : str
        Description of the termination reason.
    nit : int
        Number of iterations used to converge.
    """

    # wrap function call with args and kwargs
    x0 = np.array(x0)
    f = _function(obj, args, kwargs)
    J = _jacobian_function(obj, jac, x0, args, kwargs)
    H = _hessian_function(hess, args=args, kwargs=kwargs)
    _chk_callable(x0, f, J, H)

    # TODO: ensure hess=SR1 is only used for trust-region algorithms
    # TODO: and that if hess is callable, any non-linear constraint has a callable hess as well

    f_opt = None
    sol = None
    nit = 0
    path = None

    try:

        if (bounds is None) and (not constraints):  # unconstrained

            if method == 'SQP':

                sol, f_opt, status, nit, path = _nl_unc(f, x0, J, H, tol=tol,
                                                        maxiter=maxiter, keep_path=keep_path)

            else:

                raise ValueError("For unconstrained problems `method` must be 'SQP'.")

        else:  # constrained

            # prepare constraints
            con = _prepare_constraints(bounds, constraints, x0.size)

            if method == 'SQP':

                sol, f_opt, status, nit, path = _nl_sqp(f, x0, J, H, con, merit=merit, tol=tol,
                                                        maxiter=maxiter, keep_path=keep_path)

            elif method == 'IP':

                sol, f_opt, status, nit, path = _nl_ip(f, x0, J, H, con, tol=tol,
                                                       maxiter=maxiter, keep_path=keep_path)

            else:

                raise ValueError("For constrained problems `method` must be either 'SQP' or 'IP'.")

    except la.LinAlgError:
        status = -1

    except InfeasibleProblemError:
        status = -2

    if sol is None:
        x = None
    else:
        x = sol[:x0.size]

    res = OptimizeResult(x, f_opt, sol, status=status, nit=nit, tol=tol)
    res.success = status > 0
    res.message = TERMINATION_MESSAGES[res.status]
    res.path = path

    return res


