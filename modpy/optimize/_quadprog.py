import numpy as np
import numpy.linalg as la

from modpy._util import where
from modpy.optimize._optim_util import OptimizeResult, _chk_dimensions, _chk_system_dimensions, _ensure_vector,\
    _atleast_zeros
from modpy.optimize._nlprog_util import _max_step_lengths
from modpy.optimize._constraints import prepare_bounds, _bounds_to_equations
from modpy.optimize._presolve import Presolver
from modpy._exceptions import InfeasibleProblemError


TERMINATION_MESSAGES = {
    -4: 'NaN or +/- inf values encountered in solution.',
    -3: 'The QP is dual infeasible.',
    -2: 'The QP is primal infeasible.',
    -1: 'LinAlgError due to indeterminate system.',
    0: 'maximum number of iterations reached.',
    1: 'Tolerance termination condition is satisfied.',
}


def quadprog(H, g, A=None, b=None, C=None, d=None, bounds=None, warmstart={}, ftol=1e-6, etol=1e-6, itol=1e-6,
             mtol=1e-6, maxiter=1000):
    """
    Solves a non-linear quadratic programming problem with linear equality and inequality constraints, of the form::

        min  1/2 x'Hx + g'x
        s.t. Ax = b
        s.t. Cx >= d
        s.t. lb <= x <= ub

    Parameters
    ----------
    H : array_like, shape (n, n)
        System matrix with coefficients of the quadratic terms.
    g : array_like, shape (n,)
        System vector with coefficients of the linear terms.
    A : array_like, shape (n, me)
        System matrix with coefficients for the equality constraints.
    b : array_like, shape (me,)
        Results vector of the equality constraints.
    C : array_like, shape (n, mi)
        System matrix with coefficients for the inequality constraints.
    d : array_like, shape (mi,)
        Results vector of the inequality constraints.
    bounds : 2-tuple of array_like, optional
        Bounds on the solution vector, should be (array_like (n,), array_like (n,)). If None no bounds are applied.
    warmstart : dict
        Dictionary containing start-guesses for all or some of the required solution vectors, including
        results, slacks and Lagrangian multipliers. It may contain the following:

            'x0': array_like, shape (n,)
            'y0': array_like, shape (me,)
            'z0': array_like, shape (mi,)
            's0': array_like, shape (mi,)

    ftol : float, optional
        Tolerance related to change in the objective function.
    etol : float, optional
        Tolerance related to change in the equality constraints.
    itol : float, optional
        Tolerance related to change in the inequality constraints.
    mtol : float, optional
        Tolerance of the penalty parameter.
    maxiter : int, optional
        Maximum number of allowable iterations.

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
        Number of iterations used to converge. 0 in the unconstrained case (no iterations required).
    """

    n = g.size

    # if quadprog is called using a warmstart procedure, it is assumed
    # appropriate checks are made beforehand on sizing of the system.
    # The following checks are then skipped to save time.
    if not warmstart:
        # ensure consistent sizing of input
        A = _atleast_zeros(A, size=(0, n))
        C = _atleast_zeros(C, size=(0, n))
        b = _atleast_zeros(b, size=(0,))
        d = _atleast_zeros(d, size=(0,))

        b = _ensure_vector(b, 'b')
        d = _ensure_vector(d, 'd')

        H, g = _chk_dimensions(H, g, 'H', 'g')
        A, _ = _chk_dimensions(A, g, 'A', 'g')
        C, _ = _chk_dimensions(C, g, 'C', 'g')

        _chk_system_dimensions((H, A.T, C.T), (g, b, d))

    # pre-solve the QP
    lb, ub = prepare_bounds(bounds, n)
    PS = Presolver(g, A, b, C, d, lb, ub, H=H)

    x = None
    f = None
    sol = None
    status = 1
    message = ''
    nit = 0

    # solve system
    try:

        # pre-solve the QP
        PS.presolve_QP()
        H, g, A, b, C, d, lb, ub = PS.get_QP()

        # add bounds to inequality constraints
        C, d = _bounds_to_equations(lb, ub, C, d)

        if (not A.size) and (not C.size):  # unconstrained

            sol = _quad(H, g)

        elif not C.size:  # equality constrained

            sol = _quad_eq(H, g, A, b)

        else:  # inequality constrained (and optionally equality constrained)

            x0, y0, z0, s0 = _merge_start_guess(warmstart, n, b.size, d.size)
            sol, status, nit = _quad_iq(H, g, A, b, C, d, x0, y0, z0, s0, ftol, etol, itol, mtol, maxiter)

    except InfeasibleProblemError as e:

        status = e.status
        message = e.message

    except la.LinAlgError:

        status = -1
        message = TERMINATION_MESSAGES[status]

    if status > 0:

        x = sol[:g.size]

        # post-solve the LP
        x, f = PS.postsolve(x)
        message = TERMINATION_MESSAGES[status]

    res = OptimizeResult(x, f, sol, status=status, message=message, nit=nit)
    res.success = status > 0

    return res


def _merge_start_guess(warmstart, n, me, mi):
    """
    Merges the start-guesses based on a supplied warmstart and default values.
    Dimension consistency checks of the warmstart input is foregone to save computational time.

    Parameters
    ----------
    warmstart : dict
        Dictionary containing start-guesses for all or some of the required solution vectors, including
        results, slacks and Lagrangian multipliers. It may contain the following keys:

            'x0': array_like, shape (n,)
            'y0': array_like, shape (me,)
            'z0': array_like, shape (mi,)
            's0': array_like, shape (mi,)

    n : int
        Number of variables.
    me : int
        Number of equality constraints.
    mi : int
        Number of inequality constraints.

    Returns
    -------
    x0 : array_like, shape (n,)
        Results vector.
    y0 : array_like, shape (me,)
        Lagrangian multipliers of equality constraints.
    z0 : array_like, shape (mi,)
        Lagrangian multipliers of inequality constraints.
    s0 : array_like, shape (mi,)
        Slack variables.
    """

    if 'x0' in warmstart:
        x0 = warmstart['x0']
    else:
        x0 = np.zeros((n,))

    if 'y0' in warmstart:
        y0 = warmstart['y0']
    else:
        y0 = np.zeros((me,))

    if 'z0' in warmstart:
        z0 = warmstart['z0']
    else:
        z0 = np.ones((mi,))

    if 's0' in warmstart:
        s0 = warmstart['s0']
    else:
        s0 = np.ones((mi,))

    return x0, y0, z0, s0


def _define_tolerances(H, g, A, b, C, d, mu0, ftol, etol, itol, mtol):
    """
    Define convergence tolerances based on user supplied tolerances.

    Parameters
    ----------
    H : array_like, shape (n, n)
        System matrix with coefficients of the quadratic terms.
    g : array_like, shape (n,)
        System vector with coefficients of the linear terms.
    A : array_like, shape (n, me)
        System matrix with coefficients for the equality constraints.
    b : array_like, shape (me,)
        Results vector of the equality constraints.
    C : array_like, shape (n, mi)
        System matrix with coefficients for the inequality constraints.
    d : array_like, shape (mi,)
        Results vector of the inequality constraints.
    mu0: float
         Initial value of penalty parameter.
    ftol : float
        Tolerance related to change in the objective function.
    etol : float
        Tolerance related to change in the equality constraints.
    itol : float
        Tolerance related to change in the inequality constraints.
    mtol : float
        Tolerance of the penalty parameter.

    Returns
    -------
    epsL : float
        Tolerance of the residual of the Lagrangian equation.
    epsA : float
        Tolerance of the residual of the equality constraint equations.
    epsC : float
        Tolerance of the residual of the inequality constraint equations.
    epsM : float
        Tolerance of the penalty parameters convergence towards 0.
    """

    L = np.block([H, g.reshape((g.shape[0], 1)), A.T, C.T])
    eq = np.block([A, b.reshape((b.shape[0], 1))])
    iq = np.block([np.eye(d.size), d.reshape((d.shape[0], 1)), C])

    epsL = ftol * np.maximum(1., la.norm(L, ord=np.inf))
    epsA = etol * np.maximum(1., la.norm(eq, ord=np.inf) if eq.size else 0.)
    epsC = itol * np.maximum(1., la.norm(iq, ord=np.inf) if iq.size else 0.)
    epsM = mtol * 1e-2 * mu0

    return epsL, epsA, epsC, epsM


def _check_convergence(rL, rA, rC, mu, epsL, epsA, epsC, epsM):
    """
    Checks of the convergence criteria of a quadratic interior point method are satisfied.

    Reference: Nocedal, J, and S J Wright. 2006. Numerical Optimization. Springer New York.

    Parameters
    ----------
    rL : array_like, shape (n,)
        Residuals of the Lagrangian equation.
    rA : array_like, shape (me,)
        Residuals of the equality constraints.
    rC : array_like, shape (mi,)
        Residuals of the inequality constraints.
    mu : float
        Penalty parameter.
    epsL : float
        Tolerance of the residual of the Lagrangian equation.
    epsA : float
        Tolerance of the residual of the equality constraint equations.
    epsC : float
        Tolerance of the residual of the inequality constraint equations.
    epsM : float
        Tolerance of the penalty parameters convergence towards 0.

    Returns
    -------
    converged : bool
        Boolean of whether all termination criteria are satisfied.
    """

    nL = la.norm(rL, ord=np.inf)
    nA = la.norm(rA, ord=np.inf) if rA.size else 0.
    nC = la.norm(rC, ord=np.inf) if rC.size else 0.
    nM = np.abs(mu)

    return (nL <= epsL) & (nA <= epsA) & (nC <= epsC) & (nM <= epsM)


def _compute_residuals(H, g, A, b, C, d, x, y, z, s):
    """
    Computes the residuals for the kth iteration of a quadratic interior point method.

    Reference: Nocedal, J, and S J Wright. 2006. Numerical Optimization. Springer New York.
               Page: 482

    Parameters
    ----------
    H : array_like, shape (n, n)
        System matrix with coefficients of the quadratic terms.
    g : array_like, shape (n,)
        System vector with coefficients of the linear terms.
    A : array_like, shape (n, me)
        System matrix with coefficients for the equality constraints.
    b : array_like, shape (me,)
        Results vector of the equality constraints.
    C : array_like, shape (n, mi)
        System matrix with coefficients for the inequality constraints.
    d : array_like, shape (mi,)
        Results vector of the inequality constraints.
    x : array_like, shape (n,)
        Solution vector.
    y : array_like, shape (me,)
        Lagrangian multipliers of the equality constraints.
    z : array_like, shape (mi,)
        Lagrangian multipliers of the inequality constraints.
    s : array_like, shape (mi,)
        Slack variables.

    Returns
    -------
    rL : array_like, shape (n,)
        Residuals of the Lagrangian equation.
    rA : array_like, shape (me,)
        Residuals of the equality constraints.
    rC : array_like, shape (mi,)
        Residuals of the inequality constraints.
    rsz : array_like, shape (mi,)
        Residuals of the complimentary condition.
    """

    rL = H @ x + g - A.T @ y - C.T @ z
    rA = b - A @ x
    rC = s + d - C @ x
    rsz = s * z

    return rL, rA, rC, rsz


def _compute_step_length(s, ds, z, dz):
    """
    Computes the step-length for a quadratic interior point method.

    Reference: Nocedal, J, and S J Wright. 2006. Numerical Optimization. Springer New York.
               Page: 408 and 483

    Parameters
    ----------
    s : array_like, shape (mi,)
        Slack variables.
    ds : array_like, shape (mi,)
        Step solution to the slack variables.
    z : array_like, shape (mi,)
        Lagrangian multipliers of the inequality constraints.
    dz : array_like, shape (mi,)
        Step solution to the Lagrangian multipliers of the inequality constraints.

    Returns
    -------
    alpha : float
        Step-length.
    """

    prim = where(ds < 0., -s, np.divide, np.ones_like, fargs=(ds,))
    dual = where(dz < 0., -z, np.divide, np.ones_like, fargs=(dz,))

    a_prim = np.amin(prim) if prim.size else 1.
    a_dual = np.amin(dual) if dual.size else 1.

    return np.minimum(a_prim, a_dual)


def _assemble_KKT(H, g, A, b):
    """
    Assemble the Karush-Kuhn-Tucker system matrix for constrained quadratic optimization.

    References
    ----------
    [1] Nocedal, J., Wright, S. J. (2006). Numerical Optimization. Springer New York.

    Parameters
    ----------
    H : array_like, shape (n, n)
        System matrix with coefficients of the quadratic terms.
    g : array_like, shape (n,)
        System vector with coefficients of the linear terms.
    A : array_like, shape (m, n)
        System matrix with coefficients for the constraints.
    b : array_like, shape (m,)
        Results vector of the constraints.

    Returns
    -------
    K : array_like, shape (n + m, n + m)
        KKT system matrix.
    rhs : array_like, shape (n + m,)
        Right-hand side of the KKT system.
    """

    m = A.shape[0]

    # assemble system matrix
    K = np.block([[H, -A.T], [-A, np.zeros((m, m))]])

    # assemble RHS
    rhs = -np.block([g, b])

    return K, rhs


def _solve_KKT(H, g, A, b):
    """
    Perform a decomposition of the Karush-Kuhn-Tucker system matrix and solve it.

    Parameters
    ----------
    H : array_like, shape (n, n)
        System matrix with coefficients of the quadratic terms.
    g : array_like, shape (n,)
        System vector with coefficients of the linear terms.
    A : array_like, shape (m, n)
        System matrix with coefficients for the equality constraints.
    b : array_like, shape (m,)
        Results vector of the equality constraints.

    Returns
    -------
    xl : array_like, shape (n,)
        Solution to the KKT system containing the solution vector (x) and Lagrangian multipliers (lambda).
    Q : array_like
        Q matrix for QR decomposition.
    R : array_like
        R matrix for QR decomposition.
    """

    # QR decompose KKT matrix, TODO: LDL decomposition instead
    # [L, D, p] = ldl(KKT, 'lower', 'vector')
    # affvec = zeros(1, numel(RHS))
    # affvec(p) = L'\(D\(L\RHS(p)))
    K, rhs = _assemble_KKT(H, g, A, b)

    Q, R = la.qr(K)
    xl = Q @ la.solve(R.T, rhs)

    return xl, Q, R


def _solve_KKT_ip(H, A, C, z, s, rL, rA, rC):
    """
    Perform a decomposition of the Karush-Kuhn-Tucker system matrix and solve it. The system solved
    is the augmented system of a primal dual predictor corrector interior point algorithm.

    References
    ----------
    [1] Nocedal, J., Wright, S. J. (2006). Numerical Optimization. Springer New York.
        Equation (16.61) page 482.

    Parameters
    ----------
    H : array_like, shape (n, n)
        System matrix with coefficients of the quadratic terms.
    A : array_like, shape (n, me)
        System matrix with coefficients for the equality constraints.
    C : array_like, shape (n, mi)
        System matrix with coefficients for the inequality constraints.
    z : array_like, shape (mi,)
        Lagrangian multipliers.
    s : array_like, shape (mi,)
        Slack variables.
    rL : array_like, shape (n,)
        Residuals of the Lagrangian equation.
    rA : array_like, shape (me,)
        Residuals of the equality constraints.
    rC : array_like, shape (mi,)
        Residuals of the inequality constraints.

    Returns
    -------
    dx : array_like, shape (n,)
        Step solution to the primal problem.
    dz : array_like, shape (mi,)
        Step solution to the dual problem.
    ds : array_like, shape (mi,)
        Step solution to the slack variables.
    Q : array_like
        Q matrix for QR decomposition.
    R : array_like
        R matrix for QR decomposition.
    """

    zs = np.diag(z / s)
    H_bar = H + (C.T @ zs @ C)
    rL_bar = rL - C.T @ zs @ (rC - s)

    # solve KKT system
    # TODO: Solve using modified Choleksy instead? [1] page 482.
    sol, Q, R = _solve_KKT(H_bar, rL_bar, A, rA)

    # extract sub-vectors
    n = H.shape[0]
    dx = sol[:n]
    dz = -zs @ C @ dx + zs @ (rC - s)
    ds = -s - np.diag(s / z) @ dz

    return dx, dz, ds, Q, R


def _quad(H, g):
    """
    Solves an unconstrained non-linear quadratic problem of the form::

        min 1/2 x'Hx + g'x

    Parameters
    ----------
    H : array_like, shape (n, n)
        System matrix with coefficients of the quadratic terms.
    g : array_like, shape (n,)
        System vector with coefficients of the linear terms.

    Returns
    -------
    x : array_like, shape (n,)
        Solution to the unconstrained quadratic problem.
    """

    # TODO: use Cholesky factorization
    Q, R = la.qr(H)
    x = Q @ la.solve(R.T, g)

    return x


def _quad_eq(H, g, A, b):
    """
    Solves a non-linear quadratic problem with equality constraints, of the form::

        min 1/2 x'Hx + g'x
        subject to Ax=b

    This problem has a closed form solution based on an LDL decomposition of the KKT system.

    References
    ----------
    [1] Nocedal, J., Wright, S. J., 2006: Numerical Optimization. Springer New York.

    Parameters
    ----------
    H : array_like, shape (n, n)
        System matrix with coefficients of the quadratic terms.
    g : array_like, shape (n,)
        System vector with coefficients of the linear terms.
    A : array_like, shape (n, m)
        System matrix with coefficients for the equality constraints.
    b : array_like, shape (m,)
        Results vector of the equality constraints.

    Returns
    -------
    sol : array_like, shape (n + m,)
        Solution to the equality constrained quadratic problem. Solution vector and Lagrangian multipliers.
    """

    sol, _, _ = _solve_KKT(H, g, A, b)
    return sol


def _quad_iq(H, g, A, b, C, d, x0, y0, z0, s0, ftol=1e-6, etol=1e-6, itol=1e-6, mtol=1e-6, maxiter=1000):
    """
    Solves a non-linear quadratic problem with equality and inequality constraints, of the form::

        min 1/2 x'Hx + g'x
        s.t. Ax = b
        s.t. Cx >= d

    using a primal dual predictor corrector interior point algorithm.

    References
    ----------
    [1] Nocedal, J., Wright, S. J. (2006). Numerical Optimization. Springer New York.
        Page: 484

    Parameters
    ----------
    H : array_like, shape (n, n)
        System matrix with coefficients of the quadratic terms.
    g : array_like, shape (n,)
        System vector with coefficients of the linear terms.
    A : array_like, shape (me, n)
        System matrix with coefficients for the equality constraints.
    b : array_like, shape (me,)
        Results vector of the equality constraints.
    C : array_like, shape (mi, n)
        System matrix with coefficients for the inequality constraints.
    d : array_like, shape (mi,)
        Results vector of the inequality constraints.
    ftol : float, optional
        Tolerance related to change in the objective function.
    etol : float, optional
        Tolerance related to change in the equality constraints.
    itol : float, optional
        Tolerance related to change in the inequality constraints.
    mtol : float, optional
        Tolerance of the penalty parameter.
    maxiter : int, optional
        Maximum number of allowable iterations.

    Returns
    -------
    sol : array_like, shape (n + me + 2mi,)
        Solution to the primal problem (x), dual problem (y), lagrangian multipliers (z) and  slack variables (s).
    status : int

    """

    # initialize -------------------------------------------------------------------------------------------------------
    n = g.size   # number of variables
    me = b.size  # number of equality constraints
    mi = d.size  # number of inequality constraints

    tau_max = 0.995

    # initial value heuristic ------------------------------------------------------------------------------------------
    # assemble and solve KKT system
    rL, rA, rC, rsz = _compute_residuals(H, g, A, b, C, d, x0, y0, z0, s0)
    _, dz_aff, ds_aff, _, _ = _solve_KKT_ip(H, A, C, z0, s0, rL, rA, rC)

    # assign heuristic results prior to iterations
    x = np.array(x0)
    y = np.array(y0)
    z = np.maximum(np.ones(z0.shape), np.abs(z0 + dz_aff))
    s = np.maximum(np.ones(s0.shape), np.abs(s0 + ds_aff))
    mu = np.dot(z, s) / mi
    zs = np.diag(z / s)

    # iterations -------------------------------------------------------------------------------------------------------
    epsL, epsA, epsC, epsM = _define_tolerances(H, g, A, b, C, d, mu, ftol, etol, itol, mtol)
    rL, rA, rC, rsz = _compute_residuals(H, g, A, b, C, d, x, y, z, s)
    converged = _check_convergence(rL, rA, rC, mu, epsL, epsA, epsC, epsM)

    it = 0
    status = 0

    while (not converged) and (it < maxiter):

        # assemble and solve KKT system
        dx_aff, dz_aff, ds_aff, Q, R = _solve_KKT_ip(H, A, C, z, s, rL, rA, rC)

        # compute parameters for affine search
        a_aff = _compute_step_length(s, ds_aff, z, dz_aff)

        # update parameters for duality gap
        mu_aff = np.dot(z + a_aff * dz_aff, s + a_aff * ds_aff) / mi
        sigma = (mu_aff / mu) ** 3.

        # update residuals based on affine step
        rsz_bar = rsz + ds_aff * dz_aff - sigma * mu
        s_bar = rsz_bar / z
        rC_bar = rC - s_bar

        # assembling updated rhs of KKT system
        rLbar = rL - C.T @ zs @ rC_bar
        rhs = -np.block([rLbar, rA])

        # calculate step size
        # TODO: LDL decomposition
        sol = Q @ la.solve(R.T, rhs)

        dx = sol[:n]
        dy = sol[n:]
        dz = -zs @ (C @ dx - rC_bar)
        ds = -s_bar - np.diag(s / z) @ dz

        # check if values are nan or inf
        _sol = np.block([dx, dy, dz, ds])

        if np.any(np.isnan(_sol) | np.isinf(_sol)):
            status = -4
            break

        # calculate step-length
        alpha_p, alpha_d = _max_step_lengths((s, ds), (z, dz), tau=tau_max)

        # based on [1] (page 483) using different primal and dual steps have
        # shown to lead to faster convergence. However, based on eq. (16.65b)
        # it is evident it can also cause divergence for certain choice of
        # alpha_p > alpha_d. One method is proposed which seems to require
        # a constrained least-squares solver (bottom page 483). TODO: implement.
        # until then use: alpha = np.minimum(alpha_p, alpha_d)

        alpha = np.minimum(alpha_p, alpha_d)

        # updating estimator with calculated step-size
        x += dx * alpha  # alpha_p
        y += dy * alpha  # alpha_d
        z += dz * alpha  # alpha_d
        s += ds * alpha  # alpha_p

        # compute residuals and check for convergence
        rL, rA, rC, rsz = _compute_residuals(H, g, A, b, C, d, x, y, z, s)
        mu = np.dot(s, z) / mi
        zs = np.diag(z / s)

        converged = _check_convergence(rL, rA, rC, mu, epsL, epsA, epsC, epsM)

        # let tau_max --> 1 to increase rate of convergence
        if it > 5:
            tau_max += (1. - tau_max) * .5

        it += 1

    if converged:
        status = 1

    sol = np.hstack((x, y, z, s))

    return sol, status, it
