import numpy as np
import numpy.linalg as la

from modpy._util import where
from modpy.optimize._constraints import prepare_bounds
from modpy.optimize._optim_util import OptimizeResult, _chk_dimensions, _chk_system_dimensions, _ensure_vector, _atleast_zeros
from modpy.optimize._presolve import Presolver
from modpy._exceptions import InfeasibleProblemError


TERMINATION_MESSAGES = {
    -4: 'The LP is unconstrained.',
    -3: 'The LP is dual infeasible.',
    -2: 'The LP is primal infeasible.',
    -1: 'LinAlgError due to indeterminate system.',
    0: 'maximum number of iterations reached.',
    1: 'Tolerance termination condition is satisfied.',
}


def linprog(g, A=None, b=None, C=None, d=None, bounds=None, warmstart=(), tol=1e-6, ctol=1e-6, maxiter=1000):
    """
    Solves a linear programming problem with linear equality and inequality constraints of the form::

        min  g'x
        s.t. Ax = b
        s.t. Cx <= d
        s.t. lb <= x <= ub

    References
    ----------
    [1] Nocedal, J., Wright, S. J. (2006). Numerical Optimization. Springer New York.
    [2] MathWorks
        Link https://se.mathworks.com/help/optim/ug/linear-programming-algorithms.html

    Parameters
    ----------
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
    bounds : 2-tuple of array_like, optional
        Bounds on the solution vector, should be (array_like (n,), array_like (n,)). If None no bounds are applied.
    warmstart : tuple

    tol : float, optional
        Tolerance of the dual problem as well as duality gap.
    ctol : float, optional
        Tolerance of the primal problem as well as the complimentary conditions.
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

    # ensure consistent sizing of input --------------------------------------------------------------------------------
    n = g.size
    A = _atleast_zeros(A, size=(0, n))
    C = _atleast_zeros(C, size=(0, n))
    b = _atleast_zeros(b, size=(0,))
    d = _atleast_zeros(d, size=(0,))

    b = _ensure_vector(b, 'b')
    d = _ensure_vector(d, 'd')

    A, _ = _chk_dimensions(A, g, 'A', 'g')
    C, _ = _chk_dimensions(C, g, 'C', 'g')

    _chk_system_dimensions((A.T, C.T), (b, d), (g,))

    # solve LP ---------------------------------------------------------------------------------------------------------
    lb, ub = prepare_bounds(bounds, n)
    LP = Presolver(g, A, b, C, d, lb, ub)

    x = None
    f = None
    sol = None
    message = ''
    nit = 0

    try:

        if (not A.size) and (not C.size) and (bounds is None):  # unconstrained

            status = -4
            message = TERMINATION_MESSAGES[status]

        else:  # constrained

            # pre-solve the LP
            LP.presolve_LP()
            g, A, b, C, d, lb, ub = LP.get_LP()

            # solve the remainder of the problem
            x, sol, status, nit = _lin_ip(g, A, b, C, d, ub, tol=tol, ctol=ctol, maxiter=maxiter)

    except InfeasibleProblemError as e:

        status = e.status
        message = e.message

    except la.LinAlgError:

        status = -1
        message = TERMINATION_MESSAGES[status]

    if status > 0:

        # post-solve the LP
        x, f = LP.postsolve(x)
        message = TERMINATION_MESSAGES[status]

    res = OptimizeResult(x, f, sol, status=status, message=message, nit=nit)
    res.success = status > 0

    return res


def _check_convergence(x, t, v, w, rd, rp, rub, nu, rho, tol=1e-6, ctol=1e-6):
    prim = la.norm(rp, ord=1) + la.norm(rub, ord=1)
    dual = la.norm(rd, ord=np.inf)

    rc1 = np.amin([np.abs(x * v), np.abs(x), np.abs(v)], axis=0)

    if nu:
        rc2 = np.amin([np.abs(t * w), np.abs(t), np.abs(w)], axis=0)
    else:
        rc2 = [0.]

    comp = np.amax([*rc1, *rc2])

    return (prim <= (rho * ctol)) & (dual < (rho * tol)) & (comp < tol)


def _compute_residuals(gbar, Abar, bbar, x, y, t, v, w, ub, nu):
    n = gbar.size

    rd = gbar - Abar.T @ y - v + _bound_app(w, nu, n)
    rp = Abar @ x - bbar
    rub = ub[:nu] - x[:nu] - t
    rvx = v * x
    rwt = w * t

    return rd, rp, rub, rvx, rwt


def _compute_step_length(x, dx, t, dt, v, dv, w, dw, eta=1.):
    """
    Computes the step-length for an LP interior point method.

    References
    ----------
    [1] MathWorks
        Link: Link https://se.mathworks.com/help/optim/ug/linear-programming-algorithms.html

    Parameters
    ----------
    x : array_like, shape (n,)
        Solution to the primal problem.
    dx : array_like, shape (n,)
        Step solution to the primal problem.

    Returns
    -------
    a_prim : float
        Optimal primal step-length.
    a_dual : float
        Optimal dual step-length.
    """

    axv = where(dx < 0., -x, np.divide, np.ones_like, fargs=(dx,))
    atv = where(dt < 0., -t, np.divide, np.ones_like, fargs=(dt,))
    avv = where(dv < 0., -v, np.divide, np.ones_like, fargs=(dv,))
    awv = where(dw < 0., -w, np.divide, np.ones_like, fargs=(dw,))

    ax = np.minimum(eta * np.amin(axv), 1.) if axv.size else 1.
    at = np.minimum(eta * np.amin(atv), 1.) if atv.size else 1.
    av = np.minimum(eta * np.amin(avv), 1.) if avv.size else 1.
    aw = np.minimum(eta * np.amin(awv), 1.) if awv.size else 1.

    return np.minimum(ax, at), np.minimum(av, aw)


def _assemble_KKT_sparse(Abar, Dinv, R, rp):

    AbarDinv = Abar @ Dinv
    KKT = AbarDinv @ Abar.T
    rhs = -(AbarDinv @ R + rp)

    return KKT, rhs


def _back_substitute_sparse(dy, Abar, Dinv, R, x, t, v, w, rub, rvx, rwt, nu):
    # back substitute to find dx, dv, dw and dt

    dx = Dinv @ (Abar.T @ dy + R)  # [1] has written this incorrect. Current implementation is correct.
    dt = rub - dx[:nu]
    dv = -(rvx + v * dx) / x
    dw = -(rwt + w * dt) / t

    return dx, dt, dv, dw


def _solve_KKT_sparse(Abar, x, t, v, w, rd, rp, rub, rvx, rwt, nu):
    n = x.size

    Dinv = np.diag(1. / (v / x + _bound_app(w / t, nu, n)))
    R = -rd - v + _bound_app(w + rub * w / t, nu, n)  # notice: rvx / x = v and rwt / t = w.
    # TODO: Paper ref does not have - in front of rd?

    KKT, rhs = _assemble_KKT_sparse(Abar, Dinv, R, rp)

    # QR decompose KKT matrix, TODO: Cholesky decomposition
    Q, R_ = la.qr(KKT)

    # solve for dy
    dy = Q @ la.solve(R_.T, rhs)  # TODO: Cholesky solve
    dx, dt, dv, dw = _back_substitute_sparse(dy, Abar, Dinv, R, x, t, v, w, rub, rvx, rwt, nu)

    return dx, dy, dt, dv, dw, Q, R_


def _bound_app(xu, nu, n):
    """
    Operator which assigns zeros to vectors corresponding to unbounded upper bounds (np.inf)
    in order to map it to the appropriate problem dimension

    References
    ----------
    [1] Zhang, Y. (1996). Solving Large-Scale Linear Programs by Interior-Point Methods under the MATLAB Environment.
        Page: 3

    Parameters
    ----------
    xu : array_like, shape (n+mi,)
        Vector to be mapped.
    nu : int
        Number of variables with ub != np.inf
    n : int
        Problem dimension.

    Returns
    -------
    x : array_like, shape (n,)
        Vector with 0's at location of unbounded variables.
    """

    x = np.zeros((n,))
    x[:nu] = xu[:nu]

    return x


def _pivot_on_bounds(g, A, C, ub):
    """
    Pivots the system matrices such that the variables are given in a sorted order based on their upper ound
    with all ub=np.inf being positioned at the end.

    Parameters
    ----------
    g : array_like, shape (n,)
        System vector with coefficients of the linear terms.
    A : array_like, shape (m, n)
        System matrix with coefficients for the inequality constraints.
    C : array_like, shape (mi, n)
        System matrix with coefficients for the inequality constraints.
    ub : array_like, shape (n,)
        Upper bound of x.

    Returns
    -------
    g : array_like, shape (n,)
        System vector with coefficients of the linear terms.
    A : array_like, shape (m, n)
        System matrix with coefficients for the inequality constraints.
    C : array_like, shape (mi, n)
        System matrix with coefficients for the inequality constraints.
    ub : array_like, shape (n,)
        Upper bound of x.
    p : array_like, shape (n,)
        Pivot vector
    nu : int
        Number of variables with ub != np.inf
    """

    p = np.argsort(ub)
    g = g[p]
    A = A[:, p]
    C = C[:, p]
    ub = ub[p]

    if np.inf in ub:
        nu = np.argmax(ub == np.inf)
    else:
        nu = g.size

    return g, A, C, ub, p, nu


def _lin_ip(g, A, b, C, d, ub, tol=1e-6, ctol=1e-6, maxiter=1000):
    """
    Solves a linear programming problem with linear equality and inequality constraints of the form::

        min  g'x
        s.t. Ax = b
        s.t. Cx <= d
        s.t. lb <= x <= ub

    using an interior point algorithm.

    The majority of the algorithm is based on [1], but modified to include inequality constraints as per [2].
    [3] serves as general inspiration and understanding.

    References
    ----------
    [1] Zhang, Y. (1996). Solving Large-Scale Linear Programs by Interior-Point Methods under the MATLAB Environment.
    [2] Linear Programming Algorithms. MathWorks.
    [3] Nocedal, J., Wright, S. J. (2006). Numerical Optimization. Springer New York.

    Parameters
    ----------
    g : array_like, shape (n,)
        System vector with coefficients of the linear terms.
    A : array_like, shape (m, n)
        System matrix with coefficients for the inequality constraints.
    b : array_like, shape (m,)
        Results vector of the inequality constraints.
    C : array_like, shape (mi, n)
        System matrix with coefficients for the inequality constraints.
    d : array_like, shape (mi,)
        Results vector of the inequality constraints.
    ub : array_like, shape (n,)
        Upper bound of x.
    tol : float, optional
        Tolerance of the dual problem as well as duality gap.
    ctol : float, optional
        Tolerance of the primal problem as well as the complimentary conditions.
    maxiter : int, optional
        Maximum number of allowable iterations.

    Returns
    -------
    sol : array_like, shape (n + me + 2mi,)
        Solution to the primal problem (x), dual problem (y), lagrangian multipliers (z) and  slack variables (s).
    status : int
    """

    n = g.size
    me = b.size
    mi = d.size
    nmi = n + mi
    eta = 0.99

    # pivot problem in ascending order of upper bound value
    g, A, C, ub, p, nu = _pivot_on_bounds(g, A, C, ub)

    # prepare merged constraint system ---------------------------------------------------------------------------------
    gbar = np.block([g, np.zeros((mi,))])  # zeros for slack variables

    Abar = np.block([[A, np.zeros((me, mi))],
                     [C, np.eye(mi)]])

    bbar = np.block([b, d])

    # initial value heuristic ------------------------------------------------------------------------------------------
    x0 = np.where(ub < np.inf, ub / 2., np.ones((n,)))
    s0 = np.ones((mi,))
    y0 = np.zeros((me + mi,))
    t0 = np.ones((nu,))
    v0 = np.ones((nmi,))
    w0 = np.ones((nu,))

    # stack x0 and s0
    x0 = np.block([x0, s0])

    # take one predictor step
    rd, rp, rub, rvx, rwt = _compute_residuals(gbar, Abar, bbar, x0, y0, t0, v0, w0, ub, nu)
    dx_aff, dy_aff, dt_aff, dv_aff, dw_aff, _, _ = _solve_KKT_sparse(Abar, x0, t0, v0, w0, rd, rp, rub, rvx, rwt, nu)

    # assign heuristic results prior to iterations
    x = np.array(x0)
    y = np.array(y0)
    t = np.maximum(np.ones(t0.shape), np.abs(t0 + dt_aff))
    v = np.maximum(np.ones(v0.shape), np.abs(v0 + dv_aff))
    w = np.maximum(np.ones(w0.shape), np.abs(w0 + dw_aff))

    # parameters
    sigma_max = 0.208  # [1], no strong justification for this value
    rho = np.amax([1., la.norm(Abar), la.norm(gbar), la.norm(bbar)])

    # main loop --------------------------------------------------------------------------------------------------------
    converged = False

    it = 0
    status = 1

    while (not converged) and (it < maxiter):

        # perform predictor step
        dx_aff, dy_aff, dt_aff, dv_aff, dw_aff, Q, R_ = _solve_KKT_sparse(Abar, x, t, v, w, rd, rp, rub, rvx, rwt, nu)

        # calculate affine step-length
        a_prim_aff, a_dual_aff = _compute_step_length(x, dx_aff, t, dt_aff, v, dv_aff, w, dw_aff)

        # calculate duality gaps
        vx = ((x + a_prim_aff * dx_aff).T @ (v + a_dual_aff * dv_aff))
        wt = ((w + a_dual_aff * dw_aff).T @ (t + a_prim_aff * dt_aff))

        # calculate centering parameters
        mu = (np.dot(v, x) + np.dot(w, t)) / nmi
        mu_aff = (vx + wt) / (n + mi)
        sigma = np.minimum((mu_aff / mu) ** 3., sigma_max)

        # update RHS for corrector step
        rvx_cor = rvx + dv_aff * dx_aff - sigma * mu  # [1] does not include rvx + at front, this is a mistake
        rwt_cor = rwt + dw_aff * dt_aff - sigma * mu  # [1] does not include rwt + at front, this is a mistake
        Dinv = np.diag(1. / (v / x + _bound_app(w / t, nu, nmi)))
        R = -rd - rvx_cor / x + _bound_app(rwt_cor / t + rub * w / t, nu, nmi)

        rhs = -(Abar @ Dinv @ R + rp)

        # solve corrector step and extract increments
        dy = Q @ la.solve(R_.T, rhs)
        dx, dt, dv, dw = _back_substitute_sparse(dy, Abar, Dinv, R, x, t, v, w, rub, rvx_cor, rwt_cor, nu)

        # calculate step-length
        a_prim, a_dual = _compute_step_length(x, dx, t, dt, v, dv, w, dw, eta=eta)

        # take optimal step
        x += a_prim * dx
        y += a_dual * dy
        t += a_prim * dt
        v += a_dual * dv
        w += a_dual * dw

        # convergence check
        rd, rp, rub, rvx, rwt = _compute_residuals(gbar, Abar, bbar, x, y, t, v, w, ub, nu)
        converged = _check_convergence(x, t, v, w, rd, rp, rub, nu, rho, tol=tol, ctol=ctol)

        # let eta --> 1 to increase rate of convergence
        if it > 5:
            eta += (1. - eta) * .5
            # TODO: reduce sigma_max towards relative tolerance

        it += + 1

    if it == maxiter:
        status = 0

    # split merged constraint system back into original form
    x_ = x[:n]
    s = x[n:]

    # pivot solution back to original ordering
    x_ = x_[p]

    # pivot upper bound Lagrangian multiplier back to original ordering
    # TODO: Is this correct? Is it even required?
    t_ = _bound_app(t, nu, n)
    t_[p[:nu]] = t[:nu]
    t = t_[:nu]

    sol = np.hstack((x_, s, y, t, v, w))

    return x_, sol, status, it



# def _check_convergence(rb, rc, mu, tol):
#     cb = la.norm(rb, ord=np.inf) <= tol
#     cc = la.norm(rc, ord=np.inf) <= tol
#     cmu = abs(mu) <= tol
#
#     return cb and cc and cmu
#
#
# def _compute_residuals(g, A, b, x, y, s):
#     n = g.size
#
#     rb = A @ x - b
#     rc = A.T @ y + s - g
#     mu = (x.T @ s) / n
#
#     return rb, rc, mu


# def _initial_value_heuristic(g, A, b):
#
#     AA = A @ A.T
#
#     xt = A.T @ la.solve(AA, b)
#     yt = la.solve(AA, A @ g)
#     st = g - A.T @ yt
#
#     dx = np.maximum(-1.5 * xt.min(), 0.)
#     ds = np.maximum(-1.5 * st.min(), 0.)
#     xh = xt + dx
#     sh = st + ds
#
#     xhsh = .5 * xh.T @ sh
#     dx_h = xhsh / np.sum(sh)
#     ds_h = xhsh / np.sum(xh)
#
#     x = xh + dx_h
#     y = yt
#     s = sh + ds_h
#
#     return x, y, s

# def _compute_step_length(x, dx, s, ds, eta=1.):
#     """
#     Computes the step-length for an LP interior point method.
#
#     References
#     ----------
#     [1] Nocedal, J., and Wright, S. J. (2006). Numerical Optimization. Springer New York.
#         Pages: 408-409
#
#     Parameters
#     ----------
#     x : array_like, shape (n,)
#         Solution to the primal problem.
#     dx : array_like, shape (n,)
#         Step solution to the primal problem.
#     s : array_like, shape (n,)
#         Slack variables.
#     ds : array_like, shape (n,)
#         Step solution to the slack variables.
#
#     Returns
#     -------
#     a_prim : float
#         Primal step-length.
#     a_dual : float
#         Dual step-length.
#     """
#
#     prim = where(dx < 0., -x, np.divide, np.ones_like, fargs=(dx,))
#     dual = where(ds < 0., -s, np.divide, np.ones_like, fargs=(ds,))
#
#     a_prim = np.minimum(eta * np.amin(prim), 1.) if prim.size else 1.
#     a_dual = np.minimum(eta * np.amin(dual), 1.) if dual.size else 1.
#
#     return a_prim, a_dual


# def _assemble_KKT(A, x, s, rb, rc):
#     """
#     Assemble the Karush-Kuhn-Tucker system matrix for constrained quadratic optimization.
#
#     Parameters
#     ----------
#     A : array_like, shape (m, n)
#         System matrix for the constraints.
#     x : array_like, shape (n,)
#         Solution vector.
#     s : array_like, shape (n,)
#         Slack variables.
#     rb : array_like, shape (m,)
#         Residual vector of the primary problem.
#     rc : array_like, shape (m,)
#         Residual vector of the dual problem.
#
#     Returns
#     -------
#     K : array_like, shape (2 * n + m, 2 * n + m)
#         KKT system matrix.
#     rhs : array_like, shape (2 * m + n)
#         Right-hand side of the KKT system.
#     """
#
#     m, n = A.shape
#
#     # assemble system matrix
#     K = np.block([[np.zeros((n, n)), A.T, np.eye(n)],
#                   [A, np.zeros((m, m)), np.zeros((m, n))],
#                   [np.diag(s), np.zeros((n, m)), np.diag(x)]])
#
#     # assemble RHS
#     rhs = -np.block([rc, rb, x * s])
#
#     return K, rhs
#
#
# def _solve_KKT(A, x, s, rb, rc):
#     """
#     Perform a decomposition of the Karush-Kuhn-Tucker system matrix and solve it.
#
#     Parameters
#     ----------
#     A : array_like, shape (m, n)
#         System matrix for the constraints.
#     x : array_like, shape (n,)
#         Solution vector.
#     s : array_like, shape (n,)
#         Slack variables.
#     rb : array_like, shape (n,)
#         Residual vector of the primary problem.
#     rc : array_like, shape (n,)
#         Residual vector of the dual problem.
#
#     Returns
#     -------
#     xl : array_like, shape (n,)
#         Solution to the primary problem.
#     y : array_like, shape (m,)
#         Solution to the dual problem.
#     s : array_like, shape (n,)
#         Slack variable.
#     Q : array_like
#         Q matrix for QR decomposition.
#     R : array_like
#         R matrix for QR decomposition.
#     """
#
#     K, rhs = _assemble_KKT(A, x, s, rb, rc)
#
#     # QR decompose KKT matrix, TODO: LDL decomposition instead
#     # [L, D, p] = ldl(KKT, 'lower', 'vector')
#     # affvec = zeros(1, numel(RHS))
#     # affvec(p) = L'\(D\(L\RHS(p)))
#     Q, R = la.qr(K)
#     sol = Q @ la.solve(R.T, rhs)
#
#     m, n = A.shape
#     x = sol[:n]
#     y = sol[n:(n+m)]
#     s = sol[(n+m):]
#
#     return x, y, s, Q, R
#
#
# def _lin_ip(g, A, b, tol=1e-6, maxiter=1000):
#     """
#     Solves a linear programming problem with linear equality and inequality constraints of the form::
#
#         min g'x
#         s.t. Ax <= b
#
#     using an interior point algorithm.
#
#     References
#     ----------
#     [1] Nocedal, J., and Wright, S. J. (2006). Numerical Optimization. Springer New York.
#         Algorithm 14.3: page 411
#
#     Parameters
#     ----------
#     g : array_like, shape (n,)
#         System vector with coefficients of the linear terms.
#     A : array_like, shape (m, n)
#         System matrix with coefficients for the inequality constraints.
#     b : array_like, shape (m,)
#         Results vector of the inequality constraints.
#     tol : float, optional
#         Tolerance of the primal and dual problem as well as duality gap.
#     maxiter : int, optional
#         Maximum number of allowable iterations.
#
#     Returns
#     -------
#     sol : array_like, shape (n + me + 2mi,)
#         Solution to the primal problem (x), dual problem (y), lagrangian multipliers (z) and  slack variables (s).
#     status : int
#     """
#
#     m, n = A.shape
#     eta = 0.99
#
#     # initial value correction heuristic -------------------------------------------------------------------------------
#     AA = A @ A.T
#
#     xt = A.T @ la.solve(AA, b)
#     yt = la.solve(AA, A @ g)
#     st = g - A.T @ yt
#
#     dx = np.maximum(-1.5 * xt.min(), 0.)
#     ds = np.maximum(-1.5 * st.min(), 0.)
#     xh = xt + dx
#     sh = st + ds
#
#     xhsh = .5 * xh.T @ sh
#     dx_h = xhsh / np.sum(sh)
#     ds_h = xhsh / np.sum(xh)
#
#     x = xh + dx_h
#     y = yt
#     s = sh + ds_h
#     mu = 1000.
#
#     del AA, xt, yt, st, dx, ds, xh, sh, xhsh, dx_h, ds_h
#
#     # main loop --------------------------------------------------------------------------------------------------------
#     rb, rc, _ = _compute_residuals(g, A, b, x, y, s)
#     converged = False
#
#     it = 0
#     status = 1
#
#     while (not converged) and (it < maxiter):
#
#         # assemble and solve KKT system
#         dx_aff, dy_aff, ds_aff, Q, R = _solve_KKT(A, x, s, rb, rc)
#
#         # calculate affine step-length
#         a_prim_aff, a_dual_aff = _compute_step_length(x, dx_aff, s, ds_aff)
#
#         # calculate duality gap parameters
#         mu_aff = ((x + a_prim_aff * dx_aff).T @ (s + a_dual_aff * ds_aff)) / n
#         sigma = (mu_aff / mu) ** 3.
#         print(mu, mu_aff, sigma)
#
#         # update RHS for second linear solve
#         rhs = np.block([-rc, -rb, -x * s - dx_aff * ds_aff + sigma * mu])
#
#         # solve and extract increments
#         dv = Q @ la.solve(R.T, rhs)
#         dx = dv[:n]
#         dy = dv[n:(n+m)]
#         ds = dv[(n+m):]
#
#         # calculate step-length
#         a_prim, a_dual = _compute_step_length(x, dx, s, ds, eta=eta)
#
#         # take optimal step
#         x += a_prim * dx
#         y += a_dual * dy
#         s += a_dual * ds
#
#         # convergence check
#         rb, rc, mu = _compute_residuals(g, A, b, x, y, s)
#         converged = _check_convergence(rb, rc, mu, tol)
#
#         # let eta --> 1 to increase rate of convergence
#         if it > 5:
#             eta += (1. - eta) * .5
#
#         it += + 1
#
#     if it == maxiter:
#         status = 0
#
#     sol = np.hstack((x, y, s))
#
#     return sol, status, it
