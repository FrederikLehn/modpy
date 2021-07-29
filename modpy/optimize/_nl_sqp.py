import warnings
import numpy as np
import numpy.linalg as la

from modpy.optimize._nlprog_util import _calculate_hessian, _max_step_lengths, _lagrangian_heuristic
from modpy.optimize._line_search import _line_search, _line_search_lagrangian, MeritUncon, MeritNormal, MeritLagrangian
from modpy.optimize._quadprog import quadprog, _quad, _quad_eq, _quad_iq
from modpy.optimize._linprog import linprog
from modpy.optimize._optim_util import OptimizePath
from modpy._exceptions import InfeasibleProblemError


def _check_convergence(dL, df, A, C, y, z, tol=1e-6, ord_=2):
    """
    Perform a check for convergence of a constrained SQP.

    References
    ----------
    [1] Fletcher, R., Leyffer, S. (1998): User manuel for filterSQP.
        Equation (6), page 3.
        Link: https://www.mcs.anl.gov/~leyffer/papers/SQP_manual.pdf

    Parameters
    ----------
    dL : array_like, shape (n,)
        Gradient of the Lagrangian function.
    df : array_like, shape (n,)
        Gradient of the objective function.
    A : array_like, shape (me, n)
        Gradient of the equality constrained functions.
    C : array_like, shape (mi, n)
        Gradient of the inequality constrained functions.
    y : array_like, shape (me,)
        Lagrangian multipliers of the equality constraints.
    z : array_like, shape (me,)
        Lagrangian multipliers of the inequality constraints.
    tol : float, optional
        Tolerance related to change in the Lagrangian objective function.
    ord_ : {1, 2, np.inf}
        Order of the norm used in the convergence check.

    Returns
    -------
    converged : bool
        Check for whether termination criteria is satisfied.
    """

    ndL = la.norm(dL, ord_)
    nA = la.norm(A, ord_, axis=0)
    nC = la.norm(C, ord_, axis=0)

    mu_max = np.amax([la.norm(df, ord_), *(nA * np.abs(y)), *(nC * np.abs(z))])
    r = ndL / np.maximum(mu_max, 1.)

    return r < tol


def _nl_unc(obj, x0, jac, hess, tol=1e-6, maxiter=1000, keep_path=False):
    """
    Solves an unconstrained non-linear problem with a damped BFGS approximation of the Hessian.
    The algorithm uses backtracking and line search in the step decision.

    Given a function `fun`, the algorithm minimizes::

        min f(x)

    References
    ----------
    [1] Nocedal, J, and S J Wright. 2006. Numerical Optimization. Springer New York.

    Parameters
    ----------
    obj : callable
        Objective function to optimize. Returns a vector of shape (n,)
    x0 : array_like, shape (n,)
        Initial guess of the dependent variable.
    jac : callable
        Function for calculating the Jacobian of the objective function. Returns a vector of shape (n,)
    hess : callable
        Function for calculating the Hessian of the objective function. Returns a matrix of shape (n, n)
    tol : float, optional
        Tolerance related to change in the Lagrangian objective function.
    maxiter : int, optional
        Maximum number of allowable iterations.
    keep_path : bool, optional
        Whether to save path information. Can require substantial memory.

    Returns
    -------
    x : array_like, shape (n,)
        Optimal solution vector.
    f : float
        Function value in the optimal point.
    status : int
        Convergence flag.
    nit : int
        Number of iterations.
    """

    path = OptimizePath(keep=keep_path)
    ord_ = 2

    # evaluate function at initial value
    x = np.array(x0)
    f = obj(x)
    df = jac(x, f)
    H = hess.initialize(x0)

    # backtracking line-search parameters and merit function
    phi = MeritUncon(obj, jac)
    eta = 0.1
    tau = 0.75

    ndf = la.norm(df, ord_)
    converged = ndf < tol
    path.append(x, f, ndf)

    status = 1
    it = 0

    while (not converged) and (it < maxiter):

        # solve unconstrained QP
        p = _quad(H, -df)

        # check if return values are nan or inf
        if np.any(np.isnan(p) | np.isinf(p)):
            status = -3
            break

        # perform backtracking line-search to determine optimal step
        alpha = _line_search(phi, x, p, eta=eta, tau=tau)

        # take optimal step of main-problem
        x += alpha * p

        # BFGS and SR1 requires df(x_k) saving prior to updating df.
        # so storing in case it is required. equation (18.13) of reference.
        dfk = np.array(df)

        # evaluate function and constraints
        f = obj(x)
        df = jac(x, f)

        # calculate Hessian
        if hess.callable:
            # calculate exact hessian
            H = hess.calculate(x)

        else:
            # perform approximate Hessian update
            sk = alpha * p
            yk = df - dfk
            H = hess.update(H, sk, yk)

        # convergence
        ndf = la.norm(df, ord_)
        converged = ndf < tol

        if keep_path:
            path.append(x, f, ndf)

        it += 1

    if it == maxiter:
        status = 0

    return x, f, status, it, path


def _nl_sqp(obj, x0, jac, hess, con, merit='lagrangian', tol=1e-6, maxiter=1000, keep_path=False):
    """
    Solves a non-linear problem with equality and inequality constraints.
    The quadratic sub-problem solved in each major iteration is solved
    using a primal-dual predictor corrector interior point algorithm.
    The algorithm is an IQP with warm-start strategy.

    Given a function `fun`, the algorithm minimizes::

        min f(x)
        s.t. ce(x)=0
        s.t. ci(x)>=0

    References
    ----------
    [1] Nocedal, J., and Wright, S. J., 2006: Numerical Optimization. Springer New York.
        Problem formulation: page 536
        Algorithm 18.3: page 545

    [2] Gill, P. E., 2008: User's Guide for SNOPT Version 7.
        Link: https://web.stanford.edu/group/SOL/guides/sndoc7.pdf

    TODO: use constraint handling from: https://web.stanford.edu/group/SOL/guides/sndoc7.pdf (page 10)

    Parameters
    ----------
    obj : callable
        Objective function to optimize. Returns a vector of shape (n,)
    x0 : array_like, shape (n,)
        Initial guess of the dependent variable.
    jac : callable
        Function for calculating the Jacobian of the objective function. Returns a vector of shape (n,)
    hess : callable
        Function for calculating the Hessian of the objective function. Returns a matrix of shape (n, n)
    con : Constraints
        Class Constraints.
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

    Returns
    -------
    x : array_like, shape (n,)
        Optimal solution vector.
    f : float
        Function value in the optimal point.
    status : int
        Convergence flag.
    nit : int
        Number of iterations.
    """

    feasible = _check_linear_feasibility(con, x0.size, tol=tol)
    if not feasible:
        raise InfeasibleProblemError(-2, 'The NLP has infeasible linear constraints.')

    try:

        x, f, status, nit, path = _nl_sqp_normal(obj, x0, jac, hess, con, merit, tol, maxiter, keep_path)

    except InfeasibleProblemError:

        warnings.warn('Infeasible linearization of non-linear constraints. Changing to elastic mode.')
        x, f, status, nit, path = _nl_sqp_elastic(obj, x0, jac, hess, con, merit, tol, maxiter, keep_path)

    return x, f, status, nit, path


def _check_linear_feasibility(con, n, tol=1e-6):
    Al, bl, Cl, dl = con.get_linear(n)
    bounds = con.get_bounds(n)

    me = bl.size
    mi = dl.size
    m = me + mi

    ieq = np.block([np.eye(me), np.zeros((me, mi))])
    iiq = np.block([np.zeros((mi, me)), np.eye(mi)])

    g = np.block([np.zeros((n,)), np.ones((2 * m))])
    A = np.block([Al, -ieq, ieq])
    C = np.block([Cl, -iiq, iiq])

    bounds = (*bounds, *[(0., np.inf) for _ in range(2 * m)])

    # reverse C and dl as linprog takes Cx <= d whereas nlprog takes Cx >= d.
    res = linprog(g, A, bl, -C, -dl, bounds)

    # unconstrained LP, consequently there are no linear constraints
    if res.status == -4:
        return True

    if res.success:
        # f==0 <=> v==0 and w==0 given that v>=0 and w>=0.
        feasible = res.f <= 1e-6
    else:
        feasible = False

    return feasible


def _nl_sqp_normal(obj, x0, jac, hess, con, merit='lagrangian', tol=1e-6, maxiter=1000, keep_path=False):
    """
    Solves a non-linear problem with equality and inequality constraints.
    The quadratic sub-problem solved in each major iteration is solved
    using a primal-dual predictor corrector interior point algorithm.
    The algorithm is an IQP with warm-start strategy.

    Given a function `obj`, the algorithm minimizes::

        min f(x)
        s.t. ce(x)=0
        s.t. ci(x)>=0

    References
    ----------
    [1] Nocedal, J., and Wright, S. J. (2006). Numerical Optimization. Springer New York.
        Problem formulation 18.11: page 533
        Algorithm 18.3: page 545

    [2] Gill, P. E., Murray, W., Saunders, M. A. (2008): User's Guide for SNOPT Version 7.

    Parameters
    ----------
    obj : callable
        Objective function to optimize. Returns a vector of shape (n,)
    x0 : array_like, shape (n,)
        Initial guess of the dependent variable.
    jac : callable
        Function for calculating the Jacobian of the objective function. Returns a vector of shape (n,)
    hess : callable
        Function for calculating the Hessian of the objective function. Returns a matrix of shape (n, n)
    con : Constraints
        Class Constraints.
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

    Returns
    -------
    x : array_like, shape (n,)
        Optimal solution vector.
    f : float
        Function value at the optimal point.
    status : int
        Convergence flag.
    nit : int
        Number of iterations.
    """

    n = x0.shape[0]
    path = OptimizePath(keep=keep_path)
    ord_ = 2

    # evaluate function and constraints
    x = x0
    f = obj(x)
    df = jac(x, f)
    ce, dce = con.eval_eq(x)
    ci, dci = con.eval_iq(x)

    me = ce.size
    mi = ci.size

    # define and evaluate start-guesses
    y = _lagrangian_heuristic(dce, df)
    z = _lagrangian_heuristic(dci, df)
    s = np.ones_like(z)

    # initialize Hessian
    if hess.callable:
        B = _calculate_hessian(x, y, z, hess, con)
    else:
        B = hess.initialize(x0)

    # backtracking line-search parameters and merit function
    if merit in (1, 2, np.inf):

        phi = MeritNormal(obj, jac, con, ord_=merit)

    elif merit == 'lagrangian':

        phi = MeritLagrangian(obj, jac, con, y0=y, z0=z)

    else:

        raise ValueError("`merit` must be either 'lagrangian', 1, 2 or np.inf.")

    mu = 1000.
    rho = 0.5
    eta = 0.1
    tau = 0.75

    # main loop
    converged = False
    status = 1
    it = 0
    sub_it = 0
    dL = df - dce @ y - dci @ z

    if keep_path:
        path.append(x, f, la.norm(dL, ord_), la.norm(y * ce, ord_), la.norm(z * ci, ord_))

    while (not converged) and (it < maxiter):

        # solve quadratic sub-problem to determine step direction.
        if ci.size:  # inequality constrained QP

            sol, sub_status, sub_it = _quad_iq(B, df, dce.T, -ce, dci.T, -ci, x, y, z, s, maxiter=maxiter)

            # TODO: use quadprog, when pre-solver is sufficiently designed
            #warmstart = {'x0': x, 'y0': y, 'z0': z, 's0': s}
            #res = quadprog(B, df, dce.T, -ce, dci.T, -ci, warmstart=warmstart)
            #sub_status = res.status
            #sol = res.sol

            # check if sub-problem is primal or dual infeasible
            if sub_status in (-2, -3):
                raise InfeasibleProblemError(sub_status, 'Infeasible QP')
                #raise InfeasibleProblemError(res.status, res.message)

        else:  # equality constrained QP

            sol = _quad_eq(B, df, dce.T, -ce)

        # check if return values are nan or inf
        if np.any(np.isnan(sol) | np.isinf(sol)):
            status = -4
            break

        # unpack solution vectors
        px = sol[:n]
        py = sol[n:(n+me)]
        pz = sol[(n+me):(n+me+mi)]
        ps = sol[(n+me+mi):]

        pBp = px.T @ B @ px

        # perform backtracking line-search to determine optimal step
        if merit in (1, 2, np.inf):

            # calculate lower bound for penalty parameter of the merit function,
            # [1] eq. (18.36)
            sigma_nu = pBp > 0.
            mub = (df.T @ px + sigma_nu / 2. * pBp) / ((1. - rho) * la.norm(np.block([ce, ci]), ord=1))

            if mu < mub:
                mu = mub * 1.1

            alpha = _line_search(phi, x, px, mu, eta=eta, tau=tau)

        else:

            alpha = _line_search_lagrangian(phi, x, px, py, pz, pBp, eta=eta, tau=tau)

        # take optimal step
        x += alpha * px
        y += alpha * (py - y)
        z += alpha * (pz - z)
        s += alpha * ps

        # BFGS and SR1 requires dL(x_k, y_{k+1}, z_{k+1}) saving prior to updating dL.
        # so storing in case it is required. equation (18.13) of [1].
        dLk = df - dce @ y - dci @ z

        # evaluate function and constraints
        f = obj(x)
        df = jac(x, f)
        ce, dce = con.eval_eq(x)
        ci, dci = con.eval_iq(x)

        # calculate Lagrangian gradient
        dL = df - dce @ y - dci @ z

        # calculate Hessian
        if hess.callable:
            # calculate exact hessian
            B = _calculate_hessian(x, y, z, hess, con)

        else:
            # perform approximate Hessian update
            sk = alpha * px
            q = dL - dLk
            B = hess.update(B, sk, q)

        # check convergence at new iterate
        converged = _check_convergence(dL, df, dce, dci, y, z, tol=tol, ord_=ord_)

        if keep_path:
            nL = la.norm(dL, ord_)
            nA = la.norm(y * ce, ord_)
            nC = la.norm(z * ci, ord_)

            path.append(x, f, nL, nA, nC)  # saving path

        # As the iterations converge towards x, we can retain alphas close to 1.0.
        # for this reason the reduction in alpha in the line search is reduced
        # to achieve better convergence.
        # sub_it is used as a proxy for the difficulty of solving the sub-problem
        # and consequently when the problem is easy to solve
        if sub_it < 5:
            tau += (1. - tau) * .5

            # avoid infinite backtracking loop at tau=1
            tau = np.minimum(0.95, tau)

        it += 1

    if it == maxiter:
        status = 0

    return x, f, status, it, path


def _nl_sqp_elastic(obj, x0, jac, hess, con, merit='lagrangian', tol=1e-6, maxiter=1000, keep_path=False):
    """
    Solves a non-linear problem with equality and inequality constraints.
    The algorithm reformulates the non-linear problem as an l1-penalty problem and uses a
    damped BFGS approximation of the Hessian. The quadratic sub-problem in each iteration
    is solved using a primal-dual predictor corrector interior point algorithm.
    The algorithm is an IQP with warm-start strategy.

    Given a function `fun`, the algorithm minimizes::

        min f(x)
        s.t. ce(x)=0
        s.t. ci(x)>=0

    References
    ----------
    [1] Nocedal, J., Wright, S. J. (2006). Numerical Optimization. Springer New York.
        Problem formulation 18.12: page 536
        Algorithm 18.3: page 545

    [2]  Gill, P. E., Murray, W., Saunders, M. A. (2008). User's Guide for SNOPT Version 7.

    Parameters
    ----------
    obj : callable
        Objective function to optimize. Returns a vector of shape (n,)
    x0 : array_like, shape (n,)
        Initial guess of the dependent variable.
    jac : callable
        Function for calculating the Jacobian of the objective function. Returns a vector of shape (n,)
    hess : callable
        Function for calculating the Hessian of the objective function. Returns a matrix of shape (n, n)
    con : Constraints
        Class Constraints.
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

    Returns
    -------
    x : array_like, shape (n,)
        Optimal solution vector.
    f : float
        Function value in the optimal point.
    status : int
        Convergence flag.
    nit : int
        Number of iterations.
    """

    n = x0.shape[0]
    path = OptimizePath(keep=keep_path)
    ord_ = 2

    # evaluate function and constraints
    x = x0
    f = obj(x)
    df = jac(x, f)
    ce, dce = con.eval_eq(x)
    ci, dci = con.eval_iq(x)

    me = ce.size
    mi = ci.size

    # assembling equality constraints
    A = np.hstack((np.zeros((me, n)), -np.eye(me), np.eye(me), np.zeros((me, mi))))
    y = _lagrangian_heuristic(dce, df)

    # assembling inequality constraints
    # TODO: remove the 3 lower lines and rely on maximum alpha instead?
    C = np.block([[np.zeros((mi, n)), np.zeros((mi, 2 * me)), np.eye(mi)],
                  [np.zeros((me, n)), np.eye(me), np.zeros((me, me + mi))],
                  [np.zeros((me, n + me)), np.eye(me), np.zeros((me, mi))],
                  [np.zeros((mi, n + 2 * me)), np.eye(mi)]])

    d = np.zeros((2 * me + 2 * mi,))
    z = _lagrangian_heuristic(dci, df)
    s = np.ones_like(z)

    # assembling KKT system
    H = np.block([[np.zeros((n, n)), np.zeros((n, 2 * me + mi))],
                  [np.zeros((me, n)), np.eye(me) * 1e-6, np.zeros((me, me + mi))],
                  [np.zeros((me, me + n)), np.eye(me) * 1e-6, np.zeros((me, mi))],
                  [np.zeros((mi, 2 * me + n)), np.eye(mi) * 1e-6]])

    # initial of Hessian
    if hess.callable:
        B = _calculate_hessian(x, y, z, hess, con)
    else:
        B = hess.initialize(x0)

    # assemble vectors used in sub-problem
    vw = np.maximum((ce + dce.T @ x), np.zeros(me))
    v = np.array(vw)
    w = np.array(vw)
    t = np.maximum((ci + dci.T @ x), np.zeros(mi))
    p_sub = np.hstack((x, v, w, t))
    y_sub = np.array(y)
    z_sub = np.hstack((z, np.ones((2 * me + mi,))))
    s_sub = np.ones_like(z_sub)

    del vw

    # penalty parameter
    mu = 5000.  # TODO: 1e4 * (1. + la.norm(df))  # as per [2]? Causes errors (too large).

    # backtracking line-search parameters and merit function
    if merit in (1, 2, np.inf):

        phi = MeritNormal(obj, jac, con, ord_=merit)

    elif merit == 'lagrangian':

        phi = MeritLagrangian(obj, jac, con, y0=y, z0=z)

    else:

        raise ValueError("`merit` must be either 'lagrangian', 1, 2 or np.inf.")

    nu = 1000.
    rho = 0.5
    eta = 0.1
    tau = 0.75
    tau_max = 0.995

    # main loop
    converged = False
    status = 1
    it = 0
    dL = df - dce @ y - dci @ z

    if keep_path:
        path.append(x, f, la.norm(dL, ord_), la.norm(y * ce, ord_), la.norm(z * ci, ord_))

    while (not converged) and (it < maxiter):

        # update quadratic sub-problem matrices
        H[:n, :n] = B
        g = np.block([df, mu * np.ones((2 * me + mi))])
        A[:me, :n] = dce.T
        b = -ce
        C[:mi, :n] = dci.T
        d[:mi] = -ci

        # solve quadratic sub-problem to determine step direction.
        sol, sub_status, sub_it = _quad_iq(H, g, A, b, C, d, p_sub, y_sub, z_sub, s_sub)

        # check if sub-problem failed to converge
        if sub_status < 0:
            status = -2
            break

        # check if return values are nan or inf
        if np.any(np.isnan(sol) | np.isinf(sol)):
            status = -3
            break

        # unpack sub-problem solution vectors
        px_sub = sol[:(n + 2 * me + mi)]
        py_sub = sol[(n + 2 * me + mi):(n + 3 * me + mi)]
        pz_sub = sol[(n + 3 * me + mi):(n + 5 * me + 3 * mi)]
        ps_sub = sol[(n + 5 * me + 3 * mi):]

        # unpack main-problem solution vectors
        px = px_sub[:n]
        py = py_sub[:me]
        pz = pz_sub[:mi]
        ps = ps_sub[:mi]

        # determine maximum step-lengths of the elastic problem
        pv = px_sub[n:(n + me)]
        pw = px_sub[(n + me):(n + 2 * me)]
        pt = px_sub[(n + 2 * me):]
        av, aw, at = _max_step_lengths((v, pv), (w, pw), (t, pt), tau=tau_max)
        a_max = np.amax([av, aw, at])

        pBp = px.T @ B @ px

        # perform backtracking linesearch to determine optimal step
        if merit in (1, 2, np.inf):

            # calculate lower bound for penalty parameter of the merit function,
            # [1] eq. (18.36)
            sigma_nu = pBp > 0.
            nub = (df.T @ px + sigma_nu / 2. * pBp) / ((1. - rho) * la.norm(np.block([ce, ci]), ord=1))

            if nu < nub:
                nu = nub * 1.1

            alpha = _line_search(phi, x, px, nu, alpha0=a_max, eta=eta, tau=tau)

        else:

            alpha = _line_search_lagrangian(phi, x, px, py, pz, pBp, alpha0=a_max, eta=eta, tau=tau)

        # take optimal step of main-problem
        x += alpha * px
        y += alpha * (py - y)
        z += alpha * (pz - z)
        s += alpha * ps

        # use same optimal step for warm-start of next sub-problem
        p_sub += alpha * px_sub
        y_sub += alpha * (py_sub - y_sub)
        z_sub += alpha * (pz_sub - z_sub)
        s_sub += alpha * ps_sub

        # unpack v, w and t to measure their convergence towards 0
        v = p_sub[n:(n+me)]
        w = p_sub[(n+me):(n+2*me)]
        t = p_sub[(n+2*me):]

        # BFGS and SR1 requires dL(x_k, y_{k+1}, z_{k+1}) saving prior to updating dL.
        # so storing in case it is required. equation (18.13) of [1].
        dLk = df - dce @ y - dci @ z

        # evaluate function and constraints
        f = obj(x)
        df = jac(x, f)
        ce, dce = con.eval_eq(x)
        ci, dci = con.eval_iq(x)

        # calculate Lagrangian gradient
        dL = df - dce @ y - dci @ z

        # calculate Hessian
        if hess.callable:
            # calculate exact hessian
            B = _calculate_hessian(x, y, z, hess, con)

        else:
            # perform approximate Hessian update
            sk = alpha * px
            q = dL - dLk
            B = hess.update(B, sk, q)

        # successively increase the penalty parameter of the l1 sub-problem.
        # penalty increased based on [2] page 10 and [1] page 501.
        v_max = la.norm(v, np.inf) if v.size else 0.
        w_max = la.norm(w, np.inf) if w.size else 0.
        t_max = la.norm(t, np.inf) if t.size else 0.
        pen_upd = (v_max < 0.1) & (w_max < 0.1) & (t_max < 0.1)

        if pen_upd:
            # sub_it is used as a proxy for the difficulty of solving the sub-problem.
            if sub_it < 10:
                mu *= 10.
                tau_max += (1. - tau_max) * .5
            else:
                mu *= 1.5

        # check convergence at new iterate
        converged = _check_convergence(dL, df, dce, dci, y, z, tol=tol, ord_=ord_)

        if keep_path:
            nL = la.norm(dL, ord_)
            nA = la.norm(y * ce, ord_)
            nC = la.norm(z * ci, ord_)

            path.append(x, f, nL, nA, nC)  # saving path

        it += 1

    if it == maxiter:
        status = 0

    return x, f, status, it, path
