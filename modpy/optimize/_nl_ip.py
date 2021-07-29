import numpy as np
import numpy.linalg as la

from modpy.optimize._nlprog_util import _calculate_hessian, _max_step_lengths, _lagrangian_heuristic
from modpy.optimize._line_search import _line_search_barrier, MeritBarrier
from modpy.optimize._optim_util import OptimizePath


def _nl_ip(obj, x0, jac, hess, con, tol=1e-6, maxiter=1000, keep_path=False):
    """
    Solves a non-linear problem with equality and inequality constraints using an interior point method.

    Given a function `fun`, the algorithm minimizes::

        min f(x)
        s.t. ce(x)=0
        s.t. ci(x)>=0

    References
    ----------
    [1] Nocedal, J., Wright, S. J. (2006). Numerical Optimization. Springer New York.
        Algorithm (19.2): page 577

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

    # evaluate function and constraints
    x = x0
    f = obj(x)
    df = jac(x, f)
    ce, dce = con.eval_eq(x)
    ci, dci = con.eval_iq(x)

    me = ce.size
    mi = ci.size

    # start guess for Lagrangian multipliers and slack variable.
    y = _lagrangian_heuristic(dce, df)
    z = _lagrangian_heuristic(dci, df)
    s = np.ones_like(z)

    # assembling KKT system
    H = np.zeros((n + me + 2 * mi, n + me + 2 * mi))
    H[n:(n + mi), (n + mi + me):] = -np.eye(mi)
    H[(n + mi + me):, n:(n + mi)] = -np.eye(mi)

    # initial of Hessian
    if hess.callable:
        B = _calculate_hessian(x, y, z, hess, con)
    else:
        B = hess.initialize(x0)

    # penalty parameters and merit function
    mu = 5000.   # barrier parameter
    nu = 1000.   # penalty parameter
    phi = MeritBarrier(obj, jac, con, ord_=1)

    # backtracking line-search parameters
    rho = 0.5
    eta = 0.1
    tau = 0.75
    tau_max = 0.995
    sigma = 0.2

    # outer Lagrangian loop
    status = 1
    it = 0
    dL = df - dce @ y - dci @ z
    conv, term = _check_convergence(dL, ce, ci - s, s * z - mu, tol)
    path.append(x, f, *term, mu=mu)

    while (not conv) and (it < np.sqrt(maxiter)):

        # inner barrier loop
        it_mu = 0
        conv_mu = False

        while (not conv_mu) and (it_mu < np.sqrt(maxiter)):

            # update KKT-system
            H[:n, :n] = B
            H[n:(n + mi), n:(n + mi)] = np.diag(z / s)
            H[:n, (n + mi):(n + mi + me)] = dce
            H[(n + mi):(n + mi + me), :n] = dce.T
            H[:n, (n + mi + me):] = dci
            H[(n + mi + me):, :n] = dci.T

            r = np.block([df - dce @ y - dci @ z, z - mu / s, ce, ci - s])

            # solve the KKT system using a direct solver
            p = la.solve(H, -r)

            # check if return values are nan or inf
            if np.any(np.isnan(p) | np.isinf(p)):
                status = -3
                break

            # unpack solutions
            px = p[:n]
            ps = p[n:(n + mi)]
            py = -p[(n + mi):(n + mi + me)]
            pz = -p[(n + mi + me):(n + 2 * mi + me)]

            # calculate lower bound for penalty parameter of the merit function,
            # eq. (18.36)
            pBp = px.T @ B @ px
            sigma_nu = pBp > 0.
            nub = (df.T @ px + sigma_nu / 2. * pBp) / ((1. - rho) * la.norm(np.block([ce, ci]), ord=1))

            if nu < nub:
                nu = nub * 1.1

            # determine maximum step-sizes of the primal and dual problem
            as_max, az_max = _max_step_lengths((s, ps), (z, pz), tau=tau_max)

            # perform backtracking line-search to determine optimal step
            a_s = _line_search_barrier(phi, x, px, s, ps, mu, nu, alpha0=as_max, eta=eta, tau=tau)
            a_z = az_max

            # take optimal step
            x += a_s * px
            s += a_s * ps
            y += a_z * py
            z += a_z * pz

            # BFGS and SR1 requires dL(x_k, y_{k+1}, z_{k+1}) saving prior to updating dL.
            # so storing in case it is required. equation (18.13) of [1].
            dLk = df - dce @ y - dci @ z

            # evaluate function and constraints
            f = obj(x)
            df = jac(x, f)
            ce, dce = con.eval_eq(x)
            ci, dci = con.eval_iq(x)

            # calculate Lagrangian gradient and perform approximate Hessian update
            dL = df - dce @ y - dci @ z

            # calculate Hessian
            if hess.callable:
                # calculate exact hessian
                B = _calculate_hessian(x, y, z, hess, con)

            else:
                # perform approximate Hessian update
                sk = a_s * px
                q = dL - dLk
                B = hess.update(B, sk, q)

            # check convergence
            conv_mu, term = _check_convergence(dL, ce, ci - s, s * z - mu, tol=mu)
            path.append(x, f, *term, mu=mu)

            it_mu += 1

        if status == -3:
            break

        # using the Fiacco-McCormick approach for updating the barrier parameter.
        # using it_mu as a proxy for difficulty of solving inner iteration.
        # if problem is easy, sigma is reduced and tau_max increased to approach
        # super-linear convergence.
        # page 572 of [1].
        if it_mu < 5:
            sigma *= .5
            tau_max += (1. - tau_max) * .5

        mu *= sigma

        # check convergence at new iterate
        conv, _ = _check_convergence(dL, ce, ci - s, s * z, tol=tol)

        it += 1

    if it == maxiter:
        status = 0

    return x, f, status, it, path


def _check_convergence(dL, rA, rC, rsz, tol=1e-6, ord_=np.inf):
    nL = la.norm(dL, ord_)
    nA = la.norm(rA, ord_) if rA.size else 0.
    nC = la.norm(rC, ord_) if rC.size else 0.
    nM = la.norm(rsz, ord_) if rsz.size else 0.

    return np.amax([nL, nA, nC, nM]) < tol, (nL, nA, nC)
