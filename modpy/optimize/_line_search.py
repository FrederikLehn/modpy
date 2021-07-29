import numpy as np
import numpy.linalg as la

from modpy.optimize._lsq import lsq_linear


def _line_search(phi, x, px, mu=0., alpha0=1.0, eta=0.1, tau=0.75, maxiter=100, args=()):
    """
    Calculate the optimal step-size based on a backtracking line-search algorithm.

    References
    ----------
    [1] Nocedal, J., and Wright, S. J. (2006). Numerical Optimization. Springer New York.
        Pages 435, 437, 540-543

    Parameters
    ----------
    phi : Class MeritFunction
        Merit function.
    x : array_like, (n,)
        Current solution proposal.
    px : array_like, (n,)
        Solution to the sub-problem.
    mu : array_like, (mi,), optional
        Penalty parameter.
    alpha0 : float, optional
        Start-guess or maximum value of `a`.
    eta : float, optional
        Sufficient decrease condition parameter.
    tau : float, optional
        Reduction of step-size during backtracking.
    maxiter : int, optional
        Maximum number of iterations.
    args : tuple, optional
        Additional arguments to `phi`.

    Returns
    -------
    alpha : float
        Step-size.
    """

    # calculate merit function and directional merit function
    # at the current iterate `x` and slack variable `s`.
    phi0 = phi.eval(x, mu)
    D0 = phi.grad(x, px, mu)

    # iteratively backtrack until sufficient decrease conditions are met.
    alpha = alpha0
    it = 0

    while it < maxiter:

        # calculate next guess for the optimal step.
        xb = x + alpha * px

        if phi.eval(xb, mu) > (phi0 + eta * alpha * D0):
            alpha *= tau
            it += 1

            # terminate prematurely if alpha is too small
            if alpha < 0.01:
                it = maxiter
                break

        else:
            break

    if it == maxiter:
        alpha = alpha0

    return alpha


def _line_search_lagrangian(phi, x, px, py, pz, pHp, alpha0=1.0, eta=0.1, tau=0.75, maxiter=100, args=()):
    """
    Calculate the optimal step-size based on a backtracking line-search algorithm
    using a variation of an augmented lagrangian merit function.

    References
    ----------
    [1] Nocedal, J., and Wright, S. J. (2006). Numerical Optimization. Springer New York.
        Pages 435, 437, 540-543

    Parameters
    ----------
    phi : MeritLagrangian
        Merit function.
    x : array_like, (n,)
        Current solution proposal.
    px : array_like, (n,)
        Solution to the sub-problem.
    alpha0 : float, optional
        Start-guess or maximum value of `a`.
    eta : float, optional
        Sufficient decrease condition parameter.
    tau : float, optional
        Reduction of step-size during backtracking.
    maxiter : int, optional
        Maximum number of iterations.
    args : tuple, optional
        Additional arguments to `phi`.

    Returns
    -------
    alpha : float
        Step-size.
    """

    me = py.size

    # unpack saved variables from previous iteration
    lam = phi.lam       # previous iterate of lambda.
    rho = phi.rho       # previous iterate of rho

    # mu is the Lagrangian multipliers
    mu = np.block([py, pz])

    # calculate xi
    xi = mu - lam

    # evaluate constraints and derivatives at initial point
    ce, dce = phi.con.eval_eq(x)
    ci, dci = phi.con.eval_iq(x)

    # calculate slacks
    si = np.maximum(np.zeros_like(ci), ci - lam[me:] / rho if rho != 0. else 0.)
    s = np.block([np.zeros((me,)), si])
    cs = np.block([ce, ci]) - s

    # calculate q
    q = np.vstack((dce.T, dci.T)) @ px + cs

    # calculate merit function and derivative of merit function at previous iterate
    phi0 = phi.eval(x, lam, cs, rho)
    dphi0 = phi.grad(pHp, xi, cs, q, mu, rho)

    # update rho
    if dphi0 > (-pHp / 2.):

        if (pHp / 2.) <= (-2. * cs.T @ xi):
            rho_hat = 2. * la.norm(xi) / la.norm(cs)
        else:
            rho_hat = 0.

        rho = np.maximum(rho_hat, 2. * rho)

    # iteratively backtrack until sufficient decrease conditions are met.
    alpha = alpha0
    it = 0

    while it < maxiter:

        # calculate next guess for the optimal step.
        xb = x + alpha * px
        lamb = lam + alpha * xi
        sb = s + alpha * q

        csb = np.block([phi.con.f_eq(xb), phi.con.f_iq(xb)]) - sb

        if phi.eval(xb, lamb, csb, rho) > (phi0 + eta * alpha * dphi0):
            alpha *= tau
            it += 1

            # terminate prematurely if alpha is too small
            if alpha < 0.01:
                it = maxiter
                break

        else:
            phi.lam = lamb
            phi.rho = rho
            break

    if it == maxiter:
        alpha = alpha0

    return alpha


def _line_search_barrier(phi, x, px, s, ps, mu, nu, alpha0=1.0, eta=0.1, tau=0.75, maxiter=100, args=()):
    """
    Calculate the optimal step-size based on a backtracking line-search algorithm for a barrier problem.

    References
    ----------
    [1] Nocedal, J., and Wright, S. J. (2006). Numerical Optimization. Springer New York.
        Pages 435, 437, 540-543

    Parameters
    ----------
    phi : Class MeritFunction
        Merit function.
    x : array_like, (n,)
        Current solution proposal.
    px : array_like, (n,)
        Solution to the sub-problem.
    s : array_like, (mi,)
        Current slack variable.
    ps : array_like, (mi,)
        Slack variable solution to the sub-problem.
    mu : float
        Barrier parameter.
    nu : float
        Penalty parameter.
    alpha0 : float
        Start-guess or maximum value of `a`.
    eta : float
        Sufficient decrease condition parameter.
    tau : float
        Reduction of step-size during backtracking.
    maxiter : int
        Maximum number of iterations.
    args : tuple
        Additional arguments to `phi`.

    Returns
    -------
    alpha : float
        Step-size.
    """

    # calculate merit function and directional merit function
    # at the current iterate `x` and slack variable `s`.
    phi0 = phi.eval(x, s, mu, nu)
    D0 = phi.grad(x, px, s, ps, mu, nu)

    # iteratively backtrack until sufficient decrease conditions are met.
    alpha = alpha0
    it = 0

    while it < maxiter:

        # calculate next guess for the optimal step.
        xb = x + alpha * px
        sb = s + alpha * ps

        if phi.eval(xb, sb, mu, nu) > (phi0 + eta * alpha * D0):
            alpha *= tau
            it += 1

            # terminate prematurely if alpha is too small
            if alpha < 0.01:
                it = maxiter
                break

        else:
            break

    if it == maxiter:
        alpha = alpha0

    return alpha


class MeritFunction:
    def __init__(self, obj, jac, con=None, ord_=1):

        self.obj = obj      # callable, objective function
        self.jac = jac      # callable, Jacobian of objective function
        self.con = con      # class, constraints

        if ord_ not in (1, 2, np.inf):
            raise ValueError('`ord` must be either 1, 2 or np.inf.')

        self.ord = ord_     # int, order of norm


class MeritUncon(MeritFunction):
    def __init__(self, obj, jac):
        super().__init__(obj, jac)

    def eval(self, x, mu):
        return self.obj(x)

    def grad(self, x, px, mu):
        return self.jac(x, self.obj(x)).T @ px


class MeritNormal(MeritFunction):
    def __init__(self, obj, jac, con, ord_=1):
        super().__init__(obj, jac, con, ord_=ord_)

    def eval(self, x, mu):
        return self.obj(x) + mu * self.con_norm(x)

    def grad(self, x, px, mu):
        """
        Given that this merit function is non-smooth it is also not differentiable.
        Instead the directional derivative is calculated.
        """

        return self.jac(x, self.obj(x)).T @ px - mu * self.con_norm(x)

    def con_norm(self, x):
        ce = self.con.f_eq(x)
        ci = np.maximum(0., -self.con.f_iq(x))
        return _merit_norm(ce, ci, ord_=self.ord)


class MeritLagrangian(MeritFunction):
    """
    A variation of an Augmented Lagrangian Merit Function.

    References
    ----------
    [1] Gill, P. E., Murray, W., Saunders, M. A. (1986). Some Theoretical Properties of an Augmented Lagrangian Merit Function.
    """

    def __init__(self, obj, jac, con, y0, z0):
        super().__init__(obj, jac, con)

        self.lam = np.block([y0, z0])
        self.rho = 0.

    def eval(self, x, lam, cs, rho):
        return self.obj(x) - lam.T @ cs + rho / 2. * cs.T @ cs

    @staticmethod
    def grad(pHp, xi, cs, q, mu, rho):
        return -pHp + q.T @ mu - 2. * cs.T @ xi - rho * cs.T @ cs


class MeritBarrier(MeritFunction):
    def __init__(self, obj, jac, con, ord_=1):
        super().__init__(obj, jac, con, ord_=ord_)

    def eval(self, x, s, mu, nu):
        return self.obj(x) - mu * np.sum(np.log(s)) + nu * self.con_norm(x, s)

    def grad(self, x, px, s, ps, mu, nu):
        """
        Given that this merit function is non-smooth it is also not differentiable.
        Instead the directional derivative is calculated.
        """
        # TODO: unsure of calculation below
        return self.jac(x, self.obj(x)).T @ px - np.inner(1. / s, ps) - nu * self.con_norm(x, s)

    def con_norm(self, x, s):
        ce = self.con.f_eq(x)
        ci = self.con.f_iq(x) - s
        return _merit_norm(ce, ci, ord_=self.ord)


class FilterFunction:
    def __init__(self):
        pass


def _merit_norm(ce, ci, ord_=1):
    """
    Calculate the constraint violations of the merit-function for a standard formulation algorithm.

    Reference: Nocedal, J, and S J Wright. 2006. Numerical Optimization. Springer New York.
               Page: 435

    Parameters
    ----------
    ce : array_like, (me,)
        Evaluation of equality constraints.
    ci : array_like, (mi,)
        Evaluation of inequality constraints.
    ord_ : {1, 2, np.inf}
        Order of the merit function.

    Returns
    -------
    nc : float
        Norm of constraint violations.
    """

    if ord_ not in (1, 2, np.inf):
        raise ValueError('`ord` must be either 1, 2 or np.inf.')

    nce = 0.
    nci = 0.

    if ce.size:
        nce = la.norm(ce, ord=ord_)

    if ci.size:
        nci = la.norm(ci, ord=ord_)

    return nce + nci
