import warnings
import numpy as np
import numpy.linalg as la

from numpy.random import Generator, PCG64

from modpy.optimize._optim_util import OptimizeResult, OptimizePath, _function, _chk_callable
from modpy.optimize._constraints import _prepare_constraints


ID_INFEASIBLE = -12
ID_MAX_SIGMA = -11
ID_TOL_X = -10
ID_TOL_FUN = -9
ID_TOL_X_UP = -8
ID_COND_COV = -7
ID_STAGNATION = -6
ID_EQUAL_FUN = -5
ID_NO_EFFECT_COORD = -4
ID_NO_EFFECT_AXIS = -3
ID_MAX_RESTART = -2
ID_LIN_ALG = -1
ID_MAX_ITER = 0
ID_CONV = 1

TERMINATION_MESSAGES = {
    ID_INFEASIBLE:      'No feasible candidates',
    ID_MAX_SIGMA:       'MaxSigma criteria.',
    ID_TOL_X:           'TolX criteria.',
    ID_TOL_FUN:         'TolFun criteria.',
    ID_TOL_X_UP:        'TolXUp criteria',
    ID_COND_COV:        'ConditionCov criteria.',
    ID_STAGNATION:      'Stagnation criteria.',
    ID_EQUAL_FUN:       'EqualFunValues criteria.',
    ID_NO_EFFECT_COORD: 'NoEffectCoord criteria.',
    ID_NO_EFFECT_AXIS:  'NoEffectAxis criteria.',
    ID_MAX_RESTART:     'Maximum number of restart reached.',
    ID_LIN_ALG:         'LinAlgError due to indeterminate system.',
    ID_MAX_ITER:        'Maximum number of iterations reached.',
    ID_CONV:            'Tolerance termination condition is satisfied.',
}


MAX_SIGMA = 1e32


def cma_es(obj, x0, bounds=None, constraints=(), method='IPOP', sigma0=1., C0=None, mu=None, lam=None, lbound=np.inf,
           tol=1e-6, ftol=1e-12, xtol=1e-12, stol=1e-12, maxiter=None, max_restart=5, seed=None, keep_path=False,
           args=(), kwargs={}):
    """
    Wrapper method for Covariance Matrix Adaptation Evolution Strategy algorithm (CMA-ES).

    Parameters
    ----------
    obj : callable
        Objective function
    x0 : array_like or int
        If array_like `x0` is used as a start guess, if int it is the problem dimension.
    bounds : 2-tuple of array_like, optional
        Bounds on the solution vector, should be (array_like (n,), array_like (n,)). If None no bounds are applied.
    constraints : tuple or Constraint
        Tuple of class Constraints.
    method : {'1+1, 'mu-lam', 'IPOP'}, optional
        Which CMA-ES implementation to use.
    sigma0 : float, optional
        Coordinate wise standard deviation (step size).
    C0 : array_like, shape (n, n), optional
        Initial correlation matrix.
    mu : int, optional
        Number of selected candidates from sampled population.
    lam : int, optional
        Number of sampled candidates in the population.
    lbound : float, optional
        Analytical or empirical bound for the function value to at least obtain in order to have converged.
    tol : float, optional
        Tolerance related to change in fitness value over sequence of populations.
    ftol : float, optional
        Tolerance of the TolFun termination criteria.
    xtol : float, optional
        Tolerance of the TolX termination criteria.
    stol : float, optional
        Tolerance of the NoEffectAxis and NoEffectCoord termination criteria.
    maxiter : int, optional
        Maximum number of allowed iterations.
    max_restart : int, optional
        Maximum number of allowed restarts.
    seed : int, optional
        Seed of the random number generator used in generation step.
    keep_path : bool, optional
        Whether to save path information. Can require substantial memory.
    args : tuple, optional
        Additional arguments to `fun`.
    kwargs : dict, optional
        Additional key-word arguments to `fun`.

    Returns
    -------
    OptimizeResult with the following fields:
    x : array_like, shape (n,)
        Solution vector.
    success : bool,
        True if algorithm converged within its optimality conditions.
    status : int
        Reason for algorithm termination
    message : str
        Description of the termination reason.
    nit : int
        Number of iterations used to converge.
    """

    # prepare start guess and initial objective function value
    if isinstance(x0, int):
        x0 = np.random.randn(x0)

    n = x0.size

    # wrap function call with args and kwargs
    f = _function(obj, args, kwargs)
    _chk_callable(x0, f)

    if sigma0 < 0.:
        raise ValueError('CMA-ES requires sigma0 > 0.')

    # prepare constraints
    if (bounds is not None) or constraints:

        con = _prepare_constraints(bounds, constraints, n)

        # if con.any_equal():
        #     raise ValueError('CMA-ES algorithms does not allow equality constraints.')

    else:

        con = None

    # try:
    #
    #     if method == 'mu-lam':
    #
    #         sol, f_opt, status, nit, path = _cma_es_mu_lam(f, x0, generator, constraints=con, sigma0=sigma0, C0=C0,
    #                                                        mu=mu, lam=lam, lbound=lbound, tol=tol, ftol=ftol, xtol=xtol,
    #                                                        stol=stol, maxiter=maxiter, keep_path=keep_path)
    #
    #     elif method == 'IPOP':
    #
    #         sol, f_opt, status, nit, path = _cma_es_ipop(f, x0, generator, constraints=con, sigma0=sigma0, C0=C0, mu=mu,
    #                                                      lam=lam, lbound=lbound, tol=tol, ftol=ftol, xtol=xtol, stol=stol,
    #                                                      maxiter=maxiter, max_restart=max_restart, keep_path=keep_path)
    #
    #     elif method == '1+1':
    #
    #         if con is None:
    #
    #             sol, f_opt, status, nit, path = _cma_es_1p1(f, x0, sigma0=sigma0, tol=tol, stol=stol,
    #                                                         maxiter=maxiter, keep_path=keep_path)
    #
    #         else:
    #
    #             sol, f_opt, status, nit, path = _cma_es_1p1_con(f, x0, con, sigma0=sigma0, tol=tol, stol=stol,
    #                                                             maxiter=maxiter, keep_path=keep_path)
    #
    #     else:
    #
    #         raise ValueError("`method` must be either '1+1' or 'mu-lam'.")
    #
    #     f_opt = float(f_opt)
    #
    # except la.LinAlgError:
    #
    #     f_opt = None
    #     sol = None
    #     status = -1
    #     nit = 0
    #     path = None
    #
    # if sol is None:
    #     x = None
    # else:
    #     x = sol
    #
    # res = OptimizeResult(x, f_opt, sol, status=status, nit=nit, tol=tol)
    # res.success = status > 0
    # res.message = TERMINATION_MESSAGES[res.status]
    # res.path = path

    opt = CMAES(obj, x0, method, con, sigma0, C0, mu, lam, lbound, tol, ftol, xtol, stol, maxiter, max_restart, seed, keep_path)
    opt.run()
    res = opt.get_result()

    return res


class CMAES:
    """
    Covariance Matrix Adaptation Evolution Strategy algorithm (CMA-ES).
    """

    def __init__(self, obj, x0, method='IPOP', constraints=None, sigma0=1., C0=None, mu=None, lam=None, lbound=np.inf,
                 tol=1e-6, ftol=1e-12, xtol=1e-12, stol=1e-10, maxiter=None, max_restart=5, seed=None, keep_path=False):

        """
        Initializer of the CMA-ES algorithm class.

        Parameters
        ----------
        obj : callable
            Objective function
        x0 : array_like, shape (n,)
            Start guess
        method : {'mu-lam', 'IPOP'}, optional
            Which CMA-ES implementation to use.
        constraints : Constraints, optional
            Class Constraints.
        sigma0 : float, optional
            Coordinate wise standard deviation (step size).
        C0 : array_like, shape (n, n), optional
            Initial correlation matrix.
        mu : int, optional
            Number of selected candidates from sampled population.
        lam : int, optional
            Number of sampled candidates in the population.
        lbound : float, optional
            Analytical or empirical bound for the function value to at least obtain in order to have converged.
        tol : float, optional
            Tolerance related to change in fitness value over sequence of populations.
        ftol : float, optional
            Tolerance of the TolFun termination criteria.
        xtol : float, optional
            Tolerance of the TolX termination criteria.
        stol : float, optional
            Tolerance of the NoEffectAxis and NoEffectCoord termination criteria.
        maxiter : int, optional
            Maximum number of allowed iterations
        max_restart : int, optional
            Maximum number of allowed restarts, for IPOP only.
        seed : int, optional
            Seed of the random number generator used in generation step.
        keep_path : bool, optional
            Whether to save path information. Can require substantial memory.
        """

        # problem definition
        self.dim = None
        self.obj = obj
        self.method = method
        self.constraints = constraints

        # final solution variables
        self.x_opt = None
        self.f_opt = None

        # current solution variables
        self.m = np.array(x0)
        self.sigma = sigma0
        self.C = C0

        # provided parameters ------------------------------------------------------------------------------------------
        self.lam = lam          # population sample size
        self.mu_ori = mu        # original recombination sample size
        self.mu = None          # recombination sample size at iteration k (may change due to infeasible samples)
        self._reduced = False   # if mu is updated during iteration

        # derived parameters -------------------------------------------------------------------------------------------
        # static
        self.chiN = None    # E[||N(0, I)||]
        self.gamma = None   # constraint violation parameter

        # dynamic
        self.mu_eff = None  # variance - effectiveness of sum w_i x_i
        self.w = None       # recombination weights array
        self.cc = None      # time constant for cumulation of C
        self.cs = None      # time constant for cumulation of sigma
        self.c1 = None      # learning rate for rank-1 update of C
        self.cmu = None     # learning rate for rank-mu update of C
        self.ds = None      # damping for sigma

        # storage variables --------------------------------------------------------------------------------------------
        # static
        self.yw = None          # weighted recombination of random variables (stored for int. calculations)
        self.pc = None          # evolution path for C
        self.ps = None          # evolution paths for sigma
        self.B = None           # orthonormal eigendecomposition matrix of C
        self.D = None           # vector of eigenvalues of C
        self.BD = None          # matrix-product of B and D
        self.C = None           # covariance matrix C
        self.invsqrtC = None    # square-root of inverse covariance matrix
        self.D_min = None       # minimum eigenvalue
        self.D_max = None       # maximum eigenvalue
        self.max_sigma = None   # maximum sigma of current iteration

        # dynamic
        self.y = None           # array of sampled random numbers
        self.x = None           # array of candidate solutions
        self.f = None           # vector of objective function values

        self.x_all = None       # store all feasible candidates (for later save)

        # convergence options ------------------------------------------------------------------------------------------
        # termination criteria
        self.tol = tol
        self.ftol = ftol
        self.xtol = xtol
        self.stol = stol
        self.lbound = lbound
        self.tolx = None

        # maximum iterations
        self.maxiter = maxiter              # maximum iterations for a given (mu/mu_w-lam)-CMA-ES loop
        self.max_redraw = 20                # maximum number of redraws allowed if sample is infeasible
        self.max_restart = max_restart      # maximum number of restarts for IPOP-CMA-ES

        # counters
        self.it = 0
        self.it_total = 0
        self.it_restart = 0
        self.eig_it = 0

        # convergence status
        self.status = 1
        self.converged = False
        self.success = False

        # random generation --------------------------------------------------------------------------------------------
        self.seed = seed
        self.generator = None

        # optimization path --------------------------------------------------------------------------------------------
        self.keep_path = keep_path
        self.path = OptimizePath(keep=keep_path)

    def get_result(self):
        res = OptimizeResult(self.x_opt, self.f_opt, self.x_opt, status=self.status, nit=self.it_total, tol=self.tol)
        res.success = self.status > 0
        res.message = TERMINATION_MESSAGES[self.status]
        res.path = self.path

        return res

    def run(self):

        if self.method == 'mu-lam':

            self.initialize()
            self._run_mu_lam()
            self.it_total = self.it

        elif self.method == 'IPOP':

            self._run_ipop()

        else:

            raise ValueError("`method` must be either 'mu-lam' or 'IPOP'.")

        # if the algorithm terminated with an infeasible solution, it has diverged
        if self.constraints is not None:
            if np.any(self.constraints.f(self.m) < 0.):
                self.status = ID_INFEASIBLE
                self.converged = False

        # TODO: several termination criterias will be considered as converged, reasonable?
        if self.status in (ID_NO_EFFECT_AXIS, ID_NO_EFFECT_COORD):
            self.status = 1
            self.converged = 1

    def _run_ipop(self):
        """
        An IPOP-(mu/muw lambda)-CMA-ES.

        References
        ----------
        [1] Hansen, N., Auger, A. (2005). A Restart CMA Evolution Strategy With Increasing Population Size
            Link: http://www.cmap.polytechnique.fr/~nikolaus.hansen/cec2005ipopcmaes.pdf
        """

        while self.it_restart < self.max_restart:

            # call sub (mu/lam)-CMA-ES
            self.initialize()
            self._run_mu_lam()

            self.it_total += self.it
            self.it = 0

            # check if the inner optimizer converged
            if self.converged:
                break

            else:
                # otherwise update population size and step-size
                self.lam *= 2

                if self.status == ID_TOL_X_UP:
                    self.sigma *= 10.
                else:
                    self.sigma *= 2.

            self.it_restart += 1

        if self.it_restart == self.max_restart:
            self.status = ID_MAX_RESTART

    def _run_mu_lam(self):
        """
        A classical (mu/mu_w lambda)-CMA-ES. Primarily based on the work of Nicolaus Hansen [1], but further
        improvements made for convergence checking. Constraint handling is implemented according to [2].

        References
        ----------
        [1] Hansen, N. (2011). The CMA Evolution Strategy: A Tutorial
            Link: http://www.cmap.polytechnique.fr/~nikolaus.hansen/cmatutorial110628.pdf
        [2] Chocat, R., Brevault, L., Defoort, S. (2015). Modified Covariance Matrix Adaptation - Evolution Strategy
            for constrained optimization under uncertainty, application to rocket design.
            International Journal for Simulation and Multidisciplinary Design Optimization.
        """

        while self.it < self.maxiter:

            # draw candidates and sort based on fitness
            self._draw_candidates()
            self._sort_candidates()

            # handle constraint violations
            self._handle_constraints()
            self._adjust_constrained_candidates()

            if self.y.size:
                # update solution estimate and evolution path
                self._recombine_solution()
                self._update_evolution_paths()
                self._decompose_covariance()

                # check if algorithm has converged
                self.converged = self._check_convergence()
                if self.converged:
                    break

                # check if algorithm should be terminated/restarted
                self.status = self._check_termination_criteria()

                if self.status < 1:
                    break

                # escape flat fitness
                self._escape_flat_fitness()

            elif self.status < 1:
                break

            # save optimization path for later plotting
            if self.keep_path:
                self.path.append(self.m, self.f[0], self.f[0], sigma=self.sigma, candidates=self.x_all)

            # revert recombination parameters to default values
            if self._reduced:
                self._define_parameters(self.mu_ori)
                self._reduced = False

            self.it += 1

        # assign optimal values at the point of termination
        self.x_opt = self.m
        self.f_opt = self.obj(self.m)

        if self.it == self.maxiter:
            self.status = 0

        self.success = self.status > 0

    def initialize(self):
        self.dim = self.m.size

        # strategy parameter setting: selection
        if self.maxiter is None:
            self.maxiter = 1e3 * self.dim ** 2

        if self.lam is None:
            self.lam = int(4 + np.floor(3 * np.log(self.dim)))
        else:
            self.lam = int(self.lam)

        if self.lam < 2:
            raise ValueError('CMA-ES requires lam >= 2.')

        if self.mu_ori is None:
            self.mu_ori = self.lam / 2

        if self.mu_ori > self.lam:
            raise ValueError('CMA-ES requires mu <= lam.')

        self.chiN = self.dim ** 0.5 * (1. - 1. / (4. * self.dim) + 1. / (21. * self.dim ** 2.))
        self.gamma = 0.1 / (self.dim + 2.)

        self.max_sigma = float(self.sigma)

        self._define_parameters(self.mu_ori)
        self._initialize_static_arrays()

        # random sampling
        if isinstance(self.seed, Generator):
            self.generator = self.seed
        else:
            self.generator = Generator(PCG64(self.seed))

        self._initialize_tolerances()

    def _define_parameters(self, mu):
        n = self.dim

        # set recombination weights
        w = np.log(mu + .5) - np.log(range(1, int(mu + 1)))
        self.w = w / np.sum(w)

        # set number of points to include in recombination
        mu_eff = 1. / np.sum(w ** 2.)
        self.mu = int(np.floor(mu))

        # strategy parameter setting: adaptation
        self.cc = (4. + mu_eff / n) / (n + 4. + 2. * mu_eff / n)
        self.cs = (mu_eff + 2.) / (n + mu_eff + 5.)
        self.c1 = 2. / ((n + 1.3) ** 2. + mu_eff)
        self.cmu = np.minimum(1. - self.c1, 2. * (mu_eff - 2. + 1. / mu_eff) / ((n + 2.) ** 2. + mu_eff))
        self.ds = 1. + 2. * np.maximum(0., 0. if mu_eff < 1. else np.sqrt((mu_eff - 1.) / (n + 1.)) - 1.) + self.cs

        self.mu_eff = mu_eff

    def _initialize_static_arrays(self):
        n = self.m.size

        self.ps = np.zeros((n,))
        self.pc = np.zeros((n,))

        if self.C is None:
            self.C = np.eye(n)
            self.invsqrtC = np.eye(n)
            self.B = np.eye(n)
            self.D = np.ones((n,))
            self.BD = np.eye(n)
            self.D_min = 1.
            self.D_max = 1.
        else:
            self._decompose_covariance()

    def _initialize_dynamic_arrays(self):
        n = self.dim

        self.x = np.zeros((n, self.lam))
        self.y = np.zeros((n, self.lam))
        self.f = np.zeros((self.lam,))

    def _initialize_tolerances(self):
        self.tolx = self.sigma * self.xtol

    def _decompose_covariance(self):
        # enforce symmetry of C
        self.C = np.triu(self.C) + np.triu(self.C, 1).T

        # perform eigen-decomposition
        try:
            D, self.B = la.eig(self.C)

        except la.LinAlgError:
            self.status = ID_LIN_ALG
            return

        self.D = np.sqrt(D)
        self.BD = self.B @ np.diag(self.D)

        # calculate inverse square-root of C
        self.invsqrtC = self.B * np.diag(1. / self.D) * self.B.T

        # calculate minimum and maximum eigenvalue
        self.D_min = np.abs(np.amin(self.D))
        self.D_max = np.abs(np.amax(self.D))

    def _recombine_solution(self):
        self.yw = self.y @ self.w
        self.m += self.sigma * self.yw

    def _update_evolution_paths(self):
        n = self.dim

        # cumulation: update evolution paths
        self.ps = (1. - self.cs) * self.ps + np.sqrt(self.cs * (2. - self.cs) * self.mu_eff) * self.invsqrtC @ self.yw

        hsig = (la.norm(self.ps) ** 2. / np.sqrt(1. - (1. - self.cs) ** (2. * (self.it + 1))))\
               < ((1.4 + 2. / (n + 1.)) * self.chiN)

        self.pc = (1. - self.cc) * self.pc + hsig * np.sqrt(self.cc * (2. - self.cc) * self.mu_eff) * self.yw

        # adapt covariance matrix C
        dhsig = (1 - hsig) * self.cc * (2. - self.cc)

        self.C = (1. - self.c1 - self.cmu) * self.C + self.c1 * (np.outer(self.pc, self.pc) + dhsig * self.C) +\
                 self.cmu * self.y @ np.diag(self.w) @ self.y.T

        # adapt step size sigma
        self.sigma *= np.exp((self.cs / self.ds) * (la.norm(self.ps) / self.chiN - 1.))

    def _draw_candidates(self):
        self._initialize_dynamic_arrays()

        for k in range(self.lam):

            feasible = False
            redraw = 0
            f_k = 0  # never used

            while (not feasible) and (redraw < self.max_redraw):
                self.y[:, k] = self.BD @ self.generator.standard_normal(self.dim)
                self.x[:, k] = self.m + self.sigma * self.y[:, k]

                # calculate objective function value
                f_k = self.obj(self.x[:, k])

                # if sampled value is mathematically infeasible, redraw a new one.
                if not (np.isnan(f_k) or np.isinf(f_k)):
                    feasible = True

                redraw += 1

            if redraw == self.max_redraw:
                warnings.warn('Maximum redraws reached, continuing with infeasible candidate.')

            self.f[k] = f_k

        # remove any remaining inf or nan that were not screened out during the redrawing
        self._adjust_finite_candidates()

    def _adjust_finite_candidates(self):
        m = np.isfinite(self.f)
        self.y = self.y[:, m]
        self.x = self.x[:, m]
        self.f = self.f[m]

        # parameters that are based on mu are changed to reflect the reduced sample size
        if self.f.size < self.mu:

            self.mu = self.f.size

            # if there are no feasible candidates the problem cannot progress
            if self.mu == 0:
                self.status = ID_INFEASIBLE
                self.converged = True

    def _sort_candidates(self):
        # sort based on fitness value (lowest to highest)
        idx = np.argsort(self.f)

        # save all candidates for later
        self.x_all = np.array(self.x)

        # re-order lists
        self.y = self.y[:, idx[:self.mu]]
        self.x = self.x[:, idx[:self.mu]]
        self.f = self.f[idx]

    def _handle_constraints(self):

        if self.constraints is not None:

            nc = self.constraints.count()
            v = np.zeros((self.mu, nc))

            # calculate constraint violations
            for i in range(self.mu):
                v[i, :] = self.constraints.f(self.x[:, i])

            # check if constraints are violated
            feasible = np.sum(v < 0., axis=1) == 0
            if np.all(feasible):
                return

            # if all samples are infeasible then the algorithm will continue as though there were
            # no constraints. This is in different than the suggestion in [2], but it appears that
            # the constraint handling is ineffective if all guesses are outside the feasible region.
            # Instead the algorithm is allowed to continue searching for an optimum, assuming
            # that a feasible region will be reached along the way, in which case the normal
            # constraint handling continues again.
            if np.all(~feasible):
                return

            D_adj, violated = self._adjust_eigenvalues(v)

            if violated:
                S = self.C - self.gamma * self.B @ np.diag(D_adj) @ self.B.T

                detS = la.det(S)
                detC = la.det(self.C)

                # no additional update of the covariance matrix due to instability
                # TODO: use (np.abs(detS) < det_tol) or (np.abs(detC) < det_tol) instead?
                if (detS < 0.) or (detC < 0.):
                    return

                self.C = (detC / detS) ** (1. / self.dim) * S

                # reducing y to the number of feasible candidates
                self.y = self.y[:, feasible]

    def _adjust_constrained_candidates(self):
        # if the number of best candidates were reduced due to not being finite
        # or being infeasible, the parameters have to be updated to reflect
        # the lower number of 'mu'.
        _, mu = self.y.shape

        if mu != self.mu:

            # no feasible candidates, start re-draw
            if mu == 0:
                self.it += 1
                return False

            self._define_parameters(mu)
            self._reduced = True

            warnings.warn('Number of feasible samples is less than `mu`. Recalculating parameters.')

    def _adjust_eigenvalues(self, v):
        _, nc = v.shape

        mask = v < 0.
        mu_cj = np.sum(mask, axis=1)

        # if no constraints are violated, return the original eigenvalues
        feasible = mu_cj == 0
        if np.all(feasible):
            return self.D, False  # None removed from middle

        x = self.x[:, ~feasible]
        v = v[~feasible, :]
        mask = mask[~feasible, :]

        D_adj = self.D.copy()

        # loop over problem dimension
        for i in range(self.dim):

            sumc = 0.

            # loop over constraints
            for j in range(nc):

                mu = np.sum(mask[:, j])

                # the j'th constraint is not violated
                if mu == 0:
                    continue

                # calculate constraint weights
                w = np.log(mu + .5) - np.log(range(1, int(mu + 1)))
                w /= np.sum(w)

                # sort constraint violations
                v_temp = v[v[:, j] < 0, j]
                il = np.argsort(v_temp)
                il = np.argwhere(mask[:, j])[il].flatten()

                # loop over candidates violating constraint 'cj'
                for k, idx in enumerate(il):
                    sumc += w[k] * _eigen_projection(x[:, idx] - self.m, self.B[:, i])
                    # no division as the sum of w = 1.

            D_adj[i] = self.D[i] * sumc

        return D_adj, True  # feasible removed from middle

    def _check_convergence(self):
        """
        Test the objective function value against a lower bound  `lbound` as well as in a hypercube around
        the current best solution vector `x` to determine if `x` is a minimum.

        Returns
        -------
        converged : bool
            True/False whether `x` is a minimum below the lower bound.
        """

        #eps = np.sqrt(np.finfo(np.float64).eps)
        eps = self.tol

        f_best = self.obj(self.m)

        if f_best > self.lbound:
            return False

        converged = True
        m_ = np.copy(self.m)

        for i in range(self.dim):

            # test negative side
            m_[i] -= eps
            f = self.obj(m_)

            if f < f_best:
                if self.constraints is not None:
                    c = self.constraints.f(m_)

                    if np.all(c > 0.):
                        return False
                else:
                    return False

            # test positive side
            m_[i] += 2. * eps
            f = self.obj(m_)

            if f < f_best:
                if self.constraints is not None:
                    c = self.constraints.f(m_)

                    if np.all(c > 0.):
                        return False
                else:
                    return False

            # return to original value at i'th coordinate
            m_ -= eps

        self.status = 1

        return converged

    def _check_termination_criteria(self):

        # terminate due to NoEffectAxis of [1]
        if np.all(np.abs(0.1 * self.sigma * np.diag(self.BD)) < self.stol):
            return ID_NO_EFFECT_AXIS

        # terminate due to NoEffectCoord of [1]
        if np.all(np.abs(0.2 * self.sigma * np.diag(self.C)) < self.stol):
            return ID_NO_EFFECT_COORD

        # terminate due to ConditionCov of [1]
        if (self.D_max / self.D_min) > 1e14:
            return ID_COND_COV

        # terminate due to TolXUp of [1]
        max_sigma_k = self.sigma * self.D_max
        if (max_sigma_k / self.max_sigma) > 1e4:
            return ID_TOL_X_UP
        else:
            self.max_sigma = float(max_sigma_k)

        # terminate due to TolX criteria of [1]
        if np.all(np.abs(self.sigma * self.pc) < self.tolx) and np.all(np.abs(self.C) < self.tolx):
            return ID_TOL_X

        # terminate due to MaxSigma
        if self.sigma > MAX_SIGMA:
            return ID_MAX_SIGMA

        return self.status

    def _escape_flat_fitness(self):
        if np.abs(self.f[0] - self.f[int(np.ceil(0.7 * self.f.size))]) < 1e-12:
            self.sigma *= np.exp(0.2 + self.cs / self.ds)
            warnings.warn('Flat fitness, increasing step-size at iteration {}.'.format(self.it))


def _eigen_projection(x, b):
    """
    Project the vector `x` onto the eigenvector `b`. Here `b` is a column of the matrix B which is an orthonormal
    basis from the eigendecomposition::

        BDB = C

    Parameters
    ----------
    x : array_like, shape (n,)
        Vector.
    b : array_like, shape (n,)
        Normalized eigenvector

    Returns
    -------
    y : bool
        Projection of `x` onto `b`.
    """

    return np.dot(b, x)# * b
