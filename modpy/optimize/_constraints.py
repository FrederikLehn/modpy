import numpy as np

from modpy.optimize._optim_util import _function, _jacobian_function, _hessian_function, _chk_bounds


class Constraint:
    def __init__(self, lb=(), ub=()):
        """
        Base class of Constraints that can be passed to constrained optimizers.

        Parameters
        ----------
        lb : array_like, shape (m,)
            Lower bound.
        ub : array_like, shape (m,)
            Upper bound.
        """

        self.linear = False
        self.bounds = False

        lb = np.array(lb, dtype=np.float64)
        ub = np.array(ub, dtype=np.float64)

        # if one of the bounds are not supplied, assume equality constraints
        if lb.size and not ub.size:
            ub = lb
        elif ub.size and not lb.size:
            lb = ub

        if lb.size != ub.size:
            raise ValueError('`lb` and `ub` must have the same size.')

        self.lb = lb
        self.ub = ub

        self._bounded = np.logical_or(self.lb != -np.inf, self.ub != np.inf)
        self._finite = np.logical_and(self.lb != -np.inf, self.ub != np.inf)
        self._equal = np.equal(self.lb, self.ub) & self._bounded
        self._inequal = np.logical_not(self._equal) & self._bounded

        # size of system is elongated to account for lb and ub handled as separate equations (for inequalities)
        self.n = self.lb.size
        self.me = np.count_nonzero(self._equal)
        self.mi = np.count_nonzero(self._inequal) + np.count_nonzero(self._inequal & self._finite)
        self.ml = 0  # elongated size

    def all_equal(self):
        return np.all(self._equal)

    def all_inequal(self):
        return np.all(self._inequal)

    def any_equal(self):
        return np.any(self._equal)

    def any_inequal(self):
        return np.any(self._inequal)

    def _extract_system(self, Af, At, b):
        lb, ub = self.lb, self.ub
        e = []  # elongation array

        it = 0
        for i in range(Af.shape[0]):

            if self._equal[i]:

                At[it, :] = Af[i, :]
                b[it] = lb[i]
                it += 1

                e.append(i)

            elif not self._finite[i]:

                if lb[i] != -np.inf:
                    At[it, :] = Af[i, :]
                    b[it] = lb[i]
                else:
                    At[it, :] = -Af[i, :]
                    b[it] = -ub[i]

                it += 1

                # adding elongation
                e.append(i)

            elif self._finite[i]:

                At[it, :] = Af[i, :]
                b[it] = lb[i]
                it += 1
                At[it] = -Af[i, :]
                b[it] = -ub[i]
                it += 1

                # adding elongation
                e.append(i)
                e.append(i)

        e = np.array(e, dtype=np.int64)
        self.ml += e.size

        return At, b, e

    def f_eq(self, x):
        # implement in sub-class
        pass

    def f_iq(self, x):
        # implement in sub-class
        pass

    def jac_eq(self, x, f):
        # implement in sub-class
        pass

    def jac_iq(self, x, f):
        # implement in sub-class
        pass

    def eval_eq(self, x):
        f = self.f_eq(x)
        J = self.jac_eq(x, f)
        return f, J

    def eval_iq(self, x):
        f = self.f_iq(x)
        J = self.jac_iq(x, f)
        return f, J


class LinearConstraint(Constraint):
    def __init__(self, A, lb=(), ub=()):
        """
        Constructs a linear constraint input to a constrained optimization problem

        Given a system matrix `A` the constraint has the form::

            lb <= A @ x <= ub

        If lb=ub the constraint is an equality constraint, otherwise it is an inequality constraint.

        Parameters
        ----------
        A : array_like, shape (m, n)
            System matrix of the constrained system.
        """

        super().__init__(lb, ub)

        self.linear = True

        self.A = np.atleast_2d(A)  # passed system matrix

        # splitting the system as lower and upper bounds have to be evaluated as separate equations
        self._A = None  # elongated equality constraint system matrix
        self._b = None  # elongated equality constraint results vector
        self._C = None  # elongated inequality constraint system matrix
        self._d = None  # elongated inequality constraint results vector

        self._prepare_system()

    def _prepare_system(self):
        _, n = self.A.shape

        # equality constrained system
        A = np.zeros((self.me, n))
        b = np.zeros((self.me,))
        self._A, self._b, _ = self._extract_system(self.A[self._equal], A, b)

        # inequality constrained system
        C = np.zeros((self.mi, n))
        d = np.zeros((self.mi,))
        self._C, self._d, _ = self._extract_system(self.A[self._inequal], C, d)

    def f_eq(self, x):
        return self._A @ x - self._b

    def f_iq(self, x):
        return self._C @ x - self._d

    def jac_eq(self, x, _):
        return self._A

    def jac_iq(self, x, _):
        return self._C

    def _hess(self, m, n):
        return tuple([np.zeros((n, n)) for _ in range(m)])

    def hess_eq(self, x):
        return self._hess(*self._A.shape)

    def hess_iq(self, x):
        return self._hess(*self._C.shape)

    def get_linear(self):
        A = self._A if self._A is not None else np.zeros((0, self.n))
        b = self._b if self._b is not None else np.zeros((0,))
        C = self._C if self._C is not None else np.zeros((0, self.n))
        d = self._d if self._d is not None else np.zeros((0,))

        return A, b, C, d


class Bounds(LinearConstraint):
    def __init__(self, lb, ub):
        super().__init__(np.eye(max(len(lb), len(ub))), lb, ub)

        self.linear = False
        self.bounds = True


class NonlinearConstraint(Constraint):
    def __init__(self, fun, lb=(), ub=(), jac='3-point', hess=None, args=(), kwargs={}):
        """
        Constructs a constraint input to a constrained optimization problem

        Given a function `fun` of them form::

            lb <= f(x) <= ub

        Parameters
        ----------
        fun : callable
            Function which computes a vector of shape (m,) with call f(x, *args, **kwargs).
        jac : {'2-point', '3-point', callable}, optional
            Method for calculating the Jacobian.
            If callable, the function has to return a vector of size (m, n).
            The options '2-point' and '3-point' refer to finite difference schemes.
        hess : callable
            Hessian of the constraint functions. `hess` should be a function of the form:
                hess(x, args, kwargs)
            and return a tuple of (array_like(n, n), ...) which is m long.
            `hess` is required if a true Hessian function is also passed to the
            gradient-based optimization function. Alternatively an approximate
            Hessian scheme such as BFGS and SR1 is used, in which case the Hessian
            of the constraint is not required.
        args : tuple
            Additional arguments to `fun`.
        kwargs : dict
            Additional key-word arguments to `fun`.
        """

        super().__init__(lb, ub)

        self.fun = _function(fun, args, kwargs)
        self.jac = _jacobian_function(fun, jac, args=args, kwargs=kwargs)
        self.hess = _hessian_function(hess, constraint=True, args=args, kwargs=kwargs)

        # splitting the system as lower and upper bounds have to be evaluated as separate equations
        self._b = None      # elongated equality constraint results vector
        self._d = None      # elongated inequality constraint results vector
        self._ee = None     # equality constraints elongation vector
        self._ei = None     # inequality constraints elongation vector

        self._prepare_system()

    def _prepare_system(self):
        z = np.zeros((self.n, 0))

        # equality constrained system
        b = np.zeros((self.me,))
        _, self._b, self._ee = self._extract_system(z[self._equal], np.zeros((self.me, 0)), b)

        # inequality constrained system
        d = np.zeros((self.mi,))
        _, self._d, self._ei = self._extract_system(z[self._inequal], np.zeros((self.mi, 0)), d)

    def f_eq(self, x):
        return self.fun(x)[self._ee] - self._b

    def f_iq(self, x):
        return self.fun(x)[self._ei] - self._d

    def _jac(self, x, f, all_test, elon):
        if all_test:
            return np.empty((0, x.size))

        J = self.jac(x, f)

        if J.ndim == 1:
            J = np.atleast_2d(J)
        else:
            J = J[elon]

        return J

    def jac_eq(self, x, f):
        return self._jac(x, f, self.all_inequal(), self._ee)

    def jac_iq(self, x, f):
        return self._jac(x, f, self.all_equal(), self._ei)

    def _hess(self, x, all_test, elon):
        if all_test:
            return ()

        H_ = self.hess(x)
        H = tuple([H_[e] for e in elon])

        return H

    def hess_eq(self, x):
        return self._hess(x, self.all_inequal(), self._ee)

    def hess_iq(self, x):
        return self._hess(x, self.all_equal(), self._ei)


class Constraints:
    def __init__(self, constraints):

        self.con = constraints

    def all_equal(self):
        return np.all(np.array([c.all_equal() for c in self.con]))

    def any_equal(self):
        return np.any(np.array([c.any_equal() for c in self.con]))

    def count(self):
        return np.sum([c.ml for c in self.con])#len(self.con)

    @staticmethod
    def _merge_funs(funs):
        return np.block(list(funs)).flatten()

    @staticmethod
    def _merge_jacs(x, jacs):
        if jacs:
            J = np.vstack(jacs)
        else:
            return np.empty((0, x.size))

        #if J.shape[0] == 1:
        #    J = np.ravel(J)

        return J

    def _merge_hess(self, hess):
        H = tuple([c for h in hess for c in h])
        return H

    def f(self, x):
        return self._merge_funs((self.f_eq(x), self.f_iq(x)))

    def f_eq(self, x):
        return self._merge_funs([c.f_eq(x) for c in self.con])

    def f_iq(self, x):
        return self._merge_funs([c.f_iq(x) for c in self.con])

    def jac_eq(self, x, f):
        con = [c.jac_eq(x, f) for c in self.con]
        con = [c for c in con if c.size]

        return self._merge_jacs(x, con)

    def jac_iq(self, x, f):
        con = [c.jac_iq(x, f) for c in self.con]
        con = [c for c in con if c.size]

        return self._merge_jacs(x, con)

    def hess_eq(self, x):
        con = [c.hess_eq(x) for c in self.con]
        con = tuple([c for c in con if c])

        return self._merge_hess(con)

    def hess_iq(self, x):
        con = [c.hess_iq(x) for c in self.con]
        con = tuple([c for c in con if c])

        return self._merge_hess(con)

    def eval_eq(self, x):
        con = [c.eval_eq(x) for c in self.con]
        f, J = zip(*con)
        return self._merge_funs(f), self._merge_jacs(x, J).T

    def eval_iq(self, x):
        con = [c.eval_iq(x) for c in self.con]
        f, J = zip(*con)
        return self._merge_funs(f), self._merge_jacs(x, J).T

    def get_bounds(self, n):
        bounds = [zip(c.lb, c.ub) for c in self.con if c.bounds]

        if bounds:
            bounds = list(*bounds)
        else:
            bounds = list([(-np.inf, np.inf) for _ in range(n)])

        return bounds

    def get_linear(self, n):
        A = np.zeros((0, n))
        b = np.zeros((0,))
        C = np.zeros((0, n))
        d = np.zeros((0,))

        cl = [c.get_linear() for c in self.con if c.linear]
        if cl:
            As, bs, Cs, ds = zip(*cl)
            A = np.vstack((A, *As))
            b = np.block([b, *bs])
            C = np.vstack((C, *Cs))
            d = np.block([d, *ds])

        return A, b, C, d


# ======================================================================================================================
# Auxiliary Methods
# ======================================================================================================================
def prepare_bounds(bounds, n):

    if bounds is None:
        return np.full((n,), -np.inf), np.full((n,), np.inf)

    lb, ub = zip(*bounds)

    # change None values to np.inf
    lb = np.array([b if b is not None else -np.inf for b in lb], dtype=np.float64)
    ub = np.array([b if b is not None else np.inf for b in ub], dtype=np.float64)

    if lb.ndim == 0:
        lb = np.resize(lb, n)

    if ub.ndim == 0:
        ub = np.resize(ub, n)

    _chk_bounds(lb, ub, n)
    return lb, ub


def _bounds_to_equations(lb, ub, C, d):
    """
    Takes a set of lower and upper bounds and turns them into inequality constraints
    by appending them to matrix `C` and vector `d`.

    Parameters
    ----------
    lb : array_like, shape (n,)
        Lower bounds on solution variable.
    ub : array_like, shape (n,)
        Lower bounds on solution variable.
    C : array_like, shape (n, mi)
        System matrix with coefficients for the inequality constraints.
    d : array_like, shape (mi,)
        Results vector of the inequality constraints.

    Returns
    -------
    C : array_like, shape (n, mi + m)
        System matrix with coefficients for the inequality constraints, including bounds.
    d : array_like, shape (mi + m,)
        Results vector of the inequality constraints, including bounds.
    """

    m, n = C.shape

    C_ = np.vstack((np.eye(n), -np.eye(n)))
    d_ = np.vstack((lb, -ub))

    # remove unbounded variables from the equations
    il = np.argwhere((lb == np.inf) | (lb == -np.inf)).flatten()
    iu = n + np.argwhere((ub == np.inf) | (ub == -np.inf)).flatten()
    i = np.block([il, iu])
    C_ = np.delete(C_, i, axis=0)
    d_ = np.delete(d_, i)

    # append bound constraints to inequality constraints
    C = np.vstack((C, C_))
    d = np.block([d, d_])

    return C, d


def _prepare_constraints(bounds, constraints, n):
    if isinstance(constraints, tuple):
        con = list(constraints)
    else:
        con = [constraints]

    if bounds is not None:
        con.append(Bounds(*prepare_bounds(bounds, n)))

    return Constraints(con)
