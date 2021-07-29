import numpy as np

from modpy.optimize._derivatives import approx_difference, _damped_BFGS_update


class OptimizeResult:
    def __init__(self, x=None, f=None, sol=None, success=False, status=0, message='', nit=0, cond=None, tol=None):

        self.x = x  # optimal result (primary variables only)
        self.f = f  # optimal function value

        # relevant for specific algorithms, e.g. optimal result might include primary, dual and slack variables
        if sol is None:
            sol = x

        self.sol = sol
        self.cond = cond

        # performance
        self.success = success  # true/false whether algorithm converged
        self.status = status    # termination cause (see individual solver for description)
        self.message = message  # termination message
        self.nit = nit          # number of iterations performed
        self.tol = tol          # user provided tolerance

        # path taken by algorithm
        self.path = None


class OptimizeMMOResult:
    def __init__(self, results, track, success=False, status=0, message='', nit=0):

        self.results = results  # tuple of OptimizeResult
        self.track = track      # list of evaluation counts which an optimum was found

        self.success = success  # true/false whether at least 1 result was found
        self.status = status    # termination cause (see individual solver for description)
        self.message = message  # termination message
        self.nit = nit          # number of iterations performed


class OptimizePath:
    def __init__(self, keep=False):

        self.keep = keep  # whether to keep results when functions are called

        # main results
        self.xi = []        # list successive choices of x
        self.fi = []        # list successive objective function values
        self.ftol = []      # list of all successive f-termination criteria calculations
        self.rtol = []      # list of all successive relative tolerance criteria calculations
        self.etol = []      # list of all successive eq. constraint criteria calculations
        self.itol = []      # list of all successive ineq. constraint criteria calculations
        self.mu = []        # list of all successive barrier parameters for interior-point algorithms
        self.sigma = []     # list of all successive step-sizes for CMA algorithms

        # for evolution strategies
        self.candidates = None  # list of all candidates in each iteration

    def append(self, xi, fi, ftol, rtol=None, etol=None, itol=None, mu=None, sigma=None, candidates=None):
        if not self.keep:
            return

        if isinstance(xi, float):
            self.xi.append(xi)
        else:
            self.xi.append(np.copy(xi))

        self.fi.append(float(fi))
        self.ftol.append(float(ftol))

        if rtol is not None:
            self.rtol.append(float(rtol))

        if etol is not None:
            self.etol.append(float(etol))

        if itol is not None:
            self.itol.append(float(itol))

        if mu is not None:
            self.mu.append(float(mu))

        if sigma is not None:
            self.sigma.append(float(sigma))

        if candidates is not None:
            if self.candidates is None:
                self.candidates = np.empty((0, xi.size))

            self.candidates = np.append(self.candidates, np.copy(candidates).T, axis=0)


def _function(fun, args=(), kwargs={}):
    if fun is None:
        def fun_wrapped(_):
            return np.empty((0,))

    else:
        def fun_wrapped(x):
            f = np.atleast_1d(fun(x, *args, **kwargs))

            if f.ndim > 1:
                raise RuntimeError('`fun` return value has more than 1 dimension.')

            return f

    return fun_wrapped


def _jacobian_function(fun, jac, x0=None, args=(), kwargs={}):
    if fun is None:
        if x0 is None:
            raise ValueError('`x0` must not be None when `fun` is None.')

        def jac_wrapped(x, _=None):
            return np.empty((0, x0.size))

        return jac_wrapped

    if callable(jac):

        def jac_wrapped(x, _=None):
            return jac(x, *args, **kwargs)

    elif jac in ('2-point', '3-point'):

        def jac_wrapped(x, f):
            return approx_difference(fun, x, f0=f, method=jac, args=args, kwargs=kwargs)

    else:

        raise ValueError("`jac` must be either '2-point', '3-point' or callable.")

    return jac_wrapped


def _hessian_function(hess, constraint=False, args=(), kwargs={}):
    """
    Wrapper for preparing a Hessian for non-linear optimization and constraints.

    Parameters
    ----------
    hess : {'BFGS', 'SR1', callable}
        Function for calculating the Hessian of the objective function. If callable the input should be:
            hess(x)
        and a matrix of shape (n, n).
    constraint : boolean
        If the function is called from a constraint, a function wrapper is returned instead of a class.

    Returns
    -------
    H : Hessian of callable:
        Hessian class or callable.
    """

    if hess is None:

        H = None

    else:
        if constraint:

            def hess_wrapper(x):
                return hess(x, *args, **kwargs)

            H = hess_wrapper

        else:

            H = Hessian(hess, args, kwargs)

    return H


class Hessian:
    def __init__(self, hess=None, args=(), kwargs={}):

        self.callable = callable(hess)
        self.method = None

        if self.callable:

            def hess_wrapped(x):
                return hess(x, *args, **kwargs)

            self.hess = hess_wrapped

        else:

            self.method = hess

            if hess == 'BFGS':

                self.hess = _damped_BFGS_update

            elif hess == 'SR1':

                raise NotImplementedError('SR1 is not implemented yet')

            else:

                raise ValueError("`hess` must be either 'BFGS', 'SR1' or callable.")

    def initialize(self, x0):
        return np.eye(x0.size)

    def calculate(self, x):
        """
        Used when `hess` is callable.

        Parameters
        ----------
        x : array_like, shape (n,)
            Solution vector.

        Returns
        -------
        H : array_like, shape (n, n)
            Hessian matrix.
        """

        return self.hess(x)

    def update(self, B, s, y):
        """
        Used when `hess` is callable.

        Parameters
        ----------
        B : array_like, shape (n, n)
            Approximation of Hessian matrix.
        s : array_like, shape (n,)
            Step of solution vector, dx.
        y : array_like, shape (n,)
            Difference in derivative of Lagrangian from iteration from k to k+1.

        Returns
        -------
        B : array_like, shape (n, n)
            Updated approximation of Hessian matrix.
        """

        return self.hess(B, s, y)


def _chk_callable(x0, f, J=None, H=None):
    """
    Test that dimensions are consistent between the objective function `f`, the Jacobian function `J` and the
    solution vector `x0`

    Parameters
    ----------
    x0 : array_like, shape (n,)
        Initial guess for solution vector.
    f : callable
        Objective function.
    J : callable, optional
        Jacobian function.
    H : class Hessian, optional
        Hessian function.
    """

    n = x0.size

    if x0.ndim != 1:

        raise ValueError('`x0` must be a one-dimensional array of size (n,).')

    # test function
    try:

        f0 = f(x0)

    except (ValueError, IndexError) as e:

        raise ValueError('Function `f` unable to handle initial guess `x0`: {}.'.format(e))

    f0 = np.atleast_1d(f0)
    m = f0.size

    if J is not None:

        try:

            J0 = J(x0, f0)

        except (ValueError, IndexError):

            raise ValueError('Function `J` unable to handle initial guess `x0` and `f(x0)`.')

        J0 = np.atleast_2d(J0)
        if (J0.shape[0] != m) and (J0.shape[1] != n):

            raise ValueError('Shape of output from `J` {} inconsistent with'
                             'expected shape {}.'.format(J0.shape, (m, n)))

    if (H is not None) and H.callable:

        try:

            H0 = H.calculate(x0)

        except (ValueError, IndexError):

            raise ValueError('Function `H` unable to handle initial guess `x0`.')

        H0 = np.atleast_2d(H0)
        if (H0.shape[0] != n) and (H0.shape[1] != n):

            raise ValueError('Shape of output from `H` {} inconsistent with'
                             'expected shape {}.'.format(H0.shape, (n, n)))


def _chk_bounds(lb, ub, n):
    if lb.shape != (n,) and ub.shape != (n,):
        raise ValueError('Bounds have incorrect shape.')

    if np.any(lb >= ub):
        raise ValueError('The lower bound must be less than the upper bound.')


def _atleast_zeros(a, size=(0,)):
    """
    Ensures that the array `a` is indeed an array and not None or empty list/tuple.

    Parameters
    ----------
    b : array_like, shape (m,)
        Vector.
    bn : string, optional
        Variable name of vector `b`. Used for distinguishing between errors when checking multiple systems.

    Returns
    -------
    b : array_like, shape (m,)
        Vector.
    """

    if (a is None) or ((isinstance(a, list) or isinstance(a, tuple)) and not a):

        if isinstance(size, np.ndarray):

            a = np.zeros_like(size)

        elif isinstance(size, tuple) or isinstance(size, list):

            a = np.zeros(size)

        else:
            raise ValueError('`size` must be either an ndarray or a tuple/list.')

    return a


def _ensure_vector(b, bn='b'):
    """
    Ensures the vector `b` is in vector format.

    Parameters
    ----------
    b : array_like, shape (m,)
        Vector.
    bn : string, optional
        Variable name of vector `b`. Used for distinguishing between errors when checking multiple systems.

    Returns
    -------
    b : array_like, shape (m,)
        Vector.
    """

    if b.ndim != 1:
        if (b.ndim == 2) and (np.prod(b.shape) == b.size):
            b = np.reshape(b, (b.size,))
        else:
            raise ValueError('`{}` must be of dimension 1.'.format(bn))

    return b


def _ensure_matrix(A, An='A'):
    """
    Ensures the matrix `A` is in matrix format.

    Parameters
    ----------
    A : array_like, shape (n, m)
        Matrix.
    An : string, optional
        Variable name of matrix `A`. Used for distinguishing between errors when checking multiple systems.

    Returns
    -------
    A : array_like, shape (n, m)
        Matrix.
    """

    if A.ndim == 1:
        A = np.atleast_2d(A).T

    elif A.ndim != 2:
        raise ValueError('`{}` must be of dimension 2.'.format(An))

    return A


def _chk_dimensions(A, b, An='A', bn='b'):
    """
    Ensures the matrix `A` and vector `b` have matching dimensions. Returns the matrix and vector with appropriate
    dimensions for linear algebra if possible.

    Parameters
    ----------
    A : array_like, shape (n, m)
        System matrix.
    b : array_like, shape (m,)
        System results vector.
    An : string, optional
        Variable name of matrix `A`. Used for distinguishing between errors when checking multiple systems.
    bn : string, optional
        Variable name of vector `b`. Used for distinguishing between errors when checking multiple systems.

    Returns
    -------
    A : array_like, shape (n, m)
        System matrix.
    b : array_like, shape (m,)
        System results vector.
    """

    if ((A is None) and (b is not None)) or ((A is not None) and (b is None)):
        raise ValueError('Both `{}` and `{}` must be provided to solve the system of equations.'.format(An, bn))

    if (A is None) and (b is None):
        return None, None

    A = _ensure_matrix(A, An)
    b = _ensure_vector(b, bn)

    if b.shape[0] not in A.shape:
        raise ValueError('Inconsistent shapes between `{}` and `{}`'.format(An, bn))

    return A, b


def _chk_system_dimensions(As=(), bs=(), cs=()):
    """
    Assuming an arbitrary linear algebra computation of the form::

        As[0] @ bs[0] + As[1] @ bs[1] + ... + cs[0] + cs[1] + ...

    Ensures all the matrices in `As` and vectors in `bs` are of compatible dimensions. Further, ensures that
    the addition of the resulting vectors from A @ b are compatible for all A's and b's. Lastly checks
    that all c's in `cs` are compatible with the remaining vectors.

    Returns all of them with appropriate dimensions if possible.

    Assumption is made that As[0].shape[0] is the desired size of the resulting linear algebra operations.

    Parameters
    ----------
    As : tuple or list, optional
        Set of matrices of type array_like, shape (n, m)
    bs : tuple or list, optional
        Set of vectors of type array_like, shape (m)
    cs : tuple or list, optional
        Set of vectors of type array_like, shape (m)
    """

    if (not As) and (not bs) and (not cs):
        return

    if len(As) != len(bs):
        raise ValueError('`As` and `bs` must be of the same length.')

    if As:
        n = As[0].shape[0]
    else:
        n = cs[0].shape[0]

    for i, (A, b) in enumerate(zip(As, bs)):

        if A.shape[0] != n:
            raise ValueError('All matrices in `As` must have the same number of rows.')

        if b is None:
            b = _atleast_zeros(b, size=(A.shape[1]))

    for i, c in enumerate(cs):
        c = _ensure_vector(c, 'cs[{}]'.format(i))

        if c is None:
            c = np.zeros((n,))

        if c.shape[0] != n:
            raise ValueError('All vectors in `c` must be compatible with the result of A*b.')
