import numpy as np
import numpy.linalg as la
from numpy.random import Generator, PCG64

from modpy.proxy._proxy_util import _ensure_matrix
from modpy.optimize import cma_es, lsq_linear, least_squares


class KrigingModel:
    def __init__(self, x, z, f, method, seed=None, args=()):

        # kriging type
        self.type = None     # str, {'simple', 'ordinary', 'universal', 'nl'}

        # data
        self.x = _ensure_matrix(x)  # observation points, shape (n, m)
        self.z = np.array(z)        # responses at observation points, shape (n,)

        # trend functions
        self.f = f              # callable function for trend
        self.beta = None        # array_like, shape (k,), mean coefficients

        # covariance functions
        self.method = method    # str, method used for creating the kernel
        self.corr = None        # callable, correlation function, corr(x, theta)
        self.kernel = None      # callable, kernel(x, sigma, theta)
        self.sigma = None       # float, standard deviation
        self.theta = None       # array_like, shape (n,), bandwidths
        self.C = None           # array_like (n, n), Covariance Matrix, result of  C=kernel(x, sigma, theta)
        self.Q = None           # array_like (n, n) Q from QR decomposition of C
        self.R = None           # array_like (n, n) R from QR decomposition of C, NOT correlation matrix
        self.bounds = None      # tuple of tuple ((min1, max1), ...), bounds on theta
        self.tol = None         # float, tolerance in estimation of theta
        self.manual = True      # bool, whether parameters were set manually
        self.args = args        # tuple, additional argument to corr

        # sampling options
        self._seed = seed
        self._gen = Generator(PCG64(seed))

        # results temporarily stored in memory for prediction points
        self.x0 = None          # array_like, estimation points
        self.trends = ()        # tuple of array_like, trends
        self.covariances = ()   # tuple of array_like, covariances
        self.weights = ()       # tuple of array_like, weights
        self.means = None       # array_like, means
        self.variances = None   # array_like, variances
        self.lagrangian = ()    # tuple of array_like, Lagrangian multipliers

        self._set_correlation()

    def _set_correlation(self):

        if self.method == 'exp':

            self.corr = exponential_correlation

        elif self.method == 'gaussian':

            self.corr = gaussian_correlation

        elif self.method == 'matern32':

            self.corr = matern32_correlation

        elif self.method == 'matern52':

            self.corr = matern52_correlation

        elif self.method == 'pow-exp':

            self.corr = power_exponential_correlation

        else:

            raise ValueError("`method` must be either 'exp', 'gaussian', 'matern32', 'matern52' or 'pow-exp'.")

    def _QR_decomposition(self, C):
        raise NotImplementedError

    def _rhs(self, c0, f0):
        raise NotImplementedError

    def _ensure_weights(self):
        if not self.weights:
            raise ValueError('Weights have not been defined, call `define_weights(x)`.')

    def _ensure_means(self):
        if self.means is None:
            raise ValueError('Means have not been defined, call `mean()`.')

    def _ensure_variances(self):
        if self.variances is None:
            raise ValueError('Variances have not been defined, call `variance()`.')

    def variance(self):
        pass

    def std(self):
        return np.sqrt(self.variance())

    def define_weights(self, x0):
        x0 = _ensure_matrix(x0)
        self.x0 = x0

        n, _ = x0.shape
        k = self.z.size

        # pre-allocate
        self.trends = [np.empty((0,)) for _ in range(n)]
        self.covariances = [np.empty((0,)) for _ in range(n)]
        self.weights = [np.empty((0,)) for _ in range(n)]
        self.lagrangian = [np.empty((0,)) for _ in range(n)]

        # define the covariances and weights
        for i in range(n):

            # calculate trend and covariance
            f0 = self.f(x0[i, :])
            c0 = self.kernel(x0[i, :])

            # define RHS and solve for weights
            rhs = self._rhs(c0, f0)
            w = la.solve(self.R, self.Q.T @ rhs)

            # save for later use
            self.trends[i] = f0
            self.covariances[i] = c0
            self.weights[i] = w[:k]
            self.lagrangian[i] = -w[k:]

    def sample(self, size, posterior=True):

        C = _compute_kernel(self.sigma, _compute_matrix_correlations(self.x0, self.corr, self.theta, args=self.args))

        # compute posterior covariance matrix
        if posterior:
            self._ensure_means()
            self._ensure_variances()

            m = self.means
            C12 = np.vstack(self.covariances)
            C = C - C12 @ la.solve(self.C, C12.T)
        else:
            m = self.f(self.x0) @ self.beta

        return self._gen.multivariate_normal(m, C, size)

    def initialize(self, beta, theta, sigma, R=None):
        """
        Initializes the kriging model for a given set of hyper-parameters.

        Parameters
        ----------
        beta : array_like, shape (p,)
            Hyper-parameters of the trend-function.
        theta : array_like, shape (m,)
            Hyper-parameters of the correlation function.
        sigma : float
            Hyper-parameter of the covariance function.
        R : array_like, shape (n, n), optional
            Correlation matrix
        """

        # set hyper-parameters
        self.theta = theta
        self.beta = beta
        self.sigma = sigma

        # define the kernel function
        def kernel(x0):
            h = np.abs(self.x - x0)
            R_ = _compute_correlations(h, self.corr, theta, args=self.args)
            return _compute_kernel(sigma, R_)

        self.kernel = kernel

        # calculate the correlation matrix of the existing data (if not supplied)
        if R is None:
            R = _compute_matrix_correlations(self.x, self.corr, theta, args=self.args)

        # compute covariance matrix and its decomposition
        self.C = _compute_kernel(sigma, R)
        self.Q, self.R = self._QR_decomposition(self.C)  # TODO: better than QR?

    def initialize_ML(self, theta0, bounds=None, tol=1e-3):
        """
        Initialize the kriging model by performing a Maximum Likelihood estimation of the hyper-parameters
        to the mean function and kernel.

        Parameters
        ----------
        theta0 : array_like, shape (m+k,)
            Start-guess for the hyper-parameters of the mean and kernel function.
        bounds : tuple, optional
            See `cma_es` for details.
        tol : float, optional
            Tolerance for the ML hyper-parameter estimation.
        """

        n, m = self.x.shape

        # retrieve ML objective function
        obj = _maximum_likelihood_objective(self.type, self.x, self.z, self.f, self.corr, args=self.args)

        # ensure theta > 0
        if bounds is None:
            bounds = tuple([(1e-5, None) for _ in range(m)])

        # calculate theta numerically
        sigma0 = np.mean(theta0) / 3.
        res = cma_es(obj, theta0, bounds=bounds, sigma0=sigma0, tol=tol, seed=self._seed)

        # TODO: Perhaps try other optimizers?
        if not res.success:
            raise ValueError(res.message)

        theta = res.x

        # calculate correlation matrix.
        R = _compute_matrix_correlations(self.x, self.corr, theta, args=self.args)
        Rinv = la.solve(R, np.eye(n))

        # calculate beta analytically
        _beta_estimator = _get_beta_estimator(self.type, self.x, self.z, self.f)
        beta, mu = _beta_estimator(Rinv)

        # calculate the standard deviation analytically
        sigma, _ = _calculate_sigma_ML(self.z, mu, Rinv)

        # initialize the model
        self.initialize(beta, theta, sigma, R=R)

        # set objects for sub-sequent updates
        self.manual = False
        self.bounds = bounds
        self.tol = tol

    def update(self, x, z):
        x = _ensure_matrix(x)
        self.x = _ensure_matrix(np.vstack((self.x, x)))
        self.z = np.append(self.z, z)

        # update estimated parameters based on new data
        if self.manual:
            self.initialize(self.beta, self.theta, self.sigma)
        else:
            self.initialize_ML(self.theta, self.bounds, self.tol)


class SimpleKrigingModel(KrigingModel):
    """
    References
    ----------
    [1] Lataniotis, C., Marelli, S., Sudret, B. (2018): "The Gaussian process Modelling Module in UQLab". ETH Zürich.
    """

    def __init__(self, x, z, method, seed=None, args=()):

        def f(_):
            return None

        super().__init__(x, z, f, method, seed=seed, args=args)

        self.type = 'simple'

    def _QR_decomposition(self, C):
        return la.qr(C)

    def _rhs(self, c0, f0):
        return c0

    def mean(self):
        self._ensure_weights()

        n, _ = self.x0.shape
        m = np.zeros((n,))

        mu = self.beta

        # calculate the mean response
        for i in range(n):
            w = self.weights[i]
            m[i] = mu + w.T @ (self.z - mu).T

        self.means = m
        return m

    def variance(self):
        self._ensure_means()

        n = len(self.weights)
        v = np.ones((n,))
        s2 = self.sigma ** 2.

        # calculate the variance
        for i in range(n):
            c0 = self.covariances[i]
            w = self.weights[i]

            v[i] = s2 - w.T @ c0

        v = np.clip(v, 0., None)
        self.variances = v
        return v


class OrdinaryKrigingModel(KrigingModel):
    """
    References
    ----------
    [1] Lataniotis, C., Marelli, S., Sudret, B. (2018): "The Gaussian process Modelling Module in UQLab". ETH Zürich.
    """

    def __init__(self, x, z, method, seed=None, args=()):

        def f(x_):
            return np.ones((x_.shape[0], 1))

        super().__init__(x, z, f, method, seed=seed, args=args)

        self.type = 'ordinary'

    def _QR_decomposition(self, C):
        n, _ = C.shape
        gamma = np.block([[C, np.ones((n, 1))],
                          [np.ones((1, n)), 0.]])

        return la.qr(gamma)

    def _rhs(self, c0, f0):
        return np.block([c0, 1.])

    def mean(self):
        self._ensure_weights()

        n, _ = self.x0.shape
        m = np.zeros((n,))

        for i in range(n):
            w = self.weights[i]
            m[i] = w.T @ self.z.T

        self.means = m
        return m

    def variance(self):
        self._ensure_means()

        n = len(self.weights)
        v = np.ones((n,))
        s2 = self.sigma ** 2.

        # calculate the variance
        for i in range(n):
            c0 = self.covariances[i]
            w = self.weights[i]
            lam = self.lagrangian[i]

            # eq. (7.8) of [1]
            v[i] = s2 - w.T @ c0 + lam

        v = np.clip(v, 0., None)
        self.variances = v
        return v


class UniversalKrigingModel(KrigingModel):
    """
    References
    ----------
    [1] Lataniotis, C., Marelli, S., Sudret, B. (2018): "The Gaussian process Modelling Module in UQLab". ETH Zürich.
    """

    def __init__(self, x, z, f, method, seed=None, args=()):
        """

        Parameters
        ----------
        x : array_like, shape (n, m)
            Observed points of dimension `m`.
        z : array_like, shape (n,)
            Observed responses at points `x`.
        f : callable
            Trend function. Should return a matrix of the form:

                f(x) = [1., f1(x), f2(x), ..., fp(x)]

            which is an (n, p+1) matrix. It is required that the output is a column of ones.

        method : {'exp', 'gaussian', 'matern32', 'matern52', 'pow-exp'}
            String indicating which correlation method to use.
        args : tuple
            Additional arguments to `corr`, such as 'p' for the power-exponential method.
        """

        super().__init__(x, z, f, method, seed=seed, args=args)

        self.type = 'universal'

    def _QR_decomposition(self, C):
        n, _ = C.shape
        F = self.f(self.x)
        p = F.shape[1]

        gamma = np.block([[C, F],
                          [F.T, np.zeros((p, p))]])

        return la.qr(gamma)

    def _rhs(self, c0, f0):
        return np.block([c0, f0.flatten()])

    def mean(self):
        self._ensure_weights()

        n, _ = self.x0.shape
        m = np.zeros((n,))

        for i in range(n):
            w = self.weights[i]
            m[i] = w.T @ self.z.T

        self.means = m
        return m

    def variance(self):
        self._ensure_means()

        n = len(self.weights)
        v = np.ones((n,))
        s2 = self.sigma ** 2.

        # calculate the variance
        for i in range(n):
            f0 = self.trends[i].flatten()
            c0 = self.covariances[i]
            w = self.weights[i]
            lam = self.lagrangian[i]

            # eq. (8.8) of [1]
            v[i] = s2 - w.T @ c0 + lam.T @ f0

        v = np.clip(v, 0., None)
        self.variances = v
        return v


class NonLinearKrigingModel(KrigingModel):
    def __init__(self, x, z, f, method, seed=None, args=()):
        super().__init__(x, z, f, method, seed=seed, args=args)

        self.type = 'nl'


def exponential_correlation(h, theta):
    """
    Computes the Exponential correlation between two points with distance `h`.

    Parameters
    ----------
    h : float
        Distance between point `x1` and `x2`.
    theta : array_like, shape (m,)
        Length scale or bandwidth.

    Returns
    -------
    correlation : float
        The correlation between two points.
    """

    return np.exp(-h / theta)


def gaussian_correlation(h, theta):
    """
    Computes the Gaussian correlation between two points with distance `h`.

    Parameters
    ----------
    h : float
        Distance between point `x1` and `x2`.
    theta : array_like, shape (m,)
        Length scale or bandwidth.

    Returns
    -------
    correlation : float
        The correlation between two points.
    """

    return np.exp(-0.5 * (h / theta) ** 2.)


def matern32_correlation(h, theta):
    """
    Computes the Matérn 3/2 correlation between two points with distance `h`.

    Parameters
    ----------
    h : float
        Distance between point `x1` and `x2`.
    theta : array_like, shape (m,)
        Length scale or bandwidth.

    Returns
    -------
    correlation : float
        The correlation between two points.
    """

    sq3_h_theta = np.sqrt(3) * np.abs(h) / theta
    return (1. + sq3_h_theta) * np.exp(-sq3_h_theta)


def matern52_correlation(h, theta):
    """
    Computes the Matérn 5/2 correlation between two points with distance `h`.

    Parameters
    ----------
    h : float
        Distance between point `x1` and `x2`.
    theta : array_like, shape (m,)
        Length scale or bandwidth.

    Returns
    -------
    correlation : float
        The correlation between two points.
    """

    sq5_h_theta = np.sqrt(5) * np.abs(h) / theta
    return (1. + sq5_h_theta + 5. / 3. * (h / theta) ** 2.) * np.exp(-sq5_h_theta)


def power_exponential_correlation(h, theta, p=2.):
    """
    Computes the Power-Exponential correlation between two points with distance `h`.

    Parameters
    ----------
    h : float
        Distance between point `x1` and `x2`.
    theta : array_like, shape (m,)
        Length scale or bandwidth.
    p : float
        Power.

    Returns
    -------
    correlation : float
        The correlation between two points.
    """

    return np.exp(-(np.abs(h) / theta) ** p)


def _compute_correlations(h, corr, theta, args=()):
    """
    Computes the correlations of a Gaussian Process.

    Parameters
    ----------
    h : array_like, shape (n, m)
        Distances in `m` dimensions.
    corr : callable
        Correlation function.
    theta : array_like, shape (m,)
        Length scale or bandwidth.
    args : tuple
        Additional arguments to `corr`.

    Returns
    -------
    R : array_like, shape (n, n)
        Correlations Matrix of a Gaussian Process.
    """

    return np.prod(corr(h, theta, *args), axis=1)


def _compute_matrix_correlations(x, corr, theta, args=()):
    """
    Computes the correlations of a Gaussian Process for all points in `x` with respect to each other.

    Parameters
    ----------
    x : array_like, shape (n, m)
        Observed points of dimension `m`.
    corr : callable
        Correlation function.
    theta : array_like, shape (m,)
        Length scale or bandwidth.
    args : tuple
        Additional arguments to `corr`.

    Returns
    -------
    R : array_like, shape (n, n)
        Correlations Matrix of a Gaussian Process.
    """

    n, m = x.shape
    R = np.eye(n)

    for i in range(n):

        h = np.abs(x - x[i, :])

        for j in range(i + 1, n):

            R[i, j:] = _compute_correlations(h[j:, :], corr, theta, args=args)

    # ensure symmetry
    R = np.triu(R) + np.triu(R, 1).T

    return R


def _compute_kernel(sigma, R):
    """
    Computes the kernel of a Gaussian Process.

    Parameters
    ----------
    sigma : float
        Standard deviation.
    R : array_like
        Matrix or vector of correlation coefficients.

    Returns
    -------
    C : array_like
        Kernel or Covariance Matrix of a Gaussian Process.
    """

    return sigma ** 2. * R


def _maximum_likelihood_objective(type_, x, z, f, corr, args=()):
    """
    Creates a maximum likelihood estimate objective function based on the observed data (x, y) and the callable
    functions `mean` and `corr` for which to fit hyper-parameters to.

    The resulting objective function computes the negative log-likelihood of a Gaussian Process.

    References
    ----------
    [1] Le Riche, R. (2014). "Introduction to Kriging". HAL id: cel-01081304.
    [2] Freier, L., Wiechert, W., von Lieres, E. (2017). "Kriging with trend functions nonlinear in their parameters:
        Theory and application in enzyme kinetics". Engineering in Life Sciences 17, 916-922.
    [3] Lataniotis, C., Marelli, S., Sudret, B. (2018): "The Gaussian process Modelling Module in UQLab". ETH Zürich.

    Parameters
    ----------
    type_ : str, {'simple', 'ordinary', 'universal', 'nl'}
        Type of kriging model.
    x : array_like, shape (n, m)
        Observed points of dimension `m`.
    z : array_like, shape (n,)
        Observed responses at points `x`.
    f : callable
        Trend function.
    corr : callable
        Correlation function.
    args : tuple
        Additional arguments to `corr`.

    Returns
    -------
    obj : callable
        Objective function for fitting hyper-parameters to a Gaussian Process kernel function.
    """

    n, m = x.shape

    # define function for calculating beta
    _beta_estimator = _get_beta_estimator(type_, x, z, f)

    # define objective function
    def obj(theta):

        R = _compute_matrix_correlations(x, corr, theta, args=args)
        Rinv = la.solve(R, np.eye(n))

        # calculate the trend coefficients
        beta, mu = _beta_estimator(Rinv)

        # calculate the standard deviation
        sigma, dRd = _calculate_sigma_ML(z, mu, Rinv)

        # calculate the log likelihood
        mll = .5 * np.log(la.det(R)) + n / 2. * np.log(2. * np.pi * sigma ** 2.) + n / 2

        return mll

    return obj


def _minimum_kriging_variance_objective(x, z, mean, corr, args=()):
    """
    Creates a minimum kriging variance estimate objective function based on the observed data (x, y) and the callable
    functions `mean` and `corr` for which to fit hyper-parameters to.

    The resulting objective function computes the minimum kriging variance of a Gaussian Process.

    References
    ----------
    [1] https://sigopt.com/blog/kernel-based-interpolation-likelihood-vs-kriging-variance/

    Parameters
    ----------
    x : array_like, shape (n, m)
        Observed points of dimension `m`.
    z : array_like, shape (n,)
        Observed responses at points `x`.
    mean : callable
        Mean function.
    corr : callable
        Correlation function.
    args : tuple
        Additional arguments to `corr`.

    Returns
    -------
    obj : callable
        Objective function for fitting hyper parameters to a Gaussian Process kernel function.
    """

    # n, m = x.shape
    #
    # def obj(parameters):
    #
    #     theta = parameters[:m]  # parameters used in fitting the kernel function.
    #     beta = parameters[m:]   # parameters used in fitting the trend function
    #
    #     # calculate trend
    #     mu = mean(x, beta)
    #
    #     # calculate the correlation matrix and standard deviation
    #     sigma, R, dRd = _calculate_sigma_and_R(x, z, mu, corr, theta, args=())
    #
    #     # calculate the minimum kriging variance [1]
    #     kv = np.log(dRd) + np.log(la.norm(R))
    #
    #     return kv
    #
    # return obj


def _get_beta_estimator(type_, x, z, f):
    """
    Returns a method for estimation of the trend parameters, beta.

    References
    ----------
    [1] Lataniotis, C., Marelli, S., Sudret, B. (2018): "The Gaussian process Modelling Module in UQLab". ETH Zürich.
    [2] Freier, L., Wiechert, W., von Lieres, E. (2017). "Kriging with trend functions nonlinear in their parameters:
        Theory and application in enzyme kinetics". Engineering in Life Sciences 17, 916-922.

    Parameters
    ----------
    type_ : str, {'simple', 'ordinary', 'universal', 'nl'}
        Type of kriging model.
    x : array_like, shape (n, m)
        Observed points of dimension `m`.
    z : array_like, shape (n,)
        Observed responses at points `x`.
    f : {float, array_like, callable}
        Trend function. Depends on the type of kriging:

            Simple kriging:     float
            Ordinary kriging:   np.ones((n,))
            Universal kriging:  array_like, ((n, , m * p))
            Non-linear kriging: callable (Jacobian of a user function)

    Returns
    -------
    estimator : callable
        Method used for estimation of trend parameters, beta, and the trend, mu.
    """

    # calculate trend analytically [1]
    if type_ == 'simple':

        def estimator(_):
            beta = np.mean(z)
            return beta, beta

    elif type_ in ('ordinary', 'universal'):

        def estimator(Rinv):
            F = f(x)
            beta = lsq_linear(F, z, Rinv).x
            mu = F @ beta

            return beta, mu

    elif type_ == 'nl':

        def estimator(Rinv):
            # TODO: update to handle weighted nl_lsq
            beta = least_squares(f, np.mean(z))
            mu = f(x, beta)

            return beta, mu

    else:

        raise ValueError("`type_` must be 'simple', 'ordinary', 'universal' or 'nl'.")

    return estimator


def _calculate_sigma_ML(z, mu, Rinv):
    """
    Calculates the standard deviation, sigma.

    References
    ----------
    [1] Le Riche, R. (2014). "Introduction to Kriging". HAL id: cel-01081304.

    Parameters
    ----------
    z : array_like, shape (n,)
        Observed responses at points `x`.
    mu : array_like, shape (n,)
        Mean values.
    Rinv : array_like, shape (n, n)
        Inverted correlation matrix.

    Returns
    -------
    sigma : float
        Standard deviation
    dRd : float
        The product of (z-mu)' Rinv (z-mu)
    """

    n = z.size

    # calculate variance parameter analytically
    dmz = (z - mu)
    dRd = dmz.T @ Rinv @ dmz
    sigma = np.sqrt(dRd / n)

    return sigma, dRd
