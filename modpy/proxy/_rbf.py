import numpy as np
import numpy.linalg as la

from modpy._util import point_distance
from modpy.proxy._proxy_util import _ensure_matrix
from modpy.proxy._cross_validation import k_fold
from modpy.optimize import cma_es


class RBFModel:
    def __init__(self, x, z, method):

        # data
        self.x = _ensure_matrix(x)  # observation points
        self.z = np.array(z)        # responses at observation points

        # radial basis functions
        self.method = method    # str, method used for creating the kernel (RBF)
        self.phi = None         # callable, radial basis function, phi(x, kappa)
        self.kappa = None       # float, shape parameter

        # weights
        self.w = None           # array_like, weights

        self._set_rbf()

    def _set_rbf(self):

        if self.method == 'lin':

            self.phi = linear_rbf

        elif self.method == 'cubic':

            self.phi = cubic_rbf

        elif self.method == 'thin-plate':

            self.phi = thin_plate_rbf

        elif self.method == 'multi-quad':

            self.phi = multi_quadratic_rbf

        elif self.method == 'inv-quad':

            self.phi = inverse_quadratic_rbf

        elif self.method == 'inv-multi-quad':

            self.phi = inverse_multi_quadratic_rbf

        elif self.method == 'gaussian':

            self.phi = gaussian_rbf

        else:

            raise ValueError("`method` must be either 'lin', 'cubic', 'thin-plate',"
                             "'multi-quad', 'inv-quad', 'inv-multi-quad' or 'gaussian'.")

    def initialize(self, kappa):
        self.kappa = kappa
        self.w = _calculate_weights(self.x, self.z, self.phi, kappa)

    def initialize_CV(self, kappa0, k, bounds=None, tol=1e-3, seed=None):
        """
        Initialize the kriging model by performing a Maximum Likelihood estimation of the hyper-parameters
        to the mean function and kernel.

        Parameters
        ----------
        kappa0 : array_like, shape (m+k,)
            Start-guess for the hyper-parameters of the kernel function.
        k : int
            Number of sub-sets used in the k-fold cross validation.
        bounds : tuple, optional
            See `cma_es` for details.
        tol : float, optional
            Tolerance for the CV hyper-parameter estimation.
        seed : int
            Seed of the random number generator (both for K-fold CV and CMA-ES)
        """

        # linear and cubic RBFs are not impacted by the scale parameter kappa.
        if self.method not in ('lin', 'cubic'):

            # retrieve ML objective function
            obj = _cross_validation_objective(self.x, self.z, self.phi, k, seed=seed)

            # ensure kappa > 0
            if bounds is None:
                bounds = ((1., None),)

            # calculate theta numerically
            sigma0 = np.mean(np.amax(self.x, axis=0) - np.min(self.x, axis=0)) * 3.
            res = cma_es(obj, kappa0, bounds=bounds, sigma0=sigma0, tol=tol, seed=seed)

            # TODO: Perhaps try other optimizers?
            if not res.success:
                raise ValueError('Optimization failed: {}'.format(res.message))

            kappa = res.x

        else:

            kappa = 1.

        # initialize the model
        self.initialize(kappa)

    def predict(self, x0):
        return _predict(_ensure_matrix(x0), self.x, self.w, self.phi, self.kappa)


def linear_rbf(r, kappa):
    """
    Computes the Linear Radial Basis Function between two points with distance `r`.

    Parameters
    ----------
    r : float
        Distance between point `x1` and `x2`.
    kappa : float
        Shape parameter.

    Returns
    -------
    phi : float
        Radial basis function response.
    """

    return r * kappa


def cubic_rbf(r, kappa):
    """
    Computes the Cubic Radial Basis Function between two points with distance `r`.

    Parameters
    ----------
    r : float
        Distance between point `x1` and `x2`.
    kappa : float
        Shape parameter.

    Returns
    -------
    phi : float
        Radial basis function response.
    """

    return (r * kappa) ** 3.


def thin_plate_rbf(r, kappa):
    """
    Computes the Thin-Plate Radial Basis Function between two points with distance `r`.

    Parameters
    ----------
    r : float
        Distance between point `x1` and `x2`.
    kappa : float
        Shape parameter.

    Returns
    -------
    phi : float
        Radial basis function response.
    """

    return np.where(r <= 0., 0., (r * kappa) ** 2. * np.log(kappa * r))


def multi_quadratic_rbf(r, kappa):
    """
    Computes the Multi-Quadratic Radial Basis Function between two points with distance `r`.

    Parameters
    ----------
    r : float
        Distance between point `x1` and `x2`.
    kappa : float
        Shape parameter.

    Returns
    -------
    phi : float
        Radial basis function response.
    """

    return np.sqrt((r * kappa) ** 2. + 1.)


def inverse_quadratic_rbf(r, kappa):
    """
    Computes the Inverse-Quadratic Radial Basis Function between two points with distance `r`.

    Parameters
    ----------
    r : float
        Distance between point `x1` and `x2`.
    kappa : float
        Shape parameter.

    Returns
    -------
    phi : float
        Radial basis function response.
    """

    return 1. / (1. + (r * kappa) ** 2.)


def inverse_multi_quadratic_rbf(r, kappa):
    """
    Computes the Inverse-Multi-Quadratic Radial Basis Function between two points with distance `r`.

    Parameters
    ----------
    r : float
        Distance between point `x1` and `x2`.
    kappa : float
        Shape parameter.

    Returns
    -------
    phi : float
        Radial basis function response.
    """

    return 1. / np.sqrt((r * kappa) ** 2. + 1.)


def gaussian_rbf(r, kappa):
    """
    Computes the Gaussian Radial Basis Function between two points with distance `r`.

    Parameters
    ----------
    r : float
        Distance between point `x1` and `x2`.
    kappa : float
        Shape parameter.

    Returns
    -------
    phi : float
        Radial basis function response.
    """

    return np.exp(-(r * kappa) ** 2.)


def _calculate_weights(x, z, phi, kappa):
    """
    Calculates the weights used in the linear prediction of an interpolation point.

    Parameters
    ----------
    x : array_like, shape (n, m)
        Observed points of dimension `m`.
    z : array_like, shape (n,)
        Observed responses at points `x`.
    phi : callable
        Radial basis function.
    kappa : float
        Shape parameter input to `phi`.

    Returns
    -------
    w : array_like, shape (n,)
        Weights used in the recombination.
    """

    P = phi(point_distance(x), kappa)
    w = la.solve(P, z)  # TODO: Cholesky factorization

    return w


def _compute_kernel(x0, x, phi, kappa):
    """
    Computes the kernel matrix.

    Parameters
    ----------
    x0 : array_like, shape (p, m)
        Prediction points of dimension `m`.
    x : array_like, shape (n, m)
        Observed points of dimension `m`.
    phi : callable
        Radial basis function.
    kappa : float
        Shape parameter input to `phi`.

    Returns
    -------
    P0 : array_like, shape (p, n)
        Kernel matrix.
    """

    p, m = x0.shape
    n, _ = x.shape
    P0 = np.zeros((p, n))

    for i in range(p):

        r = la.norm(x - x0[i, :], axis=1)
        P0[i, :] = phi(r, kappa)

    return P0


def _predict(x0, x, w, phi, kappa):
    """
    Calculates the weights used in the linear prediction of an interpolation point.

    Parameters
    ----------
    x0 : array_like, shape (p, m)
        Prediction points of dimension `m`.
    x : array_like, shape (n, m)
        Observed points of dimension `m`.
    w : array_like, shape (n,)
        Weights used in the recombination.
    phi : callable
        Radial basis function.
    kappa : float
        Shape parameter input to `phi`.

    Returns
    -------
    z0 : array_like, shape (p,)
        Predicted values.
    """

    P0 = _compute_kernel(x0, x, phi, kappa)
    z0 = P0 @ w

    return z0.flatten()


def _cross_validation_objective(x, z, phi, k, seed=None):
    """
    Creates a cross-validation estimate objective function based on the observed data (x, y) and the callable
    function `phi`.

    The resulting objective function computes the RMS of the out of sample discrepancies.

    References
    ----------
    [1] Lataniotis, C., Marelli, S., Sudret, B. (2018): "The Gaussian process Modelling Module in UQLab". ETH Zürich.

    Parameters
    ----------
    x : array_like, shape (n, m)
        Observed points of dimension `m`.
    z : array_like, shape (n,)
        Observed responses at points `x`.
    phi : callable
        Radial basis function.
    k : int
        Number of sub-sets used in the k-fold cross validation.
    seed : int
        Seed of the random number generator during shuffling of the sets.

    Returns
    -------
    obj : callable
        Objective function for fitting hyper-parameters to a Gaussian Process kernel function.
    """

    n = z.size
    sets = k_fold(n, k, seed=seed)

    # define objective function
    def obj(kappa):

        errors = [np.empty((0,)) for _ in range(k)]

        for i, (train, test) in enumerate(sets):

            # train the model
            w = _calculate_weights(x[train, :], z[train], phi, kappa)

            # predict the model
            z_pred = _predict(x[test, :], x[train, :], w, phi, kappa)

            # calculate error
            errors[i] = z_pred - z[test]

        # calculate the RMS
        errors = np.hstack(errors)
        rms = np.sum(errors ** 2.)

        return rms

    return obj
