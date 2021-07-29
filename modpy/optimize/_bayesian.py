import numpy as np

from modpy.random import normal_pdf, normal_cdf


def bayesian_proposal(model, opt, acq='EI', thresh=0.5, args=(), kwargs={}):
    """
    Performs a single step optimization of the Acquisition Function, finding one or more proposals for the next
    iteration of function calls in a Bayesian Optimization problem.

    Parameters
    ----------
    model : class
        Proxy model used to generate the mean and standard deviation during the optimization process.
            mean, std = model(x)
        where 'mean' and 'std' are floats.
    opt : callable
        Optimization algorithm for finding multiple optimums. The algorithm should be callable as follows:
            (res1, res2, ...,) = opt(obj)
        where each 'res' is of class OptimizeResult and 'obj' is an objective function.
    acq : {'EI', 'POI', 'UCB'}, optional
        Acquisition function to be optimized for proposal of new parameters to be used in function calls.

            'EI': Expected Improvement
            'POI': Probability of Improvement
            'UCB': Upper Confidence Bounds

    thresh : float, optional
        thresh [0, 1] is the ratio a local optimum value must satisfy relative to the global optimum to be included
        in the proposed runs set, i.e.:
            include = f_local / f_global >= tresh
    args : tuple
        Additional arguments to `acq`. When 'acq' is a string, additional arguments such as 'y_max', 'xi' and 'kappa'
        can be passed in `args`.
    kwargs : dict
        Additional key-word arguments to `acq`.

    Returns
    -------
    res : tuple
        Tuple of class OptimizeResult.
    """

    # define acquisition function. Notice negative sign to perform maximization.
    if acq == 'EI':

        def _obj(x):
            return - _expected_improvement(*model(x), *args, **kwargs)

    elif acq == 'POI':

        def _obj(x):
            return - _probability_of_improvement(*model(x), *args, **kwargs)

    elif acq == 'UCB':

        def _obj(x):
            return - _upper_confidence_bound(*model(x), *args, **kwargs)

    else:

        raise ValueError("`acq` must either 'EI', 'POI' or 'UCB'.")

    # check threshold parameter
    if not (0 <= thresh <= 1.):
        raise ValueError('`thresh` must be between 0 and 1.')

    # perform multi-modal optimization
    results = opt(_obj)

    # negate objective function values to account for "-" in '_obj'
    for res in results:
        res.f = -res.f

    # sort the results in descending order
    fs = np.array([res.f if isinstance(res.f, float) else res.f[0] for res in results])
    idx = np.flip(np.argsort(fs))
    results = [results[i] for i in idx if results[i].success]

    #  remove all results below the threshold
    if len(results) > 1:
        fg = results[0].f
        results = [res for res in results if (res.f / fg) >= thresh]

    return tuple(results)


def _expected_improvement(mu, sigma, y_max, xi):
    """
    Calculates the expected improvement at the point 'x', for which:
        mu = mean(x)
        sigma = std(x)

    TODO: should be in log-domain?

    Parameters
    ----------
    mu : array_like, shape (n,)
        Mean.
    sigma : array_like, shape (n,)
        Standard deviation.
    y_max : float
        Maximum value from a sequence of function calls.
    xi : float
        Exploration parameter.

    Returns
    -------
    ei : array_like, shape (n,)
        Expected improvement at point 'x'.
    """

    mu_y = mu - y_max
    z = (mu_y - xi) / sigma
    return np.where(sigma == 0., 0., mu_y * normal_cdf(z) + sigma * normal_pdf(z))


def _probability_of_improvement(mu, sigma, y_max, xi):
    """
    Calculates the probability of improvement at the point 'x', for which:
        mu = mean(x)
        sigma = std(x)

    TODO: should be in log-domain?

    Parameters
    ----------
    mu : array_like, shape (n,)
        Mean.
    sigma : array_like, shape (n,)
        Standard deviation.
    y_max : float
        Maximum value from a sequence of function calls.
    xi : float
        Exploration parameter.

    Returns
    -------
    ei : array_like, shape (n,)
        Expected improvement at point 'x'.
    """

    return np.where(sigma == 0., 0., normal_cdf((mu - y_max - xi) / sigma))


def _upper_confidence_bound(mu, sigma, kappa=2.):
    """
    Calculates the upper confidence bound at the point 'x', for which:
        mu = mean(x)
        sigma = std(x)

    Parameters
    ----------
    mu : array_like, shape (n,)
        Mean.
    sigma : array_like, shape (n,)
        Standard deviation.
    kappa : float
        Exploration parameter.

    Returns
    -------
    ei : array_like, shape (n,)
        Expected improvement at point 'x'.
    """

    return mu + kappa * sigma
