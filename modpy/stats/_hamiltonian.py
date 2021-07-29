import warnings
import numpy as np
import numpy.linalg as la
from numpy.random import Generator, PCG64

from modpy.optimize._optim_util import _function, _jacobian_function, _chk_callable
from modpy.optimize import prepare_bounds
from modpy.stats._core import moving_average

from modpy.stats._stat_util import MCMCResult, MCMCPath


TERMINATION_MESSAGES = {
    0: 'Solution contains NaN or Inf results.',
    1: 'Sampling terminated successfully.',
}


def hamiltonian_mc(f, x0, samples, df='3-point', bounds=None, burn=None, max_tree_depth=10, seed=None, keep_path=False, args=(), kwargs={}):
    """
    Samples from the posterior distribution using a Hamiltonian Monte Carlo method known as NUTS.
    No-U-Turn sampler which adaptively select the path-lengths of the leapfrog integration.
    Further it includes a primal-dual method averaging for modifying the step-size parameter epsilon.
    A reflective approach is used to maintain volume conservation when the domain is bounded.

    References
    ----------
    [1] Hoffman, M. D., Gelman, A. (2014). "The No-U-Turn Sampler: Adaptively Setting Path
        Lengths in Hamiltonian Monte Carlo". Journal of Machine Learning Research 15 (1351-1381).

    [2] Afshar, H. M., Domke, J. (>=2015). "Reflection, Refraction, and Hamiltonian Monte Carlo".  NICTA.

    [3] Aki Nishimura implementation of recycled NUTS.
        https://github.com/aki-nishimura/NUTS-matlab

    Parameters
    ----------
    f : callable
        Function that returns the logarithm of the joint density function:
            p = f(x) = min(0, log-likelihood(x) + prior-probability(x))
    x0 : array_like, (m,)
        Start-guess for the sample parameters.
    samples : int
        Number of samples in the Markov-Chain.
    df : {'2-point', '3-point', callable}
        Function for calculating the gradient of the objective function. If callable the input should be:
            df(x)
        and return a vector of shape (m,)
    bounds : 2-tuple of array_like, optional
        Bounds on the solution vector, should be (array_like (n,), array_like (n,)). If None no bounds are applied.
    burn : int
        Number of samples to discard as burn-in period.
    max_tree_depth : int, optional
        Maximum number of recursions in the tree-builder algorithm.
    seed : int
        Seed of the random number generator.
    keep_path : bool, optional
        Whether to save path information. Can require substantial memory.
    args : tuple
        Additional arguments to `f` and `df`.
    kwargs : dict
        Additional key-word arguments to `f` and `df`.

    Returns
    -------
    thetas : array_like, (samples, m)
        Samples from the posterior distribution.
    """

    # wrap function call with args and kwargs
    x0 = np.array(x0)
    fun = _function(f, args, kwargs)
    dfun = _jacobian_function(f, df, x0, args, kwargs)
    _chk_callable(x0, fun, dfun)

    # set generator
    generator = Generator(PCG64(seed))

    # prepare bounded domain
    m = x0.size
    lb, ub = prepare_bounds(bounds, m)
    domain = (bounds is not None, lb, ub)

    # sample from the posterior distribution
    xp, fp, status, nit, path = _hamiltonian_mc(fun, dfun, x0, samples, domain, burn=burn,
                                                max_tree_depth=max_tree_depth, generator=generator, keep_path=keep_path)

    # collect results
    res = MCMCResult(xp, fp, samples, burn, status=status, nit=nit)
    res.success = status > 0
    res.message = TERMINATION_MESSAGES[res.status]
    res.path = path

    return res


def _hamiltonian_mc(f, df, x0, samples, domain, burn=None, max_tree_depth=10, generator=None, keep_path=False):
    """
    Samples from the posterior distribution using a Hamiltonian Monte Carlo method known as NUTS.
    No-U-Turn sampler which adaptively select the path-lengths of the leapfrog integration.
    Further it includes a primal-dual method averaging for modifying the step-size parameter epsilon.
    A reflective approach is used to maintain volume conservation when the domain is bounded.

    References
    ----------
    [1] Hoffman, M. D., Gelman, A. (2014). "The No-U-Turn Sampler: Adaptively Setting Path
        Lengths in Hamiltonian Monte Carlo". Journal of Machine Learning Research 15 (1351-1381).

    [2] Afshar, H. M., Domke, J. (>=2015). "Reflection, Refraction, and Hamiltonian Monte Carlo".  NICTA.

    [3] Aki Nishimura implementation of recycled NUTS.
        https://github.com/aki-nishimura/NUTS-matlab

    Parameters
    ----------
    f : callable
        Function that returns the logarithm of the joint density function:
            p = f(x) = min(0, log-likelihood(x) + prior-probability(x))
    df : callable
        Function that returns the gradient of the logarithm of the joint density function:
            grad = df(x, f)
        where 'grad' is an array of shape (m,).
    x0 : array_like, (m,)
        Start-guess for the sample parameters.
    samples : int
        Number of samples in the Markov-Chain.
    domain : tuple
        Tuple of (bool, lb, ub).
    burn : int, optional
        Number of samples to discard as burn-in period.
    max_tree_depth : int, optional
        Maximum number of recursions in the tree-builder algorithm.
    generator : np.random.Generator
        Random number generator.
    keep_path : bool, optional
        Whether to save path information. Can require substantial memory.

    Returns
    -------
    xp : array_like, shape (samples, m)
        Samples from the posterior distribution.
    fp : array_like, shape (samples,)
        Probabilities corresponding to the samples.
    """

    if generator is None:
        generator = Generator(PCG64(None))

    m = x0.size
    path = MCMCPath(keep=keep_path)

    if burn is None:
        burn = int(0.1 * m)

    # pre-allocate
    xp = np.zeros((samples + burn, m))
    fp = np.zeros((samples + burn,))
    xp[0] = x0
    fp[0] = f(x0)
    grad = df(x0, fp[0])

    # initial heuristic for hyper-parameters ---------------------------------------------------------------------------
    epsilon = _find_reasonable_epsilon(f, df, x0, domain, generator)
    mu = np.log(10. * epsilon)
    epsilon_bar = 1.
    H_bar = 0.
    gamma = 0.05
    t0 = 10.
    kappa = 0.75
    delta = 0.65

    # main sampling-loop -----------------------------------------------------------------------------------------------
    nit = 0

    for i in range(1, samples + burn):

        # perform single NUTS step
        xp[i], alpha_avg, it, fp[i], grad = _nuts_step(f, df, xp[i - 1], epsilon, domain, generator,
                                                       logp0=fp[i - 1], grad0=grad, max_tree_depth=max_tree_depth)

        # dual-average adaptation of step-size
        if i < burn:

            eta = 1. / (i + t0)
            H_bar = (1. - eta) * H_bar + eta * (delta - alpha_avg)
            epsilon = np.exp(mu - np.sqrt(i) / gamma * H_bar)

            eta = i ** (-kappa)
            epsilon_bar = np.exp((eta * np.log(epsilon)) + (1. - eta) * np.log(epsilon_bar))

            # save acceptance rate and step-size
            path.append(alpha_avg, steps=it, step_size=epsilon_bar)

        else:
            epsilon = epsilon_bar

            # save acceptance rate and step-size
            path.append(alpha_avg, steps=it)

        nit += it

    xp = xp[burn:, :]
    fp = fp[burn:]
    status = np.any(np.isnan(xp) | np.isinf(xp))

    # post-process path variables
    if keep_path:
        path.finalize()
        path.accept = moving_average(path.accept)

    return xp, fp, status, nit, path


def _nuts_step(f, df, theta0, epsilon, domain, generator, logp0=None, grad0=None, max_tree_depth=10):
    """
    Performs a single step of the NUTS Hamiltonian Monte Carlo method.

    References
    ----------
    [1] Hoffman, M. D., Gelman, A. (2014). "The No-U-Turn Sampler: Adaptively Setting Path
        Lengths in Hamiltonian Monte Carlo". Journal of Machine Learning Research 15 (1351-1381).

    [2] Aki Nishimura implementation of recycled NUTS.
        https://github.com/aki-nishimura/NUTS-matlab

    Parameters
    ----------
    f : callable
        Function that returns the logarithm of the joint density function:
            p = f(x) = min(0, log-likelihood(x) + prior-probability(x))
    df : callable
        Function that returns the gradient of the logarithm of the joint density function:
            grad = df(x, f)
        where 'grad' is an array of shape (m,).
    theta0 : array_like, (m,)
        Start-guess for the sample parameters.
    epsilon : float
        Step-size
    domain : tuple
        Tuple of (bool, lb, ub).
    generator : np.random.Generator
        Random generator.
    logp0 : float, optional
        Log-likelihood value from previous step.
    grad0 : array_like, shape (m,), optional
        Gradient of log-likelihood from previous step.
    max_tree_depth : int, optional
        Maximum number of recursions in the tree-builder algorithm.

    Returns
    -------
    theta : array_like, (m,)
        Sample from the posterior distribution.
    """

    # initialize
    theta0 = np.array(theta0)
    m = theta0.size

    # generate initial value of log-likelihood and gradient
    if logp0 is None:
        logp0 = f(theta0)
        grad0 = df(theta0, logp0)

    # sample random momentum
    r0 = generator.normal(size=m)

    # joint log - probability of theta and momentum. eq. (1) of [1]
    joint = float(logp0 - 0.5 * (r0.T @ r0))

    # resample u ~ uniform([0, exp(joint)]). Equivalent to(log(u) - joint) ~ exponential(1)
    logu = joint - generator.exponential(1.)

    # initialize tree
    tm = theta0  # theta minus
    tp = theta0  # theta plus
    rm = r0      # r minus
    rp = r0      # r plus
    gm = grad0   # gradient minus
    gp = grad0   # gradient plus

    # initial height dir = 0.
    j = 0

    # if all else fails, the next sample is the previous sample
    theta = theta0
    grad = grad0
    logp = logp0

    # initially the only valid point is the initial point
    n = 1

    # sub-loop - continue until step is accepted -----------------------------------------------------------------------
    converged = False

    while not converged:

        # Choose a direction: - 1 = backwards, 1 = forwards.
        v = 2 * (generator.random() < 0.5) - 1

        # double the size of the tree.
        if v == -1:

            tm, rm, gm, _, _, _, t_pri, grad_pri, logp_pri, n_pri, stop_pri, alpha, n_alpha =\
                _build_tree(tm, rm, gm, logu, v, j, epsilon, f, df, domain, joint, generator)

        else:

            _, _, _, tp, rp, gp, t_pri, grad_pri, logp_pri, n_pri, stop_pri, alpha, n_alpha =\
                _build_tree(tp, rp, gp, logu, v, j, epsilon, f, df, domain, joint, generator)

        # use Metropolis-Hastings to decide whether or not to move to a point from the half-tree we just generated
        if (not stop_pri) and (generator.random() < (n_pri / n)):

            theta = t_pri
            logp = logp_pri
            grad = grad_pri

        # update number of valid points
        n += n_pri

        # decide if it's time to stop
        converged = stop_pri or _check_convergence(tm, tp, rm, rp)

        # increment depth
        j += 1

        if j > max_tree_depth:
            warnings.warn('The current NUTS iteration reached the maximum tree depth.')
            break

    alpha_avg = float(alpha / n_alpha)

    return theta, alpha_avg, j, logp, grad


def _leapfrog(theta, r, grad, epsilon, f, df, domain):
    """
    Performs a leapfrog integration.

    Parameters
    ----------
    theta : array_like, (m,)
        Current iterate of sample parameters.
    r : array_like, (m,)
        Current momentum value.
    grad : array_like, (m,)
        Gradient of the log-likelihood.
    epsilon : float
        Step-size.
    f : callable
        Function that returns the logarithm of the joint density function:
            p = f(x) = min(0, log-likelihood(x) + prior-probability(x))
    df : callable
        Function that returns the gradient of the logarithm of the joint density function:
            grad = df(x, f)
        where 'grad' is an array of shape (m,).
    domain : tuple
        Tuple of (bool, lb, ub)

    Returns
    -------
    theta : array_like, (m,)
        Current iterate of sample parameters.
    """

    # unpack bounds
    bounded, lb, ub = domain

    # calculate first half-step evolution of momentum
    r_pri = r + 0.5 * epsilon * grad

    t0 = 0.
    if bounded:
        theta, r_pri, t0 = _momentum_reflection(theta, r_pri, epsilon, lb, ub)

    # evolve parameter
    t_pri = theta + (epsilon - t0) * r_pri

    # perform second half-step evolution of momentum
    logp_pri = f(t_pri)
    grad_pri = df(t_pri, logp_pri)
    r_pri += 0.5 * epsilon * grad_pri

    return t_pri, r_pri, logp_pri, grad_pri


def _momentum_reflection(theta, r_pri, epsilon, lb, ub):
    """
    Performs a reflective leapfrog integration honoring a bounded domain.

    References
    ----------
    [1] https://proceedings.neurips.cc/paper/2015/file/8303a79b1e19a194f1875981be5bdb6f-Paper.pdf

    Parameters
    ----------
    theta : array_like, (m,)
        Current iterate of sample parameters.
    r_pri : array_like, (m,)
        Half-step evolution of momentum.
    epsilon : float
        Step-size.
    lb : array_like, shape (m,)
            Lower bound.
    ub : array_like, shape (m,)
        Upper bound.

    Returns
    -------
    theta : array_like, (m,)
        Current iterate of sample parameters.
    """

    t0 = 0.
    bounded = True

    while bounded:

        theta, tx, bounded, idx = _find_boundary(theta, r_pri, epsilon - t0, lb, ub)

        if not bounded:
            break

        t0 += tx
        r_par, r_per = _decompose_vector(r_pri, idx)

        # ignore refraction, only allow reflection
        r_pri = r_par - r_per

    return theta, r_pri, t0


def _find_boundary(theta, r, epsilon, lb, ub):
    """

    Parameters
    ----------
    theta : array_like, (m,)
        Current iterate of sample parameters.
    r : array_like, (m,)
        Current momentum value.
    epsilon : float
        Step-size.
    lb : array_like, shape (m,)
            Lower bound.
    ub : array_like, shape (m,)
        Upper bound.

    Returns
    -------

    """

    # calculate new position of full potential
    theta_pot = theta + epsilon * r

    # calculate potential constraint violations, <0 is a violation.
    theta_lb = theta_pot - lb
    theta_ub = ub - theta_pot

    mask_lb = theta_lb < 0.
    mask_ub = theta_ub < 0.

    # find tx at violation of first constraint
    tx_lb = epsilon
    if np.any(mask_lb):
        # solve theta + tx * r - lb >= 0 for tx
        tx_lb = (lb[mask_lb] - theta[mask_lb]) / r[mask_lb]

    tx_ub = epsilon
    if np.any(mask_ub):
        # solve ub - (theta + tx * r) >= 0 for tx
        tx_ub = (ub[mask_ub] - theta[mask_ub]) / r[mask_ub]

    # find minimum of tx
    tx = np.block([tx_lb, tx_ub]).flatten()
    tx = tx[np.argmin(np.abs(tx))]

    # if no boundary was met, return
    if tx == epsilon:
        return theta, epsilon, False, None

    # calculate value at boundary
    theta_pri = theta + tx * r

    # find index of violated boundary
    if np.any(tx_lb == tx):
        idx = _reverse_mask_idx(mask_lb, np.argmin(np.abs(tx_lb)))  # tx == tx_lb

    else:
        idx = _reverse_mask_idx(mask_ub, np.argmin(np.abs(tx_ub)))  # tx == tx_ub

    return theta_pri, tx, True, idx


def _reverse_mask_idx(mask, idx):
    it = 0
    for i, m in enumerate(mask):
        if m and (it == idx):
            return i

        if m:
            it += 1


def _decompose_vector(r, idx):
    """

    Parameters
    ----------
    r : array_like, (m,)
        Current momentum value.
    idx : int
        Index of violated boundary

    Returns
    -------

    """

    # handle special case of 1-dimensional problem
    if r.size == 1:
        return np.zeros_like(r), r

    # define vector of the parallel plane
    # no need to perform projection as lengths should be equal.
    r_par = np.array(r)
    r_par[idx] = 0.

    # calculate perpendicular vector
    r_per = r - r_par

    return r_par, r_per


def _check_convergence(theta_minus, theta_plus, r_minus, r_plus):

    theta_vec = theta_plus - theta_minus
    return (theta_vec.T @ r_minus < 0.) or (theta_vec.T @ r_plus < 0.)


def _build_tree(theta, r, grad, logu, v, j, epsilon, f, df, domain, joint0, generator):

    if j == 0:

        # base case: Take a single leapfrog step in the direction 'dir'
        t_pri, r_pri, logp_pri, grad_pri = _leapfrog(theta, r, grad, v * epsilon, f, df, domain)
        joint = float(logp_pri - 0.5 * (r_pri.T @ r_pri))

        # is the new point in the slice?
        n_pri = int(logu <= joint)

        # is the simulation wildly inaccurate?
        stop_pri = bool((logu - 1000.) >= joint)

        # set the return values - minus=plus for all things here, since the "tree" is of depth 0.
        tm = t_pri
        tp = t_pri
        rm = r_pri
        rp = r_pri
        gm = grad_pri
        gp = grad_pri

        # compute the acceptance probability
        alpha_pri = np.exp(joint - joint0)

        if np.isnan(alpha_pri):
            alpha_pri = 0.
        else:
            alpha_pri = np.minimum(1., alpha_pri)

        n_alpha_pri = 1

    else:
        # recursion: implicitly build the height depth-1 left and right subtrees.
        tm, rm, gm, tp, rp, gp, t_pri, grad_pri, logp_pri, n_pri, stop_pri, alpha_pri, n_alpha_pri =\
            _build_tree(theta, r, grad, logu, v, j - 1, epsilon, f, df, domain, joint0, generator)

        # no need to keep going if the stopping criteria were met in the first subtree
        if not stop_pri:

            # the maximum number of states we would potentially recycle from the subtree
            if v == -1:

                tm, rm, gm, _, _, _, t_pri2, grad_pri2, logp_pri2, n_pri2, stop_pri2, alpha_pri2, n_alpha_pri2 =\
                    _build_tree(tm, rm, gm, logu, v, j - 1, epsilon, f, df, domain, joint0, generator)

            else:

                _, _, _, tp, rp, gp, t_pri2, grad_pri2, logp_pri2, n_pri2, stop_pri2, alpha_pri2, n_alpha_pri2 =\
                    _build_tree(tp, rp, gp, logu, v, j - 1, epsilon, f, df, domain, joint0, generator)

            # choose which subtree to propagate a sample up from
            accept = 0. if n_pri2 == 0 else (n_pri2 / (n_pri + n_pri2))

            if generator.random() < accept:

                t_pri = t_pri2
                grad_pri = grad_pri2
                logp_pri = logp_pri2

            # update the stopping criterion
            stop_pri = stop_pri2 or _check_convergence(tm, tp, rm, rp)  # stop_pri or

            # update the acceptance probability statistics
            alpha_pri += alpha_pri2
            n_alpha_pri += n_alpha_pri2

    return tm, rm, gm, tp, rp, gp, t_pri, grad_pri, logp_pri, n_pri, stop_pri, alpha_pri, n_alpha_pri


def _find_reasonable_epsilon(f, df, theta0, domain, generator):
    """
    Calculates a reasonable step-size for the NUTS algorithm. This is based on Algorithm 4 of [1].
    Notice the algorithm is changed to work in log-space to avoid underflow errors.

    References
    ----------
    [1] Hoffman, M. D., Gelman, A. (2014). "The No-U-Turn Sampler: Adaptively Setting Path
        Lengths in Hamiltonian Monte Carlo". Journal of Machine Learning Research 15 (1351-1381).

    Parameters
    ----------
    f : callable
        Function that returns the logarithm of the joint density function:
            p = f(x) = min(0, log-likelihood(x) + prior-probability(x))
    df : callable
        Function that returns the gradient of the logarithm of the joint density function:
            grad = df(x, f)
        where 'grad' is an array of shape (m,).
    theta0 : array_like, (m,)
        Start-guess for the sample parameters.
    domain : tuple
        Tuple of (bool, lb, ub)
    generator : np.random.Generator
        Random generator.

    Returns
    -------
    epsilon : float
        Step-size.
    """

    m = theta0.size

    epsilon = 1.
    r0 = generator.normal(size=m)
    logp0 = f(theta0)
    grad0 = df(theta0, logp0)

    # perform single leapfrog integration
    _, r_pri, logp_pri, _ = _leapfrog(theta0, r0, grad0, epsilon, f, df, domain)

    # calculate probability (in log space)
    p = logp0 - 0.5 * r0.T @ r0
    p_pri = logp_pri - 0.5 * r_pri.T @ r_pri

    # calculate epsilon change exponent
    accept = p_pri - p
    a = float(2 * (accept > np.log(0.5)) - 1)
    alog2 = a * np.log(2.)

    while (a * accept) > -alog2:  # condition in log-space

        epsilon *= 2. ** a

        _, r_pri, logp_pri, _ = _leapfrog(theta0, r0, grad0, epsilon, f, df, domain)

        # update log probability
        p_pri = logp_pri - 0.5 * r_pri.T @ r_pri
        accept = p_pri - p

    return epsilon
