import numpy as np
from numpy.random import Generator, PCG64

from modpy.optimize._optim_util import _chk_callable
from modpy.optimize import prepare_bounds
from modpy.stats._stat_util import MCMCResult, MCMCPath


TERMINATION_MESSAGES = {
    0: 'Solution contains NaN or Inf results.',
    1: 'Sampling terminated successfully.',
}


def metropolis_hastings(x0, proposal, log_like, log_prior, samples, burn=None, bounds=None, seed=None, keep_path=False):
    """
    Samples values from a posterior distribution using the Metropolis-Hastings algorithm.

    TODO: I guess this is actually a Metropolis algorithm only, as it does not correct for asymmetric proposals.
    TODO: Introduce the asymmetric proposal correction.

    Parameters
    ----------
    x0 : array_like, shape (m,)
        Starting point for the Markov chain.
    proposal : callable
        Function that returns a proposal for the new step in the chain. Should take the current step as input:
            x_{i+1} = proposal(x_i)
        where x_{i+1} is a vector of the same shape as x_i.
    log_like : callable
        Function that returns the log-likelihood of a given observation, x:
            L = log_like(x)
        where L is a float.
    log_prior : callable
        A function that returns the logarithm of the prior probability of a given observation, x:
            p = log_prior(x)
        where p is a float.
    samples : int
        The number of samples in the Markov Chain.
    burn : int
        Number of samples to discard as burn-in period.
    bounds : 2-tuple of array_like, optional
        Bounds on the solution vector, should be (array_like (n,), array_like (n,)). If None no bounds are applied.
    seed : int
        Seed of the random number generator.
    keep_path : bool, optional
        Whether to save path information. Can require substantial memory.

    Returns
    -------
    xp : array_like, shape (samples, m)
        Samples from the posterior distribution.
    """

    # test input functions
    x0 = np.array(x0)
    _chk_callable(x0, proposal)
    _chk_callable(x0, log_like)
    _chk_callable(x0, log_prior)

    m = x0.size

    # prepare bounds
    lb, ub = prepare_bounds(bounds, m)

    # set generator
    generator = Generator(PCG64(seed))

    # sample from the posterior distribution
    xp, fp, status, path = _mh(x0, proposal, log_like, log_prior, samples, lb, ub,
                               burn=burn, generator=generator, keep_path=keep_path)

    # collect results
    res = MCMCResult(xp, fp, samples, burn, status=status, nit=samples)
    res.success = status > 0
    res.message = TERMINATION_MESSAGES[res.status]
    res.path = path

    return res


def _mh(x0, proposal, log_like, log_prior, samples, lb, ub, burn=None, generator=None, keep_path=False):
    """
    Samples values from a posterior distribution using the Metropolis-Hastings algorithm.

    TODO: I guess this is actually a Metropolis algorithm only, as it does not correct for asymmetric proposals.
    TODO: Introduce the asymmetric proposal correction.

    Parameters
    ----------
    x0 : array_like, shape (m,)
        Starting point for the Markov chain.
    proposal : callable
        Function that returns a proposal for the new step in the chain. Should take the current step as input:
            x_{i+1} = proposal(x_i)
        where x_{i+1} is a vector of the same shape as x_i.
    log_like : callable
        Function that returns the log-likelihood of a given observation, x:
            L = log_like(x)
        where L is a float.
    log_prior : callable
        A function that returns the logarithm of the prior probability of a given observation, x:
            p = log_prior(x)
        where p is a float.
    samples : int
        The number of samples in the Markov Chain.
    burn : int
        Number of samples to discard as burn-in period.
    lb : array_like, shape (m,)
            Lower bound.
    ub : array_like, shape (m,)
        Upper bound.
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

    # set generator
    if generator is None:
        generator = Generator(PCG64(None))

    m = x0.size
    path = MCMCPath(keep=keep_path)

    if burn is None:
        burn = int(0.2 * m)

    # initialize -------------------------------------------------------------------------------------------------------
    xp = np.zeros((samples + burn, m))
    fp = np.zeros((samples + burn,))
    xp[0, :] = x0
    ppc = log_prior(x0)                 # logarithm of the prior probability of the current step
    llc = log_like(x0)                  # log-likelihood of current step
    lpc = np.minimum(0., ppc + llc)     # logarithm of the probability of the current step
    accept = 0                          # track acceptance probability

    for i in range(1, samples + burn):

        # make proposal
        x = _propose_step(xp[i - 1, :], proposal, lb, ub)

        # calculate log-prior probability and log-likelihood of the proposal
        ppp = log_prior(x)
        Lp = log_like(x)

        # calculate the probability of the proposed step
        lpp = np.minimum(0., ppp + Lp)

        if np.log(generator.random()) < (lpp - lpc):
            xp[i, :] = x
            fp[i] = lpp
            accept += 1
        else:
            xp[i, :] = xp[i - 1, :]
            fp[i] = fp[i - 1]

        lpc = lpp

        # save acceptance rate
        path.append(accept / i)

    xp = xp[burn:, :]
    fp = fp[burn:]
    status = np.any(np.isnan(xp) | np.isinf(xp))

    # post-process path variables
    if keep_path:
        path.finalize()

    return xp, fp, status, path


def _propose_step(xi, proposal, lb, ub):
    """
    Samples values from an unknown distribution using the Metropolis-Hastings algorithm.

    Parameters
    ----------
    xi : array_like, shape (m,)
        Current iterate in a Markov chain.
    proposal : callable
        Function that returns a proposal for the new step in the chain. Should take the current step as input:
            x_{i+1} = proposal(x_i)
        where x_{i+1} is a vector of the same shape as x_i.
    lb : array_like, shape (m,)
            Lower bound.
    ub : array_like, shape (m,)
        Upper bound.

    Returns
    -------
    x : array_like, shape (m,)
        The proposed new iterate in the Markov chain.
    """

    x = proposal(xi)
    return np.clip(x, lb, ub)
