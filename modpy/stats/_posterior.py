import numpy as np

from modpy.design import ksd


def posterior_ensemble(mcmc, samples, alpha=0.1, thresh=0.01, thin=True):
    """
    Select a sub-sample of representative samples from a Markov chain to serve as a posterior ensemble.

    Parameters
    ----------
    mcmc : MCMCResult
        Result from an MCMC algorithm.
    samples : int
        Number of sub-samples to select for the posterior ensemble.
    alpha : float
        Quartile threshold of posterior distribution in range [0, 1].
    thresh : float
        Threshold below which probabilities are assumed 0. This is compounded by the dimensions, so that only:
            log(p) >= dim * log(thresh)
        are included in the quartile calculation
    thin : bool
        Whether or not thin the MCMC chain.

    Returns
    -------
    xp : array_like, shape (n, m)
        Representative samples from the posterior distribution.
    fp : array_like, shape (n,)
        Log-probability of the posterior ensemble.
    """

    if not (0. < alpha < 1.):
        raise ValueError('`alpha` must be between 0 and 1.')

    if thresh > 1:
        raise ValueError('`thresh` must be less than or equal to 1.')

    # get chain and posterior probability
    if thin:
        x, f = mcmc.get_thinned()
    else:
        x, f = mcmc.x, mcmc.f

    # remove samples which can be assumed to be of 0 probability
    n, m = x.shape
    mask = f >= (np.log(thresh) * m)

    x = x[mask, :]
    f = f[mask]

    # remove samples below the alpha quartile
    p = np.percentile(f, alpha * 100)
    mask = f >= p
    x = x[mask, :]
    f = f[mask]

    # sub-sample using space-filling design
    xp, idx = ksd(x, samples)

    return xp, f[idx]
