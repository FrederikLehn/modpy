import numpy as np
from numpy.random import Generator, PCG64


def k_fold(n, k, seed=None):
    """
    Splits a set of `n` observations into `k` sub-sets and provides `k` training and test sets.

    Parameters
    ----------
    n : int
        Size of the set to be sub-divided.
    k : int
        Number of sub-sets.
    seed : int
        Seed of the random number generator during shuffling of the sets.

    Returns
    -------
    sets : tuple
        Tuple of `k` tuples with (train, test).
    """

    # sub-divide set into `k` sets
    indices = _divide_set(n, k, seed=seed)
    sets = [_split_train_test(i, indices) for i in range(k)]

    return sets


def _split_train_test(i, indices):
    """
    Construct `k` unique index vectors that subdivides a set of size `n`.

    Parameters
    ----------
    i : int
        The i'th set to use for testing.
    indices : tuple
        Tuple of arrays with indices.

    Returns
    -------
    train : array_like, shape (n - n / k,)
        Indices to the training set
    test : array_like, shape (n / k,)
        Indices to the test test
    """

    test = indices[i]
    train = np.hstack([idx for (j, idx) in enumerate(indices) if j != i])

    return train, test


def _divide_set(n, k, seed=None):
    """
    Construct `k` unique index vectors that subdivides a set of size `n`.

    Parameters
    ----------
    n : int
        Size of the set to be sub-divided.
    k : int
        Number of sub-sets.
    seed : int
        Seed of the random number generator during shuffling of the indices.

    Returns
    -------
    indices : tuple
        Tuple of `k` arrays with indices.
    """

    if k > n:
        raise ValueError('Number of sub-sets `k` must be less than or equal to the total set length `n`.')

    # special case: leave-one-out validation
    if k == n:
        return tuple([np.array([i]) for i in range(k)])

    # determine sample size of the sub-sets
    sample = int(n / k)
    sample_end = int(np.mod(n, sample))

    # prepare random generator
    gen = Generator(PCG64(seed))

    # create permuted index array
    idx = gen.permutation(np.arange(n))

    # sub-divide
    indices = [idx[(sample*i):(sample*(i+1))] for i in range(k)]

    if sample_end > 0:
        indices.append(idx[(sample * k):])

    return indices
