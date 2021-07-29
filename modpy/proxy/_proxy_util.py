import numpy as np


def _ensure_matrix(x):
    """
    Ensures the vector/matrix `x` is in matrix format.

    Parameters
    ----------
    x : array_like, shape (n,)
        Vector or matrix.

    Returns
    -------
    x : array_like, shape (m, p)
        Matrix.
    """

    x = np.array(x)

    if x.ndim == 1:
        x = np.reshape(x, (x.size, 1))
    elif x.ndim > 2:
        raise ValueError('`x` must be of dimension 1 or 2.')

    return x