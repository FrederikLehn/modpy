import numpy as np


def factorial(x):
    if not isinstance(x, int):
        raise ValueError('`x` must be an integer.')

    return np.prod(range(1, x + 1))
