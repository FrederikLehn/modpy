import numpy as np

from modpy.special._special_util import EXP


def log(x, base=EXP):
    if base <= 1:
        raise ValueError('`base` must be larger than 1.')

    logx = np.where(x > 0., np.log(x) / np.log(base), -np.inf)

    if isinstance(x, float):
        return float(logx)
    else:
        return logx
