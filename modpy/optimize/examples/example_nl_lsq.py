import numpy as np
import matplotlib.pyplot as plt

from modpy.optimize import least_squares
from modpy.plot.plot_util import default_color
from modpy.illustration.illustration_util import OPTIMIZE_PATH, test_fit_data


if __name__ == '__main__':
    x, y = test_fit_data(15)

    def f_wrapped(p, args=()):
        return p[0] * np.exp(p[1] * x)

    def fun_wrapped(p, args=()):
        return y - f_wrapped(p, *args)

    x0 = (np.array([1., -0.1]))
    res2 = least_squares(fun_wrapped, x0, jac='2-point')
    res3 = least_squares(fun_wrapped, x0, jac='3-point')

    fig = plt.figure()
    ax = fig.gca()
    ax.scatter(x, y, 20, zorder=1, c='k')
    ax.plot(x, 1.3 * np.exp(-0.7 * x), c=default_color(0), label='True (1.3 exp(-0.7 x))')
    ax.plot(x, f_wrapped(res2.x), c=default_color(1), label='J (2-point) ({} exp({} x))'.format(*np.round(res2.x, 2)))
    ax.plot(x, f_wrapped(res3.x), c=default_color(2), label='J (3-point) ({} exp({} x))'.format(*np.round(res3.x, 2)))

    ax.grid(True)
    ax.legend()

    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')

    fig.savefig(OPTIMIZE_PATH + 'nl_lsq.png')
