import numpy as np
import matplotlib.pyplot as plt

from modpy.optimize import bisection_scalar, secant_scalar, newton_scalar
from modpy.plot.plot_util import default_color, set_font_sizes
from modpy.illustration.illustration_util import OPTIMIZE_PATH, test_root_scalar_function


def plot_root_result():
    x = np.linspace(-2., 2., 100)
    y = test_root_scalar_function(x)

    res1 = bisection_scalar(test_root_scalar_function, -2., 2.)
    res2 = secant_scalar(test_root_scalar_function, -.5, .5)

    fig = plt.figure()
    ax = fig.gca()

    xlim = (x.min(), x.max())
    ylim = (y.min(), y.max())

    ax.plot(x, y, c=default_color(0), label='$f(x)$')
    ax.plot(xlim, (0., 0.), 'k--')
    ax.plot((0., 0.), ylim, 'k--')

    if res1.success:
        ax.plot((res1.x, res1.x), ylim, c=default_color(1), label='bisection')

    if res2.success:
        ax.plot((res2.x, res2.x), ylim, c=default_color(2), label='secant')

    ax.grid(True)
    ax.legend()

    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    fig.savefig(OPTIMIZE_PATH + 'root.png')


def plot_newton_path():
    n = 100

    x_min = 0.
    x_max = 10.
    x_mid = (x_min + x_max) / 2.

    multiplier = 1.
    y_min = multiplier * x_min
    y_max = multiplier * x_max
    y_mid = multiplier * x_mid

    x = np.linspace(x_min, x_max, n)
    a = 1.
    b = -10.
    c = 24.2 #26.
    y = a * x ** 2. + b * x + c

    p_y = 10.
    D = b ** 2. - 4. * a * (c - p_y)
    p1_x = 3.  # -b + np.sqrt(D) / (2. * a)
    p2_x = 7.  # -b - np.sqrt(D) / (2. * a)

    head_length = (x_max - x_min) * 0.03
    head_width = (x_max - x_min) * 0.03

    # define functions
    def f(x_):
        return a * x_ ** 2. + b * x_ + c

    def df(x_):
        return 2. * a * x_ + b

    # solve newton far-away
    x0_1 = 9.
    x0_2 = 6.8
    res1 = newton_scalar(f, df, x0_1, keep_path=True)
    res2 = newton_scalar(f, df, x0_2, keep_path=True)

    print(res1.x, res2.x)

    # infill the result to show up/down on y-axis
    n1 = len(res1.path.xi) * 2
    n2 = len(res2.path.xi) * 2

    x1 = np.zeros((n1,))
    x2 = np.zeros((n2,))
    y1 = np.zeros((n1,))
    y2 = np.zeros((n2,))

    x1[:-1:2] = res1.path.xi
    x1[1::2] = res1.path.xi
    y1[1::2] = res1.path.fi

    x2[:-1:2] = res2.path.xi
    x2[1::2] = res2.path.xi
    y2[1::2] = res2.path.fi

    # plot newton steps ------------------------------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(30, 14))
    ax1, ax2 = axes.flatten()

    ax1.plot(x, y, c=default_color(0), lw=3., label='Residual', zorder=3)
    ax1.plot(x1, y1, 'o-', c=default_color(1), lw=3., ms=10, label='Previous time-step', zorder=4)
    ax1.plot(x2, y2, 'o-', c=default_color(2), lw=3., ms=10, label='Improved start-guess', zorder=5)
    ax1.plot([x_min, x_max], [0., 0.], 'k--', zorder=2)
    #ax1.arrow(x0_1, 0., -(x0_1 - x0_2 - head_length), 0., head_width=head_width, head_length=head_length, lw=3.,
    #          fc=default_color(3), ec=default_color(3), zorder=6)

    # configure figure
    ax1.grid(True)
    ax1.legend(loc='upper center')

    # set labels
    ax1.set_title('Newton Iterations')
    ax1.set_xlabel('$x$')
    ax1.set_ylabel('Linear System Residual, $r=Ax-b$')
    ax1.set_xlim([x_min, x_max])
    #ax.set_ylim([y_min, y_max])
    set_font_sizes(ax1, 25)

    # plot error -------------------------------------------------------------------------------------------------------
    ax2.semilogy(np.arange(int(n1 / 2) - 1), res1.path.rtol[:], 'o-', c=default_color(1), lw=3., ms=10, label='Previous time-step', zorder=4)
    ax2.semilogy(np.arange(int(n2 / 2) - 1), res2.path.rtol[:], 'o-', c=default_color(2), lw=3., ms=10, label='Improved start-guess', zorder=5)

    # configure figure
    ax2.grid(True)
    ax2.legend(loc='upper right')

    # set labels
    ax2.set_title('Error Reduction')
    ax2.set_xlabel('Iterations')
    ax2.set_ylabel('Relative Error')
    ax2.set_xlim([x_min, x_max])
    # ax.set_ylim([y_min, y_max])
    set_font_sizes(ax2, 25)

    fig.savefig(OPTIMIZE_PATH + 'newton_solver_path.png')


if __name__ == '__main__':
    #plot_root_result()
    plot_newton_path()

