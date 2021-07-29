import numpy as np
import matplotlib.pyplot as plt

from modpy.optimize import cma_es
from modpy.plot.plot_util import cm_parula
from modpy.illustration.illustration_util import OPTIMIZE_PATH, test_function_2d

from modpy.optimize._cma_es import CMAES


if __name__ == '__main__':
    m = 100
    x_ = np.linspace(-5., 5., m)
    y_ = np.linspace(-5., 5., m)
    X, Y = np.meshgrid(x_, y_)
    P = [X.flatten(), Y.flatten()]

    #x0 = np.reshape(np.array([-1., 8.]), (2, 1))

    x0 = np.random.randn(2)
    #res = cma_es(test_function_2d, 2, method=method, sigma0=2.)
    opt = CMAES(test_function_2d, x0, method='mu-lam', seed=3)
    opt.run()
    res = opt.get_result()

    print(res.x)
    print('STATUS', res.status)

    # plot white color outside min/max of contour range
    cmap = cm_parula
    cm_parula.set_under('w')
    cm_parula.set_over('w')

    fig = plt.figure()
    ax = fig.gca()
    obj = test_function_2d(P)
    levels = np.block([np.arange(0., 20., 2.5), np.arange(20., 100., 5.), np.arange(100., 200., 10.)])
    cont = ax.contourf(X, Y, np.reshape(obj, (m, m)), levels, cmap=cmap, vmin=0., vmax=200.)
    plt.colorbar(cont)

    # add optimum
    ax.scatter(res.x[0], res.x[1], 20, zorder=1, c='r')

    fig.savefig(OPTIMIZE_PATH + 'cma_es.png')
