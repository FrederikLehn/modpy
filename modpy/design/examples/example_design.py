import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MaxNLocator

from modpy.random import *
from modpy.design import *
from modpy.illustration.illustration_util import DESIGN_PATH


if __name__ == '__main__':

    n_sfd = 20

    dists = JointDistribution((NormalDist(3., 1., bounds=(0., 6.)),
                               TriangularDist(1., 6., 2.5),
                               BetaDist(2., 5., 5., 12.)))

    designs = (mono_sensitivity_design(dists, scale=False),
               full_factorial_design(dists, level=2, scale=False),
               ccf_design(dists, scale=False),
               latin_hypercube_design(dists, samples=n_sfd, scale=False),
               kennard_stone_design(dists, n_sfd, 100, scale=False),
               wsp_design(dists, n_sfd, 100, scale=False, method='iterative'))

    names = ('mono', 'full_factorial', 'ccf', 'latin', 'kennard_stone', 'wsp')

    points = np.array([[-1, -1, -1],
                       [1, -1, -1],
                       [1, 1, -1],
                       [-1, 1, -1],
                       [-1, -1, 1],
                       [1, -1, 1],
                       [1, 1, 1],
                       [-1, 1, 1]])

    for i, (design, name) in enumerate(zip(designs, names)):

        # plot cube
        fig = plt.figure()
        fig.subplots_adjust(left=-0.1, right=1.1, bottom=-0.1, top=1.1)
        ax = fig.add_subplot(111, projection='3d')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.zaxis.set_major_locator(MaxNLocator(integer=True))

        r = [-1, 1]
        X, Y = np.meshgrid(r, r)
        Z = np.ones_like(X)
        ax.plot_surface(X, Y, Z, alpha=0.1, color='w', edgecolor='k')
        ax.plot_surface(X, Y, -Z, alpha=0.1, color='w', edgecolor='k')
        ax.plot_surface(X, -Z, Y, alpha=0.1, color='w', edgecolor='k')
        ax.plot_surface(X, Z, Y, alpha=0.1, color='w', edgecolor='k')
        ax.plot_surface(Z, X, Y, alpha=0.1, color='w', edgecolor='k')
        ax.plot_surface(-Z, X, Y, alpha=0.1, color='w', edgecolor='k')
        #ax.scatter3D(points[:, 0], points[:, 1], points[:, 2])

        # plot design
        ax.scatter3D(design[:, 0], design[:, 1], design[:, 2], color='r')

        # plot arrows from reference to points

        ax.axis('off')

        #ax.set_xlabel('$X_1$')
        #ax.set_ylabel('$X_2$')
        #ax.set_zlabel('$X_3$')

        fig.savefig(DESIGN_PATH + 'designs_{}.png'.format(name), bbox='tight')
