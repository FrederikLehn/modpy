import numpy as np
from numpy.testing import assert_, assert_raises, assert_array_almost_equal

from modpy.tests.test_util import run_unit_test
from modpy.optimize import quadprog


class TestQuadProg:
    def test_unconstrained_result(self):
        H = np.array([(6., 2., 1.),
                      (2., 5., 2.),
                      (1., 2., 4.)])
        g = np.zeros((3,))

        res = quadprog(H, g)
        x_true = np.array([0., 0., 0.], np.float64)

        if not res.success:
            raise ValueError(res.message)

        assert_array_almost_equal(res.x, x_true, decimal=15)

    def test_equality_result(self):
        H = np.array([(6., 2., 1.),
                      (2., 5., 2.),
                      (1., 2., 4.)])

        g = np.array([-8, -3, -3], dtype=np.float64)

        A = np.array([(1., 0., 1.,),
                      (0., 1., 1.)])

        b = np.array([3, 0], dtype=np.float64)

        res = quadprog(H, g, A, b)
        x_true = np.array([2, -1, 1], np.float64)

        if not res.success:
            raise ValueError(res.message)

        assert_array_almost_equal(res.x, x_true, decimal=15)

    def test_inequality_result(self):
        H = np.array([(2., 0.),
                      (0., 2.)])

        g = np.array([-2, -5], dtype=np.float64)

        C = np.array([(1., -2.),
                      (-1., -2.),
                      (-1., 2.),
                      (1., 0.),
                      (0., 1.)])

        d = np.array([-2, -6, -2, 0, 0], dtype=np.float64)

        res = quadprog(H, g, None, None, C, d)
        x_true = np.array([1.4, 1.7], np.float64)

        if not res.success:
            raise ValueError(res.message)

        assert_array_almost_equal(res.x, x_true, decimal=6)

    def test_constrained_result(self):
        # from: https://se.mathworks.com/help/optim/ug/quadprog.html

        H = np.array([(1., -1., 1.),
                      (-1., 2., -2.),
                      (1., -2., 4.)])

        g = np.array([2, -3, 1], dtype=np.float64)
        A = np.ones((1, 3))
        b = np.array([0.5], dtype=np.float64)
        C = np.vstack((np.eye(3), -np.eye(3)))
        d = np.array([0, 0, 0, -1, -1, -1], dtype=np.float64)

        res = quadprog(H, g, A, b, C, d)
        x_true = np.array([0., .5, 0.], np.float64)

        if not res.success:
            raise ValueError(res.message)

        assert_array_almost_equal(res.x, x_true, decimal=6)

    def test_bounded(self):

        H = np.array([(1., -1., 1.),
                      (-1., 2., -2.),
                      (1., -2., 4.)])

        g = np.array([2, -3, 1], dtype=np.float64)
        bounds = ((None, 1.), (0., 1.), (0., 1.))

        res = quadprog(H, g, bounds=bounds)
        x_true = np.array([-5/3, 1, 2/3], dtype=np.float64)

        if not res.success:
            raise ValueError(res.message)

        assert_array_almost_equal(res.x, x_true, decimal=6)


if __name__ == '__main__':
    run_unit_test(TestQuadProg)
