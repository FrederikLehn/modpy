import numpy as np
from numpy.testing import assert_, assert_raises, assert_array_almost_equal

from modpy.tests.test_util import run_unit_test
from modpy.optimize import linprog


class TestLinProg:
    def test_feasible_system_result(self):
        g = np.array([-5., -4., -6.])

        C = np.array([(1., -1., 1.),
                      (3., 2., 4.),
                      (3., 2., 0.)], dtype=np.float64)

        d = np.array([20., 42., 30.], dtype=np.float64)

        bounds = ((0., np.inf), (0., np.inf), (0., np.inf))

        res = linprog(g, C=C, d=d, bounds=bounds)
        x_true = np.array([0., 15., 3.], np.float64)

        if not res.success:
            raise ValueError(res.message)

        assert_array_almost_equal(res.x, x_true, decimal=6)

    def test_inequality_result(self):
        g = np.array([-1., -1. / 3.])

        C = np.array([(1., 1.),
                      (1., 1. / 4.),
                      (1., -1.),
                      (-1. / 4., -1.),
                      (-1., -1.),
                      (-1., 1.)])

        d = np.array([2., 1., 2., 1., -1., 2.], dtype=np.float64)

        res = linprog(g, C=C, d=d)
        x_true = np.array([2. / 3., 4. / 3.], np.float64)

        if not res.success:
            raise ValueError(res.message)

        assert_array_almost_equal(res.x, x_true, decimal=6)

    def test_all_constraints_result(self):
        g = np.array([-1., -1. / 3.], dtype=np.float64)
        A = np.atleast_2d(np.array([1., 1. / 4.]))
        b = np.array([1. / 2.])

        C = np.array([(1., 1.),
                      (1., 1. / 4.),
                      (1., -1.),
                      (-1. / 4., -1.),
                      (-1., -1.),
                      (-1., 1.)])

        d = np.array([2., 1., 2., 1., -1., 2.], dtype=np.float64)
        bounds = ((-1., 1.5), (-0.5, 1.25))

        res = linprog(g, A, b, C, d, bounds=bounds)
        x_true = np.array([0.1875, 1.25], np.float64)

        if not res.success:
            raise ValueError(res.message)

        assert_array_almost_equal(res.x, x_true, decimal=6)

    def test_equality_inequality_result(self):
        g = np.array([-1., -1. / 3.], dtype=np.float64)
        A = np.atleast_2d(np.array([1., 1. / 4.]))
        b = np.array([0.5])

        C = np.array([(1., 1.),
                      (1., 1. / 4.),
                      (1., -1.),
                      (-1. / 4., -1.),
                      (-1., -1.),
                      (-1., 1.)])

        d = np.array([2., 1., 2., 1., -1., 2.], dtype=np.float64)

        res = linprog(g, A, b, C, d)
        x_true = np.array([.0, 2.], np.float64)

        if not res.success:
            raise ValueError(res.message)

        assert_array_almost_equal(res.x, x_true, decimal=6)


if __name__ == '__main__':
    run_unit_test(TestLinProg)
