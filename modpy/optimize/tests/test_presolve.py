import numpy as np
from numpy.testing import assert_, assert_raises, assert_array_almost_equal, assert_array_equal

from modpy.tests.test_util import run_unit_test
from modpy.optimize._presolve import Presolver
from modpy._exceptions import InfeasibleLPError


class TestPresolve:
    # TODO: these routines will fail if scaling happens.

    def test_zero_row(self):
        n = 2

        g_ = np.array([-5., 0.])

        A_ = np.array([(0., 0.),
                       (1., 1.),
                       (1., -1.)], dtype=np.float64)

        C_ = np.zeros((0, n))
        d_ = np.zeros((0,))

        lb_ = np.array([0., -np.inf], dtype=np.float64)
        ub_ = np.array([np.inf, np.inf], dtype=np.float64)

        # test feasible ------------------------------------------------------------------------------------------------
        b_ = np.array([0., 42., 30.], dtype=np.float64)

        LP = Presolver(g_, A_, b_, C_, d_, lb_, ub_)
        LP.presolve_LP()

        assert_array_equal(LP.A, A_[1:, :])
        assert_array_equal(LP.b, b_[1:])

        # test infeasible ----------------------------------------------------------------------------------------------
        b_ = np.array([1., 42., 30.], dtype=np.float64)

        LP = Presolver(g_, A_, b_, C_, d_, lb=lb_, ub=ub_)

        assert_raises(InfeasibleLPError, LP.presolve_LP)

    def test_zero_column(self):
        n = 3
        A_ = np.array([(1., 0., 1.),
                       (1., 0., 1.),
                       (1., 0., 1.)], dtype=np.float64)

        b_ = np.array([9., 9., 9.], dtype=np.float64)
        C_ = np.zeros((0, n))
        d_ = np.zeros((0,))

        # test feasible ------------------------------------------------------------------------------------------------
        lb_ = np.array([0., 0., 0.], dtype=np.float64)
        ub_ = np.array([5., 5., 5.], dtype=np.float64)

        # test c_j = 0, j=1
        g_ = np.array([-5., 0., -1])
        LP = Presolver(g_, A_, b_, C_, d_, lb_, ub_)
        LP.presolve_LP()

        idx = LP.idx_x
        assert_array_equal(idx, [0, 2])
        assert_array_equal(LP.g, g_[idx])
        assert_array_equal(LP.A, A_[:, idx])
        assert_array_equal(LP.lb, lb_[idx])
        assert_array_equal(LP.ub, ub_[idx])
        assert LP.x[1] == (LP.lb[1] + LP.ub[1]) / 2.

        # test c_j > 0, j=1
        g_ = np.array([-5., 10., -1])
        LP = Presolver(g_, A_, b_, C_, d_, lb_, ub_)
        LP.presolve_LP()

        idx = LP.idx_x
        assert_array_equal(idx, [0, 2])
        assert_array_equal(LP.g, g_[idx])
        assert_array_equal(LP.A, A_[:, idx])
        assert_array_equal(LP.lb, lb_[idx])
        assert_array_equal(LP.ub, ub_[idx])
        assert LP.x[1] == LP.lb[1]

        # test c_j < 0, j=1
        g_ = np.array([-5., -10., -1])
        LP = Presolver(g_, A_, b_, C_, d_, lb_, ub_)
        LP.presolve_LP()

        idx = LP.idx_x
        assert_array_equal(idx, [0, 2])
        assert_array_equal(LP.g, g_[idx])
        assert_array_equal(LP.A, A_[:, idx])
        assert_array_equal(LP.lb, lb_[idx])
        assert_array_equal(LP.ub, ub_[idx])
        assert LP.x[1] == LP.ub[1]

        # test infeasible ----------------------------------------------------------------------------------------------
        lb_ = np.array([0., -np.inf, 0.], dtype=np.float64)
        ub_ = np.array([5., np.inf, 5.], dtype=np.float64)

        # test c_j > 0, j=1
        g_ = np.array([-5., 10., -1])
        LP = Presolver(g_, A_, b_, C_, d_, lb_, ub_)

        assert_raises(InfeasibleLPError, LP.presolve_LP)

        # test c_j < 0, j=1
        g_ = np.array([-5., -10., -1])
        LP = Presolver(g_, A_, b_, C_, d_, lb_, ub_)

        assert_raises(InfeasibleLPError, LP.presolve_LP)

    def test_forcing_constraint(self):
        n = 3
        A_ = np.array([(1., 0., 1.),
                       (1., 0., 1.),
                       (1., 0., 1.)], dtype=np.float64)

        b_ = np.array([0., 42., 30.], dtype=np.float64)
        C_ = np.zeros((0, n))
        d_ = np.zeros((0,))

        # test infeasible ----------------------------------------------------------------------------------------------
        lb_ = np.array([0., 0., 0.], dtype=np.float64)
        ub_ = np.array([5., 5., 5.], dtype=np.float64)

        # test c_j = 0, j=1
        g_ = np.array([-5., 0., -1])
        LP = Presolver(g_, A_, b_, C_, d_, lb_, ub_)
        assert_raises(InfeasibleLPError, LP.presolve_LP)

    def test_fixed_variable(self):
        n = 3
        g_ = np.array([-5., 10., -1])

        A_ = np.zeros((0, n))
        b_ = np.zeros((0,))
        C_ = np.array([(1., 1., 1.),
                       (1., 2., 1.),
                       (1., 1., 3.)], dtype=np.float64)

        d_ = np.array([5., 42., 30.], dtype=np.float64)
        lb_ = np.array([0., 5., 0.], dtype=np.float64)
        ub_ = np.array([5., 5., 5.], dtype=np.float64)

        LP = Presolver(g_, A_, b_, C_, d_, lb_, ub_)
        LP.presolve_LP()

        idx = LP.idx_x
        assert_array_equal(idx, [0, 2])
        assert_array_equal(LP.g, g_[idx])
        assert_array_equal(LP.A, A_[:, idx])
        assert_array_equal(LP.b, b_ - A_[:, 1] * lb_[1])
        assert LP.k == g_[1] * lb_[1]

    def test_singleton_rows_eq(self):
        n = 2

        g_ = np.array([-5., 0.])

        A_ = np.array([(0., 1.),
                       (1., 0.),
                       (1., -1.)], dtype=np.float64)

        b_ = np.array([0., 42., 30.], dtype=np.float64)
        C_ = np.zeros((0, n))
        d_ = np.zeros((0,))

        lb_ = np.array([0., -np.inf], dtype=np.float64)

        # test feasible ------------------------------------------------------------------------------------------------
        ub_ = np.array([np.inf, np.inf], dtype=np.float64)

        LP = Presolver(g_, A_, b_, C_, d_, lb_, ub_)
        LP.presolve_LP()

        assert_array_equal(LP.A, np.atleast_2d(A_[LP.idx_e, LP.idx_x]))
        assert_array_equal(LP.b, b_[LP.idx_e])

        # test infeasible ----------------------------------------------------------------------------------------------
        ub_ = np.array([np.inf, 1.], dtype=np.float64)
        LP = Presolver(g_, A_, b_, C_, d_, lb=lb_, ub=ub_)

        assert_raises(InfeasibleLPError, LP.presolve_LP)

    def test_singleton_rows_iq(self):
        n = 2

        g_ = np.array([-5., 0.])
        A_ = np.zeros((0, n))
        b_ = np.zeros((0,))

        C_ = np.array([(0., 1.),
                       (1., 0.),
                       (1., -1.)], dtype=np.float64)

        d_ = np.array([0., 42., 30.], dtype=np.float64)

        ub_ = np.array([np.inf, np.inf], dtype=np.float64)

        # test feasible ------------------------------------------------------------------------------------------------
        lb_ = np.array([0., -np.inf], dtype=np.float64)

        LP = Presolver(g_, A_, b_, C_, d_, lb_, ub_)
        LP.presolve_LP()

        assert_array_equal(LP.C, np.atleast_2d(C_[LP.idx_i, LP.idx_x]))
        assert_array_equal(LP.d, d_[LP.idx_i])
        assert_array_equal(LP.ub, d_[[1, 0]])

        # test infeasible ----------------------------------------------------------------------------------------------
        lb_ = np.array([50., -np.inf], dtype=np.float64)
        LP = Presolver(g_, A_, b_, C_, d_, lb=lb_, ub=ub_)
        assert_raises(InfeasibleLPError, LP.presolve_LP)

    def test_duplicate_rows(self):
        g_ = np.array([-1., -1. / 3.], dtype=np.float64)
        A_ = np.atleast_2d(np.array([1., 1. / 4.]))
        b_ = np.array([0.5])

        C_ = np.array([(1., 1.),
                       (1., 1. / 4.),
                       (1., -1.),
                       (-1. / 4., -1.),
                       (-1., -1.),
                       (-1., 1.)])

        lb_ = np.array([-1., -0.5])
        ub_ = np.array([1.5, 1.25])

        # test feasible ------------------------------------------------------------------------------------------------
        d_ = np.array([2., 1., 2., 1., -2., -2.], dtype=np.float64)
        LP = Presolver(g_, A_, b_, C_, d_, lb_, ub_)
        LP.presolve_LP()

        idx = LP.idx_i
        assert_array_equal(LP.C, C_[idx])

        # test infeasible ----------------------------------------------------------------------------------------------
        d_ = np.array([2., 1., 2., 1., -1., 2.], dtype=np.float64)

        LP = Presolver(g_, C_, d_, A_, b_, lb_, ub_)  # notice A/C and b/d switched on purpose
        assert_raises(InfeasibleLPError, LP.presolve_LP)


if __name__ == '__main__':
    run_unit_test(TestPresolve)
