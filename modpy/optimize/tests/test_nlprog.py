import numpy as np
from numpy.testing import assert_, assert_raises, assert_array_almost_equal

from modpy.tests.test_util import run_unit_test
from modpy.optimize import nlprog
from modpy.optimize import Bounds, LinearConstraint, NonlinearConstraint
from modpy.optimize.benchmarks.benchmark_suite import TR2, G01, G03, G04, G05, G06, G07, G09, G11,\
    g06, g06_nl_con, g06_nl_lb, g06_nl_ub


def rosenbrock(x):
    return np.sum(100. * (x[:-1] ** 2. - x[1:]) ** 2. + (x[:-1] - 1.) ** 2.)


def fun_eq(x):
    return np.exp(np.prod(x)) - .5 * (x[0] ** 3. + x[1] ** 3. + 1.) ** 2.


def ce_eq(x):
    return np.array([np.sum(x ** 2.) - 10., x[1] * x[2] - 5. * x[3] * x[4], x[0] ** 3. + x[1] ** 3. + 1.], dtype=np.float64)


def jac_f_eq(x):
    ex = np.exp(np.prod(x))
    return np.array([x[1] * x[2] * x[3] * x[4] * ex - 3. * (x[0] ** 3. + x[1] ** 3. + 1) * x[0] ** 2.,
                     x[0] * x[2] * x[3] * x[4] * ex - 3. * (x[0] ** 3. + x[1] ** 3. + 1) * x[1] ** 2.,
                     x[0] * x[1] * x[3] * x[4] * ex,
                     x[0] * x[1] * x[2] * x[4] * ex,
                     x[0] * x[1] * x[2] * x[3] * ex], dtype=np.float64)


def jac_c_eq(x):
    return np.vstack((np.array([2. * x[0], 2. * x[1], 2. * x[2], 2. * x[3], 2. * x[4]]),
                      np.array([0., x[2], x[1], -5. * x[4], - 5 * x[3]]),
                      np.array([3. * x[0] ** 2., 3. * x[1] ** 2., 0., 0., 0.])))


def fun_iq(x):
    return (x[0] ** 2. + x[1] - 11.) ** 2. + (x[0] + x[1] ** 2. - 7.) ** 2.


def ci_iq_nl(x):
    return np.array([(x[0] + 2.) ** 2. - x[1]], dtype=np.float64)


def jac_f_iq(x):
    return np.array([4. * (x[0] ** 2. + x[1] - 11.) * x[0] + 2. * x[1] ** 2. + 2. * x[0] - 14.,
                     2. * x[0] ** 2. + 2. * x[1] - 22. + 4. * ( x[1] ** 2. + x[0] - 7) * x[1]], dtype=np.float64)


def jac_ci_iq_nl(x):
    return np.array([2. * x[0] + 4., -1.])


class TestNLProg:

    _method = None

    def test_uncon_no_jac_result(self):

        if not self._method == 'SQP':
            return

        x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2], dtype=np.float64)

        res = nlprog(rosenbrock, x0, method=self._method)
        x_true = np.array([1., 1., 1., 1., 1.], np.float64)

        if not res.success:
            raise ValueError(res.message)

        assert_array_almost_equal(res.x, x_true, decimal=6)

    def test_uncon_exact_result(self):

        if not self._method == 'SQP':
            return

        x0 = np.array([1.3, 0.7], dtype=np.float64)

        def jac_rosen(x):
            return np.array([400. * x[0] ** 3. - 400. * x[0] * x[1] + 2. * x[0] - 2.,
                             200. * (x[1] - x[0] ** 2.)])

        def hess_rosen(x):
            return np.array([(1200. * x[0] ** 2. - 400. * x[1] + 2, -400. * x[0]),
                             (-400. * x[0], 200.)])

        res = nlprog(rosenbrock, x0, method=self._method, jac=jac_rosen, hess=hess_rosen)
        x_true = np.array([1., 1.], np.float64)

        if not res.success:
            raise ValueError(res.message)

        assert_array_almost_equal(res.x, x_true, decimal=6)

    def test_equality_result(self):

        x0 = np.array([-1.8, 1.7, 1.9, -0.8, -0.8], dtype=np.float64)
        con = NonlinearConstraint(ce_eq, lb=(0., 0., 0.), jac=jac_c_eq)

        res = nlprog(fun_eq, x0, jac=jac_f_eq, constraints=con, method=self._method)
        x_true = np.array([-1.71, 1.59, 1.82, -0.763, -0.763], np.float64)

        if not res.success:
            raise ValueError(res.message)

        assert_array_almost_equal(res.x, x_true, decimal=2)

    def test_equality_no_jac_result(self):

        x0 = np.array([-1.8, 1.7, 1.9, -0.8, -0.8], dtype=np.float64)
        con = NonlinearConstraint(ce_eq, lb=(0., 0., 0.))

        res = nlprog(fun_eq, x0, constraints=con, method=self._method)
        x_true = np.array([-1.71, 1.59, 1.82, -0.763, -0.763], np.float64)

        if not res.success:
            raise ValueError(res.message)

        assert_array_almost_equal(res.x, x_true, decimal=2)

    def test_inequality_result(self):

        x0 = np.array([2., 5.], dtype=np.float64)
        C = np.array([-4., 10.], dtype=np.float64)
        con = (NonlinearConstraint(ci_iq_nl, lb=(0.,), ub=(np.inf,), jac=jac_ci_iq_nl),
               LinearConstraint(C, lb=(0.,), ub=(np.inf,)))

        res = nlprog(fun_iq, x0, jac=jac_f_iq, constraints=con, method=self._method)
        x_true = np.array([3., 2.], np.float64)

        if not res.success:
            raise ValueError(res.message)

        assert_array_almost_equal(res.x, x_true, decimal=6)

    def test_inequality_no_jac_result(self):

        x0 = np.array([2., 5.], dtype=np.float64)
        C = np.array([-4., 10.], dtype=np.float64)
        con = (NonlinearConstraint(ci_iq_nl, lb=(0.,), ub=(np.inf,)),
               LinearConstraint(C, lb=(0.,), ub=(np.inf,)))

        res = nlprog(fun_iq, x0, constraints=con, method=self._method)
        x_true = np.array([3., 2.], np.float64)

        if not res.success:
            raise ValueError(res.message)

        assert_array_almost_equal(res.x, x_true, decimal=6)

    def test_bounds_result(self):

        x0 = np.array([4., 1.], dtype=np.float64)
        bounds = ((0., 5.), (0., 5.))

        res = nlprog(fun_iq, x0, jac=jac_f_iq, bounds=bounds, method=self._method)
        x_true = np.array([3., 2.], np.float64)

        if not res.success:
            raise ValueError(res.message)

        assert_array_almost_equal(res.x, x_true, decimal=6)

    def test_equality_and_bounds_result(self):

        x0 = np.array([2., 5.], dtype=np.float64)
        A = np.array([-4., 10.], dtype=np.float64)  # equality constraint matrix
        con = LinearConstraint(A, lb=(0.,))
        bounds = ((0., 5.), (0., 5.))

        res = nlprog(fun_iq, x0, jac=jac_f_iq, bounds=bounds, constraints=con, method=self._method)
        x_true = np.array([3.21644062, 1.28657625], np.float64)

        if not res.success:
            raise ValueError(res.message)

        assert_array_almost_equal(res.x, x_true, decimal=6)

    def test_TR2(self):
        test = TR2()
        res = nlprog(test.obj, test.x_ini, bounds=test.bounds, constraints=test.constraints, method=self._method)

        if not res.success:
            raise ValueError(res.message)

        assert_array_almost_equal(res.x, test.x_opt, decimal=6)

    def test_G01_approx(self):
        test = G01()
        x0 = np.ones((13,)) * 10.

        res = nlprog(test.obj, x0, bounds=test.bounds, constraints=test.constraints, method=self._method)

        if not res.success:
            raise ValueError(res.message)

        assert_array_almost_equal(res.x, test.x_opt, decimal=4)

    def test_G03_approx(self):
        test = G03()
        x0 = np.ones((10,)) * 1.

        res = nlprog(test.obj, x0, bounds=test.bounds, constraints=test.constraints, method=self._method)

        if not res.success:
            raise ValueError(res.message)

        assert_array_almost_equal(res.x, test.x_opt, decimal=6)

    def test_G04_approx(self):
        test = G04()
        x0 = np.ones((5,)) * 10.

        res = nlprog(test.obj, x0, bounds=test.bounds, constraints=test.constraints, method=self._method)

        if not res.success:
            raise ValueError(res.message)

        assert_array_almost_equal(res.x, test.x_opt, decimal=6)

    def test_G05_approx(self):
        test = G05()
        x0 = np.array([900., 900., 0.3, 0.3])

        res = nlprog(test.obj, x0, bounds=test.bounds, constraints=test.constraints, method=self._method)

        if not res.success:
            raise ValueError(res.message)

        assert_array_almost_equal(res.x, test.x_opt, decimal=4)

    def test_G06_approx(self):
        test = G06()
        x0 = np.array([50., 75.])

        res = nlprog(test.obj, x0, bounds=test.bounds, constraints=test.constraints, method=self._method)

        if not res.success:
            raise ValueError(res.message)

        assert_array_almost_equal(res.x, test.x_opt, decimal=6)

    def test_G06_exact(self):
        test = G06()
        x0 = np.array([50., 75.])

        def g06_jac(x):
            return np.array([3. * (x[0] - 10.) ** 2., 3. * (x[1] - 20.) ** 2.], dtype=np.float64)

        def g06_hess(x):
            return np.array([(6. * (x[0] - 10.), 0.),
                             (0., 6. * (x[1] - 20.))])

        def g06_con_jac(x):
            return -np.array([(-2. * (x[0] - 5.), -2. * (x[1] - 5.)),
                              (2. * (x[0] - 6.), 2. * (x[1] - 5.))])

        def g06_con_hess(x):
            return (-np.array([(-2., 0.),
                               (0., -2.)]),
                    -np.array([(2., 0.),
                               (0., 2.)]))

        con = NonlinearConstraint(g06_nl_con, lb=g06_nl_lb, ub=g06_nl_ub, jac=g06_con_jac, hess=g06_con_hess)

        res = nlprog(test.obj, x0, jac=g06_jac, hess=g06_hess, bounds=test.bounds, constraints=con, method=self._method)

        if not res.success:
            raise ValueError(res.message)

        assert_array_almost_equal(res.x, test.x_opt, decimal=6)

    def test_G07_approx(self):
        test = G07()
        x0 = np.ones((10,)) * 3.

        res = nlprog(test.obj, x0, bounds=test.bounds, constraints=test.constraints, method=self._method)

        if not res.success:
            raise ValueError(res.message)

        assert_array_almost_equal(res.x, test.x_opt, decimal=6)

    def test_G09_approx(self):
        test = G09()
        x0 = np.ones((7,)) * 3.

        res = nlprog(test.obj, x0, bounds=test.bounds, constraints=test.constraints, method=self._method)

        if not res.success:
            raise ValueError(res.message)

        assert_array_almost_equal(res.x, test.x_opt, decimal=6)

    def test_G11_approx(self):
        test = G11()
        x0 = np.ones((2,)) * 2.

        res = nlprog(test.obj, x0, bounds=test.bounds, constraints=test.constraints, method=self._method)

        if not res.success:
            raise ValueError(res.message)

        if res.x[0] > 0:
            assert_array_almost_equal(res.x, test.x_opt[0], decimal=6)
        else:
            assert_array_almost_equal(res.x, test.x_opt[1], decimal=6)


class TestSQP(TestNLProg):
    _method = 'SQP'


class TestIP(TestNLProg):
    _method = 'IP'


if __name__ == '__main__':
    run_unit_test(TestSQP)
    run_unit_test(TestIP)
