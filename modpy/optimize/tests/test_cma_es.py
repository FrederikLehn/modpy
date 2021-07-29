import numpy as np
from numpy.testing import assert_, assert_raises, assert_array_almost_equal

from modpy.tests.test_util import run_unit_test
from modpy.optimize import cma_es
from modpy.optimize import Bounds
from modpy.optimize.benchmarks.benchmark_suite import Sphere, Ellipsoid, Cigar


DIM = 2
SEED = 1234


class TestCMAES:

    _method = None

    def test_sphere(self):
        test = Sphere(dim=DIM)
        x0 = np.ones((DIM,)) * 10.

        res = cma_es(test.obj, x0=x0, method=self._method, tol=1e-10, seed=SEED)

        if not res.success:
            raise ValueError(res.message)

        assert_array_almost_equal(res.x, test.x_opt, decimal=4)

    def test_ellipsoid(self):
        test = Ellipsoid(dim=DIM)
        x0 = np.ones((DIM,)) * 10.

        res = cma_es(test.obj, x0=x0, method=self._method, tol=1e-10, seed=SEED)

        if not res.success:
            raise ValueError(res.message)

        assert_array_almost_equal(res.x, test.x_opt, decimal=4)

    def test_cigar(self):
        test = Cigar(dim=DIM)
        x0 = np.ones((DIM,)) * 10.

        res = cma_es(test.obj, x0=x0, method=self._method, tol=1e-10, seed=SEED)

        if not res.success:
            raise ValueError(res.message)

        assert_array_almost_equal(res.x, test.x_opt, decimal=4)


class TestMuLam(TestCMAES):
    _method = 'mu-lam'


class TestIPOP(TestCMAES):
    _method = 'IPOP'


if __name__ == '__main__':
    run_unit_test(TestMuLam)
    #run_unit_test(TestIPOP)
