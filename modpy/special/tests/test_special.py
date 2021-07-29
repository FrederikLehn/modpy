import numpy as np
from numpy.testing import assert_, assert_raises, assert_array_almost_equal

from modpy.tests.test_util import run_unit_test
from modpy.special import *


class TestErf:
    def test_result(self):
        # values from: https://en.wikipedia.org/wiki/Error_function
        x = np.array([-3.5, -2.5, -2.1, -1.4, -.7, 0., .1, .3, .9, 1.6])
        y = np.array([-0.999999257,
                      -0.999593048,
                      -0.997020533,
                      -0.952285120,
                      -0.677801194,
                      0.,
                      0.112462916,
                      0.328626759,
                      0.796908212,
                      0.976348383])

        assert_array_almost_equal(erf(x), y, decimal=7)


class TestErfInv:
    # values from: https://en.wikipedia.org/wiki/Error_function
    x = np.array([-3.5, -2.5, -2.1, -1.4, -.7, 0., .1, .3, .9, 1.6])
    y = np.array([-0.999999257,
                  -0.999593048,
                  -0.997020533,
                  -0.952285120,
                  -0.677801194,
                  0.,
                  0.112462916,
                  0.328626759,
                  0.796908212,
                  0.976348383])

    assert_array_almost_equal(erfinv(y), x, decimal=1)


class TestGamma:
    def test_result(self):
        # values from: https://en.wikipedia.org/wiki/Gamma_function
        x = np.array([-3./2., -.5, .5, 1., 3./2., 2., 5./2., 3., 7./2., 4.])
        y = np.array([2.36327180120735470306,
                      -3.54490770181103205459,
                      1.77245385090551602729,
                      1.,
                      0.88622692545275801364,
                      1.,
                      1.32934038817913702047,
                      2.,
                      3.32335097044784255118,
                      6.])

        assert_array_almost_equal(gamma(x), y, decimal=14)


class TestGammaLn:
    def test_result(self):
        # values from: https://www.mathworks.com/help/symbolic/gammaln.html
        x = np.array([1./5., 1./2., 2./3., 8./7., 3.])
        y = np.array([1.5241, 0.5724, 0.30332, -0.0667, 0.6931])

        assert_array_almost_equal(gammaln(x), y, decimal=3)


class TestGammaInc:
    def test_result(self):
        # values from: https://en.wikipedia.org/wiki/Gamma_function
        x = np.array([0., 1., 10., 100.])
        y = np.array([0., 0.84270079, 0.99999226, 1.])

        assert_array_almost_equal(gammainc(x, .5), y, decimal=5)


class TestGammaIncc:
    def test_result(self):
        # values from: https://en.wikipedia.org/wiki/Gamma_function
        x = np.array([0., 1., 10., 100., 1000.])
        y = np.array([1., 1.57299207e-01, 7.74421643e-6, 2.08848758e-45, 0.])

        assert_array_almost_equal(gammaincc(x, .5), y, decimal=9)


class TestGammaIncInv:
    def test_result(self):
        # values from: https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.gammainccinv.html
        y = np.array([0., .1, .5, 1.])
        x = np.array([0., 0.00789539, 0.22746821, np.inf])

        assert_array_almost_equal(gammaincinv(y, .5), x, decimal=8)


class TestGammaInccInv:
    def test_result(self):
        # values from: https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.gammainccinv.html
        y = np.array([0., .1, .5, 1.])
        x = np.array([np.inf, 1.35277173, 0.22746821, 0.])

        assert_array_almost_equal(gammainccinv(y, .5), x, decimal=8)


class TestBeta:
    def test_result_a(self):
        a = np.arange(1, 11).astype(dtype=np.float64)
        b = 3.

        y = np.array([1./3.,
                      1./12.,
                      1./30.,
                      1./60.,
                      1./105.,
                      1./168.,
                      1./252.,
                      1./360.,
                      1./495.,
                      1./660.])

        assert_array_almost_equal(beta(a, b), y, decimal=15)

    def test_result_b(self):
        a = 2.
        b = np.arange(1, 8).astype(dtype=np.float64)

        y = np.array([1./2., 1./6., 1./12., 1./20., 1./30., 1./42., 1./56.])

        assert_array_almost_equal(beta(a, b), y, decimal=15)


class TestBetaInc:
    def test_result_x(self):
        # values from https://www.ncl.ucar.edu/Document/Functions/Built-in/betainc.shtml
        x = np.array([0.2, 0.5])
        y = np.array([0.85507, 0.98988])

        assert_array_almost_equal(betainc(x, .5, 5.), y, decimal=5)

    def test_result_a(self):
        # values from https://www.mathworks.com/help/matlab/ref/betainc.html
        x = .5
        a = np.arange(1, 11).astype(dtype=np.float64)
        b = 3.

        y = np.array([0.875,
                      0.6875,
                      0.5,
                      0.34375,
                      0.2265625,
                      0.14453125,
                      0.08984375,
                      0.0546875,
                      0.03271484375,
                      0.019287109375])

        assert_array_almost_equal(betainc(x, a, b), y, decimal=5)


class TestBetaIncInv:
    def test_result(self):
        # values from https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.betaincinv.html
        a = np.array([1.2, 7.5, 0.6, 3.4])
        b = np.array([3.1, 0.4, 4.3, 2.2])
        x = np.array([0.2, 0.5, 0.7, 0.9])
        y = betainc(x, a, b)

        assert_array_almost_equal(betaincinv(y, a, b), x, decimal=10)


if __name__ == '__main__':
    run_unit_test(TestErf)
    run_unit_test(TestErfInv)
    run_unit_test(TestGamma)
    run_unit_test(TestGammaLn)
    run_unit_test(TestGammaInc)
    run_unit_test(TestGammaIncc)
    run_unit_test(TestGammaIncInv)
    run_unit_test(TestGammaInccInv)
    run_unit_test(TestBeta)
    run_unit_test(TestBetaInc)
    run_unit_test(TestBetaIncInv)
