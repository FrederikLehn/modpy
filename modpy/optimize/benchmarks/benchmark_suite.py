import inspect
from warnings import warn
import numpy as np
import numpy.linalg as la

from modpy.optimize import LinearConstraint, NonlinearConstraint
from modpy.optimize._constraints import prepare_bounds


# ======================================================================================================================
# UNCONSTRAINED TEST FUNCTIONS
# ======================================================================================================================
# ----------------------------------------------------------------------------------------------------------------------
# QUADRATIC (CONVEX) TEST FUNCTIONS
# ----------------------------------------------------------------------------------------------------------------------
# sphere (convex)
def sphere(x):
    return np.sum(x ** 2.)


# ellipsoid (convex)
def ellipsoid(x, s=1e6):
    n = x.size
    return np.sum(s ** (np.arange(n) / (n - 1.)) * x ** 2.)


# cigar (convex)
def cigar(x, s=1e6):
    return x[0] ** 2. + np.sum(s * x[1:] ** 2.)


# discus (convex)
def discus(x, s=1e6):
    return s * x[0] ** 2. + np.sum(x[1:] ** 2.)


# cigar-discus (convex)
def cigar_discus(x, s=1e6):
    return s * x[0] ** 2. + np.sum(np.sqrt(s) * x[1:-1] ** 2.) + x[-1] ** 2.


# two-axes (convex)
def two_axes(x, s=1e6, theta=0.5):
    m = int(x.size * theta)
    return np.sum(s * x[:m] ** 2.) + np.sum(x[m:] ** 2.)


# ----------------------------------------------------------------------------------------------------------------------
# QUADRATIC (NON-CONVEX) TEST FUNCTIONS
# ----------------------------------------------------------------------------------------------------------------------
# diff-powers (non-convex)
def diff_powers(x):
    n = x.size
    return np.sum(np.abs(x) ** np.arange(2, n + 2))


# Rosenbrock (non-convex)
def rosenbrock(x):
    return np.sum(100. * (x[:-1] ** 2. - x[1:]) ** 2. + (x[:-1] - 1.) ** 2.)


# ----------------------------------------------------------------------------------------------------------------------
# NON-QUADRATIC TEST FUNCTIONS
# ----------------------------------------------------------------------------------------------------------------------
# Powell (n=m=2)
def powell(x):
    f1 = 1e4 * x[0] * x[1] - 1.
    f2 = np.exp(-x[0]) + np.exp(-x[1]) - 1.0001
    return f1 ** 2 + f2 ** 2


# Brown (n=2, m=3)
def brown(x):
    f1 = x[0] - 1e6
    f2 = x[1] - 2. * 1e-6
    f3 = x[0] * x[1] - 2.
    return f1 ** 2 + f2 ** 2 + f3 ** 2


# Beale (n=2, m=3)
def beale(x):
    f1 = 1.5 - x[0] * (1. - x[1])
    f2 = 2.25 - x[0] * (1. - x[1] ** 2.)
    f3 = 2.625 - x[0] * (1. - x[1] ** 3.)
    return f1 ** 2 + f2 ** 2 + f3 ** 2


# helical valley (n=m=3)
def helical_valley(x):
    f1_ = (x[2] - 10. * np.arctan(x[1] / x[0]) / (2. * np.pi))
    f1 = 10. * (f1_ - (5. if x[0] > 0. else 0.))
    f2 = 10. * (np.sqrt(x[0] ** 2. + x[1] ** 2.) - 1.)
    f3 = x[2]
    return f1 ** 2 + f2 ** 2 + f3 ** 2


# Powell singular (n=m=4)
def powell_singular(x):
    f1 = x[0] + 10. * x[1]
    f2 = np.sqrt(5.) * (x[2] - x[3])
    f3 = (x[1] - 2. * x[2]) ** 2.
    f4 = np.sqrt(10.) * (x[0] - x[3]) ** 2.
    return f1 ** 2 + f2 ** 2 + f3 ** 2 + f4 ** 2


# Wood (n=4, m=6)
def wood(x):
    f1 = 10. * (x[1] - x[0] ** 2.)
    f2 = 1. - x[0]
    f3 = np.sqrt(90.) * (x[3] - x[2] ** 2.)
    f4 = 1. - x[2]
    f5 = np.sqrt(10.) * (x[1] + x[3] - 2.)
    f6 = 10. ** (-.5) * (x[1] - x[3])
    return f1 ** 2 + f2 ** 2 + f3 ** 2 + f4 ** 2 + f5 ** 2 + f6 ** 2


# ======================================================================================================================
# CONSTRAINED TEST FUNCTIONS
# ======================================================================================================================
# General comment for all constraints in this section. They are formulated as con(x)<=0.
# They are all reformulated to con(x)>=0 by a minus to fit with algorithm implementations.

# ----------------------------------------------------------------------------------------------------------------------
# G-class Test Functions
# ----------------------------------------------------------------------------------------------------------------------
# g01
def g01(x):
    return 5. * np.sum(x[:4]) - 5. * np.sum(x[:4] ** 2.) - np.sum(x[4:])


g01_lin_con = -np.block([[2., 2., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0.],
                         [2., 0., 2., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1.],
                         [0., 2., 2., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
                         [-8., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
                         [0., -8., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
                         [0., 0., -8., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
                         [0., 0., 0., -2., -1., 0., 0., 0., 0., 1., 0., 0., 0.],
                         [0., 0., 0., 0., 0., -2., -1., 0., 0., 0., 1., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., -2., -1., 0., 0., 1., 0.]])

g01_lin_lb = np.array([-10., -10., -10., 0., 0., 0., 0., 0., 0.])
g01_lin_ub = np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])
g01_bounds = (*[(0., 1.) for _ in range(9)], *[(0., 100.) for _ in range(3)], (0., 1.))


# g02 (notice "-" as this is a maximization problem)
def g02(x):
    n = x.size
    return -np.abs((np.sum(np.cos(x) ** 4.) - 2. * np.prod(np.cos(x) ** 2.)) /
                   np.sqrt(np.sum(np.arange(1, n + 1) * x ** 2.)))


g02_lin_con = -np.ones((1, 20))
g02_lin_lb = np.array([-0.75 * 20.])  # 0.75 or 7.5?
g02_lin_ub = np.array([np.inf])


def g02_nl_con(x):
    return -np.array([-np.prod(x)])


g02_nl_lb = np.array([0.75])
g02_nl_ub = np.array([np.inf])
g02_bounds = tuple([(0., 10.) for _ in range(20)])


# g03 (notice "-" as this is a maximization problem)
def g03(x):
    n = float(x.size)
    return -(np.sqrt(n)) ** n * np.prod(x)


# equality constraint
def g03_nl_con(x):
    return np.array([np.sum(x ** 2.)])


g03_nl_lb = np.array([1.])
g03_nl_ub = np.array([1.])
g03_bounds = tuple([(0., 1.) for _ in range(10)])


# g04
def g04(x):
    return 5.3578547 * x[2] ** 2. + 0.8356891 * x[0] * x[4] + 37.293239 * x[0] - 40792.141


def g04_nl_con(x):
    return -np.array([0.0056858 * x[1] * x[4] + 0.0006262 * x[0] * x[3] - 0.0022053 * x[2] * x[4],
                      -0.0056858 * x[1] * x[4] - 0.0006262 * x[0] * x[3] + 0.0022053 * x[2] * x[4],
                      0.0071317 * x[1] * x[4] + 0.0029955 * x[0] * x[1] + 0.0021813 * x[2] ** 2.,
                      -0.0071317 * x[1] * x[4] - 0.0029955 * x[0] * x[1] - 0.0021813 * x[2] ** 2.,
                      0.0047026 * x[2] * x[4] + 0.0012547 * x[0] * x[2] + 0.0019085 * x[2] * x[3],
                      -0.0047026 * x[2] * x[4] - 0.0012547 * x[0] * x[2] - 0.0019085 * x[2] * x[3]])


g04_nl_lb = np.array([-6.665593, -85.334407, -29.48751, 9.48751, -15.699039, 10.699039])
g04_nl_ub = np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])
g04_bounds = ((78., 102.), (33., 45.), *[(27., 45.) for _ in range(3)])


# g05
def g05(x):
    return 3. * x[0] + 0.000001 * x[0] ** 3. + 2. * x[1] + (0.000002 / 3.) * x[1] ** 3.


g05_lin_con = -np.block([[0., 0., 1., -1.],
                         [0., 0., -1., 1.]])

g05_lin_lb = np.array([-0.55, -0.55])
g05_lin_ub = np.array([np.inf, np.inf])


# equality constraints
def g05_nl_con(x):
    return np.array([1000. * np.sin(-x[2] - 0.25) + 1000. * np.sin(-x[3] - 0.25) - x[0],
                     1000. * np.sin(x[2] - 0.25) + 1000. * np.sin(x[2] - x[3] - 0.25) - x[1],
                     1000. * np.sin(x[3] - 0.25) + 1000. * np.sin(x[3] - x[2] - 0.25)])


g05_nl_lb = np.array([-894.8, -894.8, -1294.8])
g05_nl_ub = np.array([-894.8, -894.8, -1294.8])
g05_bounds = ((0., 1200.), (0., 1200.), (-0.55, 0.55), (-0.55, 0.55))


# g06: Floudas and Pardalos
def g06(x):
    return (x[0] - 10.) ** 3. + (x[1] - 20.) ** 3.


def g06_nl_con(x):
    return -np.array([-(x[0] - 5.) ** 2. - (x[1] - 5.) ** 2.,
                      (x[0] - 6.) ** 2. + (x[1] - 5.) ** 2.])


g06_nl_lb = np.array([100., -82.81])
g06_nl_ub = np.array([np.inf, np.inf])
g06_bounds = ((13., 100.), (0., 100.))


# g07: Hock and Schittkowski
def g07(x):
    return x[0] ** 2. + x[1] ** 2. + x[0] * x[1] - 14. * x[0] - 16. * x[1] + (x[2] - 10.) ** 2. + \
           4. * (x[3] - 5.) ** 2. + (x[4] - 3.) ** 2. + 2. * (x[5] - 1.) ** 2. + 5 * x[6] ** 2. + \
           7. * (x[7] - 11.) ** 2. + 2. * (x[8] - 10.) ** 2. + (x[9] - 7.) ** 2. + 45.


g07_lin_con = -np.block([[4., 5., 0., 0., 0., 0., -3., 9., 0., 0.],
                         [10., -8., 0., 0., 0., 0., -17., 2., 0., 0.],
                         [-8., 2., 0., 0., 0., 0., 0., 0., 5., -2.]])

g07_lin_lb = np.array([-105., 0., -12.])
g07_lin_ub = np.array([np.inf, np.inf, np.inf])


def g07_nl_con(x):
    return -np.array([-3. * x[0] + 6. * x[1] + 12. * (x[8] - 8.) ** 2. - 7. * x[9],
                      3. * (x[0] - 2.) ** 2. + 4. * (x[1] - 3.) ** 2. + 2. * x[2] ** 2. - 7. * x[3],
                      x[0] ** 2. + 2. * (x[1] - 2.) ** 2. - 2. * x[0] * x[1] + 14. * x[4] - 6. * x[5],
                      5. * x[0] ** 2. + 8. * x[1] + (x[2] - 6.) ** 2. - 2. * x[3],
                      (x[0] - 8.) ** 2. + 4. * (x[1] - 4.) ** 2. + 6. * x[4] ** 2. - 2. * x[5]])


g07_nl_lb = np.array([0., -120., 0., -40., -60.])
g07_nl_ub = np.array([np.inf, np.inf, np.inf, np.inf, np.inf])
g07_bounds = tuple([(-10., 10) for _ in range(10)])


# g08 (notice "-" as this is a maximization problem)
def g08(x):
    pi2 = 2. * np.pi
    return -np.sin(pi2 * x[0]) ** 3. * np.sin(pi2 * x[1]) / (x[0] ** 3. * (x[0] + x[1]))


def g08_nl_con(x):
    return -np.array([x[0] ** 2. - x[1],
                      -x[0] + (x[1] - 4.) ** 2.])


g08_nl_lb = np.array([1., 1.])
g08_nl_ub = np.array([np.inf, np.inf])
g08_bounds = ((0., 10.), (0., 10.))


# g09: Hock and Schittkowski
def g09(x):
    return (x[0] - 10.) ** 2. + 5. * (x[1] - 12.) ** 2. + x[2] ** 4. + 3. * (x[3] - 11.) ** 2. + 10. * x[4] ** 6. +\
           7. * x[5] ** 2. + x[6] ** 4. - 4. * x[5] * x[6] - 10. * x[5] - 8. * x[6]


def g09_nl_con(x):
    return -np.array([2. * x[0] ** 2. + 3. * x[1] ** 4. + x[2] + 4. * x[3] ** 2. + 5. * x[4],
                      23. * x[0] + x[1] ** 2. + 6. * x[5] ** 2. - 8. * x[6],
                      7. * x[0] + 3. * x[1] + 10. * x[2] ** 2. + x[3] - x[4],
                      4. * x[0] ** 2. + x[1] ** 2. - 3. * x[0] * x[1] + 2. * x[2] ** 2. + 5. * x[5] - 11. * x[6]])


g09_nl_lb = np.array([-127., -196., -282., 0.])
g09_nl_ub = np.array([np.inf, np.inf, np.inf, np.inf])
g09_bounds = tuple([(-10., 10) for _ in range(7)])


# g10: Hock and Schittkowski
def g10(x):
    return x[0] + x[1] + x[2]


g10_lin_con = -np.block([[0., 0., 0., 0.0025, 0., 0.0025, 0., 0.],
                         [0., 0., 0., -0.0025, 0.0025, 0., 0.0025, 0.],
                         [0., 0., 0., 0., -0.01, 0., 0., 0.01]])

g10_lin_lb = np.array([-1., -1., -1.])
g10_lin_ub = np.array([np.inf, np.inf, np.inf])


def g10_nl_con(x):
    return -np.array([-x[0] * x[5] + 833.33252 * x[3] + 100. * x[0],
                      -x[1] * x[6] + 1250. * x[4] + x[1] * x[3] - 1250. * x[3],
                      -x[2] * x[7] + x[2] * x[4] - 2500. * x[4]])


g10_nl_lb = np.array([-83333.333, 0., 1250000.])
g10_nl_ub = np.array([np.inf, np.inf, np.inf])
g10_bounds = ((100., 10000.), *[(1000., 10000) for _ in range(2)], *[(10., 1000) for _ in range(5)])


# g11
def g11(x):
    return x[0] ** 2. + (x[1] - 1.) ** 2.


# equality constraint
def g11_nl_con(x):
    return np.array([x[1] - x[0] ** 2.])


g11_nl_lb = np.array([0.])
g11_nl_ub = np.array([0.])
g11_bounds = ((-1., 1.), (-1., 1.))


# g12 (notice "-" as this is a maximization problem)
# TODO (not sure I quite understand it)


# g13
def g13(x):
    return np.exp(np.prod(x))


# equality constraint
def g13_nl_con(x):
    return np.array([np.sum(x ** 2.),
                     x[1] * x[2] - 5. * x[3] * x[4],
                     x[0] ** 3. + x[1] ** 3.])


g13_nl_lb = np.array([10., 0., -1.])
g13_nl_ub = np.array([10., 0., -1.])
g13_bounds = ((-2.3, 2.3), (-2.3, 2.3), *[(-3.2, 3.2) for _ in range(3)])


# ----------------------------------------------------------------------------------------------------------------------
# Additional Test Functions
# ----------------------------------------------------------------------------------------------------------------------
# TR2: Kramer and Schwefel
def tr2(x):
    return x[0] ** 2. + x[1] ** 2.


tr2_lin_con = np.array([-1., -1.])
tr2_lin_lb = np.array([-np.inf])
tr2_lin_ub = np.array([-2])


# 2.40: Schwefel
def s240(x):
    return -np.sum(x)


s240_lin_con = 9. + np.arange(1, 6)
s240_lin_lb = np.array([-np.inf])
s240_lin_ub = np.array([50000.])
s240_bounds = tuple([(0., np.inf) for _ in range(5)])


# 2.41: Schwefel
def s241(x):
    return -np.sum(x * np.arange(1, 6))


s241_lin_con = 9. + np.arange(1, 6)
s241_lin_lb = np.array([-np.inf])
s241_lin_ub = np.array([50000.])
s241_bounds = tuple([(0., np.inf) for _ in range(5)])


# HB: Himmelblau
def himmelblau5d(x):
    return 5.3578547 * x[2] ** 2. + 0.8356891 * x[0] * x[4] + 37.293239 * x[0] - 40792.141


def himmelblau5d_nl_con(x):
    h1 = 85.334407 + 0.0056858 * x[1] * x[4] + 0.00026 * x[0] * x[3] - 0.0022053 * x[2] * x[4]
    h2 = 80.51249 + 0.0071317 * x[1] * x[4] + 0.0029955 * x[0] * x[1] + 0.0021813 * x[2] ** 2.
    h3 = 9.300961 + 0.0047026 * x[2] * x[4] + 0.0012547 * x[0] * x[2] + 0.0019085 * x[2] * x[3]

    return -np.array([-h1, h1, -h2, h2, -h3, h3])


himmelblau5d_nl_lb = np.array([0., -92., 90., -110., 20., -25.])
himmelblau5d_nl_ub = np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])
himmelblau5d_bounds = tuple([(78., 102.), (33., 45.), *[(27., 45.) for _ in range(3)]])


# ======================================================================================================================
# MULTI-MODAL TEST FUNCTIONS
# ======================================================================================================================
# ----------------------------------------------------------------------------------------------------------------------
# General functions (for global optimization)
# ----------------------------------------------------------------------------------------------------------------------
# Rastrigin
def rastrigin(x):
    return np.sum(x ** 2. - 10. * np.cos(2. * np.pi * x)) + 10. * x.size


# Ackley
def ackley(x):
    a = 20.
    b = 0.2
    c = 2. * np.pi
    return -a * np.exp(-b * np.sqrt(np.mean(x ** 2.))) - np.exp(np.mean(np.cos(c * x))) + a + np.exp(1)


# Lévi N.13 (n=2)
def levin13(x):
    pi2 = 2. * np.pi
    pi3 = 3. * np.pi
    return np.sin(pi3 * x[0]) ** 2. + (x[0] - 1.) ** 2. * (1. + np.sin(pi3 * x[1]) ** 2.) +\
           (x[1] - 1.) ** 2. * (1. + np.sin(pi2 * x[1]) ** 2.)


# cross-in-tray (n=2)
def cross_in_tray(x):
    return -0.0001 * (np.abs(np.sin(x[0]) * np.sin(x[1]) * np.exp(np.abs(100. - np.sqrt(x[0] ** 2. + x[1] ** 2.) / np.pi))) + 1) ** 0.1


# eggholder (n=2)
def eggholder(x):
    return -(x[1] + 47.) * np.sin(np.sqrt(np.abs(x[0] / 2. + x[1] + 47.))) - x[0] * np.sin(np.sqrt(np.abs(x[0] - (x[1] + 47.))))


# Hölder table (n=2)
def holder_table(x):
    return -np.abs(np.sin(x[0]) * np.cos(x[1]) * np.exp(np.abs(1. - np.sqrt(x[0] ** 2. + x[1] ** 2.) / np.pi)))


# Schaffer N.2
def schaffern2(x):
    return 0.5 + (np.sin(x[0] ** 2. - x[1] ** 2.) ** 2. - 0.5) / (1. + 0.001 * (x[0] ** 2. + x[1] ** 2.)) ** 2.


# Schaffer N.4
def schaffern4(x):
    return 0.5 + (np.cos(np.sin(np.abs(x[0] ** 2. - x[1] ** 2.))) ** 2. - 0.5) / (1. + 0.001 * (x[0] ** 2. + x[1] ** 2.)) ** 2.


# Styblinski-Tang
def styblinski_tang(x):
    return .5 * np.sum(x ** 4. - 16. * x ** 2. + 5. * x)


# Griewank
def griewank(x):
    return np.sum(x ** 2.) / 4000. - np.prod(np.cos(x / np.sqrt(np.arange(1, x.shape[0] + 1)))) + 1.


# Weierstrass
# TODO: does not quite seem to match with Wikipedia?
def weierstrass(x):
    alpha = .5
    beta = 3.
    kmax = 20
    D = x.shape[0]

    c1 = alpha ** np.arange(kmax + 1)
    c2 = 2. * np.pi * beta ** np.arange(kmax + 1)
    f = 0.
    c = -D * np.sum(c1 * np.cos(c2 * 0.5))

    for i in range(D):
        f += np.sum(c1 * np.cos(c2 * (x[i] + 0.5)))
    return f + c


# F8F2 (Expanded, extended Griewank + Rosenbrock)
def f8f2(x):
    f2 = 100. * (x[0] ** 2 - x[1]) ** 2. + (1. - x[0]) ** 2.
    return 1. + (f2 ** 2) / 4000. - np.cos(f2)


# FEF8F2 function
def fef8f2(x):
    D = x.shape[0]
    f = 0.
    for i in range(D - 1):
        f += f8f2(x[[i, i + 1]] + 1)

    f += f8f2(x[[D - 1, 0]] + 1)

    return f


# ----------------------------------------------------------------------------------------------------------------------
# CEC2013 functions (for MMO benchmarking)
# ----------------------------------------------------------------------------------------------------------------------
# F1: Five-Uneven-Peak Trap
# Variable ranges: x in [0, 30]
# No. of global peaks: 2
# No. of local peaks:  3.
# notice return of -obj to convert from maximum o minimum.
def five_uneven_peak_trap(x):

    result = 0  # None?
    if 0. <= x < 2.5:
        result = 80. * (2.5 - x)
    elif 2.5 <= x < 5.:
        result = 64. * (x - 2.5)
    elif 5. <= x < 7.5:
        result = 64. * (7.5 - x)
    elif 7.5 <= x < 12.5:
        result = 28. * (x - 7.5)
    elif 12.5 <= x < 17.5:
        result = 28. * (17.5 - x)
    elif 17.5 <= x < 22.5:
        result = 32. * (x - 17.5)
    elif 22.5 <= x < 27.5:
        result = 32. * (27.5 - x)
    elif 27.5 <= x <= 30.:
        result = 80. * (x - 27.5)

    return -result


# F2: Equal Maxima
# Variable ranges: x in [0, 1]
# No. of global peaks: 5
# No. of local peaks:  0.
# notice return of -obj to convert from maximum o minimum.
def equal_maxima(x):
    return -np.sin(5. * np.pi * x) ** 6.


# F3: Uneven Decreasing Maxima
# Variable ranges: x in [0, 1]
# No. of global peaks: 1
# No. of local peaks:  4.
# notice return of -obj to convert from maximum o minimum.
def uneven_decreasing_maxima(x):
    return -np.exp(-2. * np.log(2.) * ((x - 0.08) / 0.854) ** 2) * (np.sin(5. * np.pi * (x ** 0.75 - 0.05))) ** 6.


# F4: Himmelblau
# Variable ranges: x, y in [-6, 6]
# No. of global peaks: 4
# No. of local peaks:  0.
# NOTE: Not quite the one given in CEC2013, but the "True" Himmelblau function
def himmelblau(x):
    return (x[0] ** 2. + x[1] - 11.) ** 2. + (x[0] + x[1] ** 2. - 7.) ** 2.


# F5: Six-Hump Camel Back
# Variable ranges: x in [-1.9, 1.9]; y in [-1.1, 1.1]
# No. of global peaks: 2
# No. of local peaks:  2.
# notice return of -obj to convert from maximum o minimum.
def six_hump_camel_back(x):
    x2 = x[0] ** 2
    x4 = x[0] ** 4
    y2 = x[1] ** 2
    expr1 = (4.0 - 2.1 * x2 + x4 / 3.0) * x2
    expr2 = x[0] * x[1]
    expr3 = (4.0 * y2 - 4.0) * y2
    return -(-1.0 * (expr1 + expr2 + expr3))


# F6: Shubert
# Variable ranges: x_i in  [-10, 10]^n, i=1,2,...,n
# No. of global peaks: n*3^n
# No. of local peaks: many
# notice return of -obj to convert from maximum o minimum.
def shubert(x):

    result = 1
    soma = [0] * x.size
    D = x.size

    for i in range(D):
        for j in range(1, 6):
            soma[i] = soma[i] + (j * np.cos((j + 1) * x[i] + j))
        result = result * soma[i]

    return -(-result)


# F7: Vincent
# Variable range: x_i in [0.25, 10]^n, i=1,2,...,n
# No. of global optima: 6^n
# No. of local optima:  0.
# notice return of -obj to convert from maximum o minimum.
def vincent(x):
    return -np.sum(np.sin(10. * np.log(x))) / x.size


# F8: Modified Rastrigin - All Global Optima
# Variable ranges: x_i in [0, 1]^n, i=1,2,...,n
# No. of global peaks: \prod_{i=1}^n k_i
# No. of local peaks:  0.
# notice return of -obj to convert from maximum o minimum.
def modified_rastrigin_all(x):

    D = x.size

    if D == 2:
        k = [3, 4]
    elif D == 8:
        k = [1, 2, 1, 2, 1, 3, 1, 4]
    elif D == 16:
        k = [1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 3, 1, 1, 1, 4]
    else:
        raise ValueError('`x` must be of size 2, 8 or 16.')

    k = np.array(k, dtype=np.float64)
    return -(-np.sum(10. + 9. * np.cos(2. * np.pi * k * x)))


# ======================================================================================================================
# TEST FUNCTION CLASSES
# ======================================================================================================================
# optimums found at: https://en.wikipedia.org/wiki/Test_functions_for_optimization
# optimums found at: https://www.osti.gov/servlets/purl/6650344
# optimums found at: https://sgpp.sparsegrids.org/docs/namespacesgpp_1_1optimization_1_1test__problems.html
# optimums found at: http://www-optima.amp.i.kyoto-u.ac.jp/member/student/hedar/Hedar_files/TestGO_files/Page2424.htm
# optimums found at: https://notendur.hi.is/tpr/software/sres/testfcn.pdf
# optimums found at: https://onlinelibrary.wiley.com/doi/pdf/10.1002/9781119136507.app2

# additional problems: https://arxiv.org/pdf/1308.4008.pdf


class TestFunction:
    def __init__(self, dim=2):

        self.name = ''
        self.dim = dim  # dimension of the problem

        self.x_ini = None
        self.obj = None
        self.bounds = None
        self.constraints = ()

        # used for global optimization benchmarking
        self.x_opt = None
        self.f_opt = None

        # used for MMO benchmarking
        self.n_gopt = 1
        self.n_lopt = 0

    def get(self, x0, stochastic=False):
        x0 = self.start_guess(x0)

        input_ = {'obj': self.obj,
                 'x0': x0,
                 'bounds': self.bounds,
                 'constraints': self.constraints}

        if stochastic:

            sigma0 = self.start_deviation()
            lbound = self.get_lbound()
            input_ = {**input_,
                      'sigma0': sigma0,
                      'lbound': lbound}

        return input_

    def start_guess(self, x0=None):

        # decide on initial guess
        if (self.x_ini is None) and ((x0 is None) or (x0.size != self.dim)):
            x0_ = np.random.randn(self.dim)
        elif self.x_ini is not None:
            x0_ = self.x_ini
        else:
            x0_ = x0

        return np.array(x0_)

    def start_deviation(self):
        if self.bounds is not None:
            lb, ub = prepare_bounds(self.bounds, self.dim)
            sigma0 = np.mean(ub - lb) / 4
        else:
            sigma0 = 1.

        return sigma0

    def get_lbound(self):

        if self.f_opt is not None:

            if isinstance(self.f_opt, tuple):
                f_opt = np.amin(np.array(self.f_opt))
            else:
                f_opt = self.f_opt

            f_opt += 1e-2

        else:

            f_opt = np.inf

        return f_opt

    def found_optimum(self, res):

        if isinstance(self.x_opt, tuple):  # multiple optimums

            d = [la.norm(res.x - x_opt, np.inf) for x_opt in self.x_opt]
            idx = np.argsort(d)
            return self.x_opt[idx[0]]

        else:  # single optimum
            return self.x_opt

    def found_optimum_value(self, res):

        if isinstance(self.x_opt, tuple):  # multiple optimums

            d = np.block([np.abs(res.f - f_opt) for f_opt in self.f_opt])
            idx = np.argsort(d)
            return self.f_opt[idx[0]]

        else:  # single optimum
            return self.f_opt

    def check(self, res, ftol=1e-6, xtol=1e-4):
        converged = res.success

        f_success = True
        if converged and (self.f_opt is not None):

            ftol *= 1e2

            if isinstance(self.f_opt, tuple):  # multiple optimums

                for i, f_opt in enumerate(self.f_opt):

                    fn = abs(res.f - f_opt)
                    f_success = fn <= ftol

                    if f_success:
                        break

            else:  # single optimum
                fn = abs(res.f - self.f_opt)
                f_success = fn <= ftol

            if not f_success:
                res.message = 'Failed to satisfy ftol: {}<={}'.format(fn, ftol)

        x_success = True
        if converged and f_success and (self.x_opt is not None):

            if isinstance(self.x_opt, tuple):  # multiple optimums

                for i, x_opt in enumerate(self.x_opt):

                    xn = la.norm(res.x - x_opt, np.inf)
                    x_success = xn <= xtol

                    if x_success:
                        break

            else:  # single optimum
                xn = la.norm(res.x - self.x_opt, np.inf)
                x_success = xn <= xtol

            if not x_success:
                res.message = 'Failed to satisfy xtol: {}<={}'.format(xn, xtol)

        return converged and x_success and f_success

    def check_mmo(self, res_mmo, ftol=1e-6, xtol=1e-4):
        # check each optimum in res_mmo for having converged using `check` and check
        # that the results are different from each other.

        results = []
        track = []

        for i, res in enumerate(res_mmo.results):
            success = self.check(res, ftol, xtol)

            if success:
                different = True

                for r in results:

                    if np.allclose(res.x, r.x):
                        different = False
                        break

                if different:
                    results.append(res)
                    track.append(res_mmo.track[i])

        return len(results) > 0, results, track


class Sphere(TestFunction):
    def __init__(self, dim=2, bounds=None):
        super().__init__(dim)

        self.name = 'Sphere'

        self.obj = sphere
        self.bounds = bounds

        self.x_opt = np.zeros((dim,))
        self.f_opt = 0.


class Ellipsoid(TestFunction):
    def __init__(self, dim=2, bounds=None):
        super().__init__(dim)

        self.name = 'Ellipsoid'

        self.obj = ellipsoid
        self.bounds = bounds

        self.x_opt = np.zeros((dim,))
        self.f_opt = 0.


class Cigar(TestFunction):
    def __init__(self, dim=2, bounds=None):
        super().__init__(dim)

        self.name = 'Cigar'

        self.obj = cigar
        self.bounds = bounds

        self.x_opt = np.zeros((dim,))
        self.f_opt = 0.


class Discus(TestFunction):
    def __init__(self, dim=2, bounds=None):
        super().__init__(dim)

        self.name = 'Discus'

        self.obj = discus
        self.bounds = bounds

        self.x_opt = np.zeros((dim,))
        self.f_opt = 0.


class CigarDiscus(TestFunction):
    def __init__(self, dim=2, bounds=None):
        super().__init__(dim)

        self.name = 'Cigar-Discus'

        self.obj = cigar_discus
        self.bounds = bounds

        self.x_opt = np.zeros((dim,))
        self.f_opt = 0.


class TwoAxes(TestFunction):
    def __init__(self, dim=2, bounds=None):
        super().__init__(dim)

        self.name = 'Two Axes'

        self.obj = two_axes
        self.bounds = bounds

        self.x_opt = np.zeros((dim,))
        self.f_opt = 0.


class DiffPowers(TestFunction):
    def __init__(self, dim=2, bounds=None):
        super().__init__(dim)

        self.name = 'Diff Powers'

        self.obj = diff_powers
        self.bounds = bounds

        self.x_opt = np.zeros((dim,))
        self.f_opt = 0.


class Rosenbrock(TestFunction):
    def __init__(self, dim=2, bounds=None):
        super().__init__(dim)

        self.name = 'Rosenbrock'

        self.obj = rosenbrock
        self.bounds = bounds

        self.x_opt = np.ones((dim,))
        self.f_opt = 0.


class Powell(TestFunction):
    def __init__(self, dim=2):
        super().__init__(dim)

        if dim != 2:
            warn('`dim` passed to Powell is incorrect, continuing with dim=2.')

        self.name = 'Powell'
        self.dim = 2
        self.obj = powell
        self.x_ini = np.array([0., 1.])
        self.x_opt = np.array([1.098 * 1e-5, 9.106])
        self.f_opt = 0.


class Brown(TestFunction):
    def __init__(self, dim=2):
        super().__init__(dim)

        if dim != 2:
            warn('`dim` passed to Brown is incorrect, continuing with dim=2.')

        self.name = 'Brown'
        self.dim = 2
        self.obj = brown
        self.x_ini = np.array([1., 1.])
        self.x_opt = np.array([1e6, 2. * 1e-6])
        self.f_opt = 0.


class Beale(TestFunction):
    def __init__(self, dim=2):
        super().__init__(dim)

        if dim != 2:
            warn('`dim` passed to Beale is incorrect, continuing with dim=2.')

        self.name = 'Beale'
        self.dim = 2
        self.obj = beale
        self.x_ini = np.array([1., 1.])
        self.x_opt = np.array([3.0, 0.5])
        self.f_opt = 0.


class HelicalValley(TestFunction):
    def __init__(self, dim=3):
        super().__init__(dim)

        if dim != 3:
            warn('`dim` passed to HelicalValley is incorrect, continuing with dim=3.')

        self.name = 'Helical Valley'
        self.dim = 3
        self.obj = helical_valley
        self.x_ini = np.array([-1., 0., 0.])
        self.x_opt = np.array([1., 0., 0.])
        self.f_opt = 0.


class PowellSingular(TestFunction):
    def __init__(self, dim=4):
        super().__init__(dim)

        if dim != 4:
            warn('`dim` passed to PowellSingular is incorrect, continuing with dim=4.')

        self.name = 'Powell Singular'
        self.dim = 4
        self.obj = powell_singular
        self.x_ini = np.array([3., -1., 0., 1.])
        self.x_opt = np.array([0., 0., 0., 0.])
        self.f_opt = 0.


class Wood(TestFunction):
    def __init__(self, dim=4):
        super().__init__(dim)

        if dim != 4:
            warn('`dim` passed to Wood is incorrect, continuing with dim=4.')

        self.name = 'Wood'
        self.dim = 4
        self.obj = wood
        self.x_ini = np.array([-3., -1., -3., -1.])
        self.x_opt = np.array([1., 1., 1., 1.])
        self.f_opt = 0.


class G01(TestFunction):
    def __init__(self, dim=13):
        super().__init__(dim)

        if dim != 13:
            warn('`dim` passed to G01 is incorrect, continuing with dim=13.')

        self.name = 'G01'
        self.dim = 13

        self.obj = g01
        self.bounds = g01_bounds
        self.constraints = LinearConstraint(g01_lin_con, lb=g01_lin_lb, ub=g01_lin_ub)

        self.x_opt = np.array([1., 1., 1., 1., 1., 1., 1., 1., 1., 3., 3., 3., 1.])
        self.f_opt = -15.


class G02(TestFunction):
    def __init__(self, dim=20):
        super().__init__(dim)

        if dim != 20:
            warn('`dim` passed to G02 is incorrect, continuing with dim=20.')

        self.name = 'G02'
        self.dim = 20

        self.obj = g02
        self.bounds = g02_bounds
        self.constraints = (LinearConstraint(g02_lin_con, lb=g02_lin_lb, ub=g02_lin_ub),
                            NonlinearConstraint(g02_nl_con, lb=g02_nl_lb, ub=g02_nl_ub))

        # global maximum is unknown, the below value is reported as the best found.
        self.f_opt = -0.80361910412559


class G03(TestFunction):
    def __init__(self, dim=10):
        super().__init__(dim)

        if dim != 10:
            warn('`dim` passed to G03 is incorrect, continuing with dim=10.')

        self.name = 'G03'
        self.dim = 10

        self.obj = g03
        self.bounds = g03_bounds
        self.constraints = NonlinearConstraint(g03_nl_con, lb=g03_nl_lb, ub=g03_nl_ub)

        self.x_opt = np.full((self.dim,), 1. / np.sqrt(self.dim))
        self.f_opt = 1.


class G04(TestFunction):
    def __init__(self, dim=5):
        super().__init__(dim)

        if dim != 5:
            warn('`dim` passed to G04 is incorrect, continuing with dim=5.')

        self.name = 'G04'
        self.dim = 5

        self.obj = g04
        self.bounds = g04_bounds
        self.constraints = NonlinearConstraint(g04_nl_con, lb=g04_nl_lb, ub=g04_nl_ub)

        self.x_opt = np.array([78, 33, 29.995256025682, 45, 36.775812905788])
        self.f_opt = -30665.539


class G05(TestFunction):
    def __init__(self, dim=4):
        super().__init__(dim)

        if dim != 4:
            warn('`dim` passed to G05 is incorrect, continuing with dim=4.')

        self.name = 'G05'
        self.dim = 4

        self.obj = g05
        self.bounds = g05_bounds
        self.constraints = (LinearConstraint(g05_lin_con, lb=g05_lin_lb, ub=g05_lin_ub),
                            NonlinearConstraint(g05_nl_con, lb=g05_nl_lb, ub=g05_nl_ub))

        self.x_opt = np.array([679.9453, 1026.067, 0.1188764, -0.3962336])
        self.f_opt = 5126.4981


class G06(TestFunction):
    def __init__(self, dim=2):
        super().__init__(dim)

        if dim != 2:
            warn('`dim` passed to G06 is incorrect, continuing with dim=2.')

        self.name = 'G06'
        self.dim = 2

        self.obj = g06
        self.bounds = g06_bounds
        self.constraints = NonlinearConstraint(g06_nl_con, lb=g06_nl_lb, ub=g06_nl_ub)

        self.x_ini = np.array([50., 70.])  # not from references
        self.x_opt = np.array([14.095, 0.8429608])
        self.f_opt = -6961.81381


class G07(TestFunction):
    def __init__(self, dim=10):
        super().__init__(dim)

        if dim != 10:
            warn('`dim` passed to G07 is incorrect, continuing with dim=10.')

        self.name = 'G07'
        self.dim = 10

        self.obj = g07
        self.bounds = g07_bounds
        self.constraints = (LinearConstraint(g07_lin_con, lb=g07_lin_lb, ub=g07_lin_ub),
                            NonlinearConstraint(g07_nl_con, lb=g07_nl_lb, ub=g07_nl_ub))

        self.x_opt = np.array([2.171996, 2.363683, 8.773926, 5.095984, 0.9906548,
                               1.430574, 1.321644, 9.828726, 8.280092, 8.375927])
        self.f_opt = 24.3062091


class G08(TestFunction):
    def __init__(self, dim=2):
        super().__init__(dim)

        if dim != 2:
            warn('`dim` passed to G08 is incorrect, continuing with dim=2.')

        self.name = 'G08'
        self.dim = 2

        self.obj = g08
        self.bounds = g08_bounds
        self.constraints = NonlinearConstraint(g08_nl_con, lb=g08_nl_lb, ub=g08_nl_ub)

        self.x_opt = np.array([1.2279713, 4.2453733])
        self.f_opt = 0.095825


class G09(TestFunction):
    def __init__(self, dim=7):
        super().__init__(dim)

        if dim != 7:
            warn('`dim` passed to G09 is incorrect, continuing with dim=7.')

        self.name = 'G09'
        self.dim = 7

        self.obj = g09
        self.bounds = g09_bounds
        self.constraints = NonlinearConstraint(g09_nl_con, lb=g09_nl_lb, ub=g09_nl_ub)

        self.x_opt = np.array([2.330499, 1.951372, -0.4775414, 4.365726, -0.6244870, 1.038131, 1.594227])
        self.f_opt = 680.630057


class G10(TestFunction):
    def __init__(self, dim=8):
        super().__init__(dim)

        if dim != 8:
            warn('`dim` passed to G10 is incorrect, continuing with dim=8.')

        self.name = 'G10'
        self.dim = 8

        self.obj = g10
        self.bounds = g10_bounds
        self.constraints = (LinearConstraint(g10_lin_con, lb=g10_lin_lb, ub=g10_lin_ub),
                            NonlinearConstraint(g10_nl_con, lb=g10_nl_lb, ub=g10_nl_ub))

        self.x_opt = np.array([579.3167, 1359.943, 5110.071, 182.0174, 295.5985, 217.9799, 286.4162, 395.5979])
        self.f_opt = 7049.331  # 7049.2480?


class G11(TestFunction):
    def __init__(self, dim=2):
        super().__init__(dim)

        if dim != 2:
            warn('`dim` passed to G11 is incorrect, continuing with dim=2.')

        self.name = 'G11'
        self.dim = 2

        self.obj = g11
        self.bounds = g11_bounds
        self.constraints = NonlinearConstraint(g11_nl_con, lb=g11_nl_lb, ub=g11_nl_ub)

        self.x_opt = (np.array([1. / np.sqrt(2), 0.5]),
                      np.array([-1. / np.sqrt(2), 0.5]))

        self.f_opt = (0.75, 0.75)


class G13(TestFunction):
    def __init__(self, dim=5):
        super().__init__(dim)

        if dim != 5:
            warn('`dim` passed to G13 is incorrect, continuing with dim=5.')

        self.name = 'G13'
        self.dim = 5

        self.obj = g13
        self.bounds = g13_bounds
        self.constraints = NonlinearConstraint(g13_nl_con, lb=g13_nl_lb, ub=g13_nl_ub)

        self.x_opt = np.array([-1.717143, 1.595709, 1.827247, -0.7636413, -0.763645])
        self.f_opt = 0.0539498


class TR2(TestFunction):
    def __init__(self, dim=2):
        super().__init__(dim)

        if dim != 2:
            warn('`dim` passed to TR2 is incorrect, continuing with dim=2.')

        self.name = 'TR2'
        self.dim = 2

        self.obj = tr2
        self.constraints = LinearConstraint(tr2_lin_con, lb=tr2_lin_lb, ub=tr2_lin_ub)

        self.x_ini = np.array([50., 50.])
        self.x_opt = np.array([1., 1.])
        self.f_opt = 2.


class S240(TestFunction):
    def __init__(self, dim=5):
        super().__init__(dim)

        if dim != 5:
            warn('`dim` passed to S240 is incorrect, continuing with dim=5.')

        self.name = 'S240'
        self.dim = 5

        self.obj = s240
        self.bounds = s240_bounds
        self.constraints = LinearConstraint(s240_lin_con, lb=s240_lin_lb, ub=s240_lin_ub)

        self.x_ini = np.array([250., 250., 250., 250., 250.])
        self.x_opt = np.array([5000., 0., 0., 0., 0.])
        self.f_opt = -5000.


class S241(TestFunction):
    def __init__(self, dim=5):
        super().__init__(dim)

        if dim != 5:
            warn('`dim` passed to S241 is incorrect, continuing with dim=5.')

        self.name = 'S241'
        self.dim = 5

        self.obj = s241
        self.bounds = s241_bounds
        self.constraints = LinearConstraint(s241_lin_con, lb=s241_lin_lb, ub=s241_lin_ub)

        self.x_ini = np.array([250., 250., 250., 250., 250.])
        self.x_opt = np.array([0., 0., 0., 0., 3571.42847])
        self.f_opt = -125000. / 7.


class Himmelblau5D(TestFunction):
    def __init__(self, dim=5):
        super().__init__(dim)

        if dim != 5:
            warn('`dim` passed to Himmelblau5D is incorrect, continuing with dim=5.')

        self.name = 'Himmelblau'
        self.dim = 5

        self.obj = himmelblau5d
        self.bounds = himmelblau5d_bounds
        self.constraints = NonlinearConstraint(himmelblau5d_nl_con, lb=himmelblau5d_nl_lb, ub=himmelblau5d_nl_ub)

        #self.x_ini = None
        self.x_opt = None
        self.f_opt = -30665.539


class Rastrigin(TestFunction):
    def __init__(self, dim):
        super().__init__(dim)

        self.name = 'Rastrigin'
        self.obj = rastrigin
        self.bounds = tuple((-5.12, 5.12) for _ in range(dim))

        self.x_opt = (np.zeros((dim,)),)
        self.f_opt = 0.

        self.n_gopt = 1
        self.n_lopt = 11 * dim - 1  # 11 minima per dim, in the search space


class Ackley(TestFunction):
    def __init__(self, dim):
        super().__init__(dim)

        self.name = 'Ackley'
        self.obj = ackley
        self.bounds = tuple((-32.768, 32.768) for _ in range(dim))

        self.x_opt = (np.zeros((dim,)),)
        self.f_opt = 0.


class LeviN13(TestFunction):
    def __init__(self, dim=2):
        super().__init__(dim)

        if dim != 2:
            warn('`dim` passed to LeviN13 is incorrect, continuing with dim=2.')

        self.name = 'Lévi N.13'
        self.dim = 2

        self.obj = levin13
        self.bounds = ((-10., 10.), (-10., 10.))

        self.x_opt = np.array([1., 1.])
        self.f_opt = 0.


class CrossInTray(TestFunction):
    def __init__(self, dim=2):
        super().__init__(dim)

        if dim != 2:
            warn('`dim` passed to CrossInTray is incorrect, continuing with dim=2.')

        self.name = 'Cross-In-Tray'
        self.dim = 2

        self.obj = cross_in_tray
        self.bounds = ((-10., 10.), (-10., 10.))

        self.x_opt = (np.array([1.349406685353340, 1.349406608602084]),
                      np.array([1.349406685353340, -1.349406608602084]),
                      np.array([-1.349406685353340, 1.349406608602084]),
                      np.array([-1.349406685353340, -1.349406608602084]))

        self.f_opt = (-2.06261218, -2.06261218, -2.06261218, -2.06261218)
        self.n_gopt = 4


class Eggholder(TestFunction):
    def __init__(self, dim=2):
        super().__init__(dim)

        if dim != 2:
            warn('`dim` passed to Eggholder is incorrect, continuing with dim=2.')

        self.name = 'Eggholder'
        self.dim = 2

        self.obj = eggholder
        self.bounds = ((-512., 512.), (-512., 512.))

        self.x_opt = np.array([512., 404.2319])
        self.f_opt = -959.6407


class HolderTable(TestFunction):
    def __init__(self, dim=2):
        super().__init__(dim)

        if dim != 2:
            warn('`dim` passed to HolderTable is incorrect, continuing with dim=2.')

        self.name = 'Hölder Table'
        self.dim = 2

        self.obj = holder_table
        self.bounds = ((-10., 10.), (-10., 10.))

        self.x_opt = (np.array([8.05502, 9.66459]),
                      np.array([8.05502, -9.66459]),
                      np.array([-8.05502, 9.66459]),
                      np.array([-8.05502, -9.66459]))

        self.f_opt = (-19.2085, -19.2085, -19.2085, -19.2085)
        self.n_gopt = 4


class SchafferN2(TestFunction):
    def __init__(self, dim=2):
        super().__init__(dim)

        if dim != 2:
            warn('`dim` passed to SchafferN2 is incorrect, continuing with dim=2.')

        self.name = 'Schaffer N.2'
        self.dim = 2

        self.obj = schaffern2
        self.bounds = ((-100., 100.), (-100., 100.))

        self.x_opt = np.array([0., 0.])
        self.f_opt = 0.


class SchafferN4(TestFunction):
    def __init__(self, dim=2):
        super().__init__(dim)

        if dim != 2:
            warn('`dim` passed to SchafferN4 is incorrect, continuing with dim=2.')

        self.name = 'Schaffer N.4'
        self.dim = 2

        self.obj = schaffern4
        self.bounds = ((-100., 100.), (-100., 100.))

        self.x_opt = np.array([0., 1.253115])
        self.f_opt = 0.292579


class StyblinskiTang(TestFunction):
    def __init__(self, dim):
        super().__init__(dim)

        self.name = 'Styblinski-Tang'

        self.obj = schaffern4
        self.bounds = tuple([(-5., 5.) for _ in range(dim)])

        self.x_opt = np.full((dim,), -2.903534)
        self.f_opt = -39.16599 * dim


class Griewank(TestFunction):
    def __init__(self, dim):
        super().__init__(dim)

        self.name = 'Griewank'
        self.obj = griewank
        self.bounds = tuple((-600., 600.) for _ in range(dim))

        self.x_opt = (np.zeros((dim,)),)
        self.f_opt = 0.

        self.n_gopt = 1
        self.n_lopt = 191 * dim - 1     # 191 minima per dim - 1 in the search space
                                        # https://mathworld.wolfram.com/GriewankFunction.html


class FiveUnevenPeakTrap(TestFunction):
    def __init__(self, dim=1):
        super().__init__(dim)

        if dim != 1:
            warn('`dim` passed to FiveUnevenPeakTrap is incorrect, continuing with dim=1.')

        self.name = 'Five Uneven Peak Trap'
        self.dim = 1

        self.obj = five_uneven_peak_trap
        self.bounds = ((0., 30.),)

        self.x_opt = None
        self.f_opt = -200.

        self.n_gopt = 2
        self.n_lopt = 3


class EqualMaxima(TestFunction):
    def __init__(self, dim=1):
        super().__init__(dim)

        if dim != 1:
            warn('`dim` passed to EqualMaxima is incorrect, continuing with dim=1.')

        self.name = 'Equal Maxima'
        self.dim = 1

        self.obj = equal_maxima
        self.bounds = ((0., 1.),)

        self.x_opt = (0.1, 0.3, 0.5, 0.7, 0.9)
        self.f_opt = -1.

        self.n_gopt = 5
        self.n_lopt = 0


class UnevenDecreasingMaxima(TestFunction):
    def __init__(self, dim=1):
        super().__init__(dim)

        if dim != 1:
            warn('`dim` passed to UnevenDecreasingMaxima is incorrect, continuing with dim=1.')

        self.name = 'Uneven Decreasing Maxima'
        self.dim = 1

        self.obj = uneven_decreasing_maxima
        self.bounds = ((0., 1.),)

        self.x_opt = 0.08
        self.f_opt = -1.

        self.n_gopt = 1
        self.n_lopt = 4


class Himmelblau(TestFunction):
    def __init__(self, dim=2):
        super().__init__(dim)

        if dim != 2:
            warn('`dim` passed to Himmelblau is incorrect, continuing with dim=2.')

        self.name = 'Himmelblau'
        self.dim = 2

        self.obj = himmelblau
        self.bounds = ((-5., 5.), (-5., 5.))

        self.x_opt = (np.array([3., 2.]),
                      np.array([-2.805118, 3.131312]),
                      np.array([-3.779310, -3.283186]),
                      np.array([3.584428, -1.848126]))

        self.f_opt = (0., 0., 0., 0.)

        self.n_gopt = 4
        self.n_lopt = 0


class SixHumpCamelBack(TestFunction):
    def __init__(self, dim=2):
        super().__init__(dim)

        if dim != 2:
            warn('`dim` passed to SixHumpCamelBack is incorrect, continuing with dim=2.')

        self.name = 'Six Hump Camel Back'
        self.dim = 2

        self.obj = six_hump_camel_back
        self.bounds = ((-1.9, 1.9), (-1.1, 1.1))

        self.x_opt = (np.array([0.0898, -0.7126]),
                      np.array([-0.0898, 0.7126]))

        self.f_opt = (-1.0316, -1.0316)

        self.n_gopt = 2
        self.n_lopt = 5


class Shubert(TestFunction):
    def __init__(self, dim=2):
        super().__init__(dim)

        self.name = 'Shubert'

        self.obj = shubert
        self.bounds = tuple((-10., 10.) for _ in range(dim))

        self.f_opt = tuple([-186.7309088 for _ in range(dim * 3 ** dim)])

        self.n_gopt = dim * 3 ** dim
        self.n_lopt = 742  # TODO: in 2D only


class Vincent(TestFunction):
    def __init__(self, dim=2):
        super().__init__(dim)

        self.name = 'Vincent'

        self.obj = vincent
        self.bounds = tuple((0.25, 10.) for _ in range(dim))

        self.f_opt = tuple([-1. for _ in range(6 ** dim)])

        self.n_gopt = 6 ** dim
        self.n_lopt = 0


class ModifiedRastrigin(TestFunction):
    def __init__(self, dim=2):
        super().__init__(dim)

        self.name = 'Modified Rastrigin'

        self.obj = modified_rastrigin_all
        self.bounds = tuple((0., 1.) for _ in range(dim))

        self.f_opt = tuple([2. for _ in range(6 ** dim)])

        self.n_gopt = 12
        self.n_lopt = 0


# ======================================================================================================================
# TEST FUNCTION ITERABLES
# ======================================================================================================================
UNCON_QUAD_CONVEX = (Sphere, Ellipsoid, Cigar, Discus, CigarDiscus, TwoAxes)
UNCON_QUAD_NON_CONVEX = (DiffPowers, Rosenbrock)
UNCON_QUAD = (*UNCON_QUAD_CONVEX, *UNCON_QUAD_NON_CONVEX)
UNCON_NL = (Powell, Brown, Beale, HelicalValley, PowellSingular, Wood)
UNCON_ALL = (*UNCON_QUAD_CONVEX, *UNCON_QUAD_NON_CONVEX, *UNCON_NL)

EQ_NL = (G03, G11, G13)
INEQ_NL = (G01, G04, G06, G07, G08, G09, TR2, S240, S241, Himmelblau5D)  # , G02 , G10
EQ_INEQ_NL = (G05,)
CON_ALL = (*EQ_NL, *INEQ_NL, *EQ_INEQ_NL)

GLO_OPT_ALL = (Rastrigin, Ackley, LeviN13, Himmelblau, CrossInTray, Eggholder, HolderTable, SchafferN2, SchafferN4, StyblinskiTang)

OPTIM_ALL = (*UNCON_ALL, *CON_ALL, *GLO_OPT_ALL)
