import numpy as np

from modpy._util import split_set, convert_input, convert_output
from modpy.special._special_util import _poly_eval, _poly_eval1, EXP_NEG2


def erf(x):
    """

    NOTE:
    # from https://stackoverflow.com/questions/457408/is-there-an-easily-available-implementation-of-erf-for-python
    # answer by John D. Cook


    """

    # save the sign of x
    sign = np.where(x >= 0., 1., -1.)
    x_ = abs(x)

    # constants
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911

    # A&S formula 7.1.26
    t = 1.0 / (1.0 + p * x_)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x_ ** 2.)
    return sign * y


def erfinv(z):
    """
    Calculates the inverse of the error function, i.e.::

        x = erf**-1(z)

    NOTE:
    # inspiration from https://github.com/dougthor42/PyErf/blob/master/pyerf/pyerf.py
    # re-written to interact with numpy

    Parameters
    ----------
    z : float or array_like, shape (n,)
        Variable to be evaluated.

    Returns
    -------
    x : float array_like, shape (n,)
        Resulting value.
    """

    z = convert_input(z)

    if np.any((z < -1.) & (1. < z)):
        raise ValueError('All values of `z` must satisfy in -1 <= z <= 1.')

    x = _ndtri((z + 1.) / 2.) / np.sqrt(2.)

    return convert_output(z, x)


def _ndtri(y):
    """
    Port of cephes ``ndtri.c``: inverse normal distribution function.
    See https://github.com/jeremybarnes/cephes/blob/master/cprob/ndtri.c

    NOTE:
    # inspiration from https://github.com/dougthor42/PyErf/blob/master/pyerf/pyerf.py
    # re-written to interact with numpy
    """

    # approximation for 0 <= abs(z - 0.5) <= 3/8
    P0 = np.array([
        -5.99633501014107895267E1,
        9.80010754185999661536E1,
        -5.66762857469070293439E1,
        1.39312609387279679503E1,
        -1.23916583867381258016E0
    ])

    Q0 = np.array([
        1.95448858338141759834E0,
        4.67627912898881538453E0,
        8.63602421390890590575E1,
        -2.25462687854119370527E2,
        2.00260212380060660359E2,
        -8.20372256168333339912E1,
        1.59056225126211695515E1,
        -1.18331621121330003142E0,
    ])

    # Approximation for interval z = sqrt(-2 log y ) between 2 and 8
    # i.e., y between exp(-2) = .135 and exp(-32) = 1.27e-14.
    P1 = np.array([
        4.05544892305962419923E0,
        3.15251094599893866154E1,
        5.71628192246421288162E1,
        4.40805073893200834700E1,
        1.46849561928858024014E1,
        2.18663306850790267539E0,
        -1.40256079171354495875E-1,
        -3.50424626827848203418E-2,
        -8.57456785154685413611E-4,
    ])

    Q1 = np.array([
        1.57799883256466749731E1,
        4.53907635128879210584E1,
        4.13172038254672030440E1,
        1.50425385692907503408E1,
        2.50464946208309415979E0,
        -1.42182922854787788574E-1,
        -3.80806407691578277194E-2,
        -9.33259480895457427372E-4,
    ])

    # Approximation for interval z = sqrt(-2 log y ) between 8 and 64
    # i.e., y between exp(-32) = 1.27e-14 and exp(-2048) = 3.67e-890.
    P2 = np.array([
        3.23774891776946035970E0,
        6.91522889068984211695E0,
        3.93881025292474443415E0,
        1.33303460815807542389E0,
        2.01485389549179081538E-1,
        1.23716634817820021358E-2,
        3.01581553508235416007E-4,
        2.65806974686737550832E-6,
        6.23974539184983293730E-9,
    ])

    Q2 = np.array([
        6.02427039364742014255E0,
        3.67983563856160859403E0,
        1.37702099489081330271E0,
        2.16236993594496635890E-1,
        1.34204006088543189037E-2,
        3.28014464682127739104E-4,
        2.89247864745380683936E-6,
        6.79019408009981274425E-9,
    ])

    shape = y.shape
    y_ = y.flatten()
    sign = np.ones_like(y_)
    i = np.argwhere(y > (1. - EXP_NEG2))
    sign[i] = 0
    y_[i] = 1. - y_[i]

    # OMITTED AS SEARCHING FOR INDEXES WOULD RUIN PERFORMANCE
    # Shortcut case where high precision is not needed
    # between -0.135 and 0.135
    # if y > EXP_NEG2:
    #     y -= 0.5
    #     y2 = y ** 2
    #     x = y + y * (y2 * _poly_eval(y2, P0) / _poly_eval1(y2, Q0))
    #     x = x * ROOT_2PI
    #     return x

    x = np.sqrt(-2. * np.log(y_))
    x0 = x - np.log(x) / x

    z = 1. / x

    i, i_ = split_set(x < 8.)  # y > exp(-32) = 1.2664165549e-14

    x1 = np.zeros_like(z)
    x1[i] = (z * _poly_eval(z, P1) / _poly_eval1(z, Q1))[i]
    x1[i_] = (z * _poly_eval(z, P2) / _poly_eval1(z, Q2))[i_]

    x = x0 - x1
    x = np.where(sign, -x, x)

    x = np.where(y == 0., -np.inf, np.where(y == 1., np.inf, x))
    return np.reshape(x, shape)
