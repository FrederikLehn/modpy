import numpy as np
from numpy.random import Generator, PCG64

from modpy import scale_translate

from modpy.special import EXP, log

from modpy.random._random_util import _chk_mmm_inp, _chk_log_mmm_inp, _chk_root_mmm_inp, _chk_normal_inp,\
    _chk_beta_inp

from modpy.random._normal import normal_pdf, normal_cdf, normal_ppf, lognormal_pdf, lognormal_cdf, lognormal_ppf,\
    rootnormal_pdf, rootnormal_cdf, rootnormal_ppf, logn2n_par, sqrtn2n_par

from modpy.random._normal import trunc_normal_pdf, trunc_normal_cdf, trunc_normal_ppf, trunc_normal_sample,\
    trunc_lognormal_pdf, trunc_lognormal_cdf, trunc_lognormal_ppf, trunc_rootnormal_pdf, trunc_rootnormal_cdf,\
    trunc_rootnormal_ppf, _tn2tstdn_par

from modpy.random._uniform import uniform_pdf, uniform_cdf, uniform_ppf, loguniform_pdf, loguniform_cdf,\
    loguniform_ppf, rootuniform_pdf, rootuniform_cdf, rootuniform_ppf

from modpy.random._triangular import triangular_pdf, triangular_cdf, triangular_ppf, logtriangular_pdf, logtriangular_cdf,\
    logtriangular_ppf, roottriangular_pdf, roottriangular_cdf, roottriangular_ppf

from modpy.random._exponential import exponential_pdf, exponential_cdf, exponential_ppf

from modpy.random._gamma import gamma_pdf, gamma_cdf, gamma_ppf

from modpy.random._beta import beta_pdf, beta_cdf, beta_ppf, logbeta_pdf, logbeta_cdf,\
    logbeta_ppf, rootbeta_pdf, rootbeta_cdf, rootbeta_ppf

from modpy.random._transform import lin2log, lin2root


class JointDistribution:
    def __init__(self, marginals=(), correlation=None, copula=None):

        self.marginals = marginals
        self.correlation = correlation
        self.copula = copula

    def pdf(self, x):
        # TODO: account for correlation
        return np.prod([m.pdf(x) for m in self.marginals])

    def log_pdf(self, x):
        # TODO: account for correlation
        return np.sum(np.log([m.pdf(x) for m in self.marginals]))

    def sample(self, size):
        # TODO: account for correlation through copula
        N = np.prod(size)
        samples = np.hstack(tuple([d.sample(N) for d in self.marginals]))

        m = len(self.marginals)
        if isinstance(size, int):
            rs = (size, m)
        else:
            rs = (*size, m)

        return np.reshape(samples, rs)


class Distribution:
    def __init__(self, param=(), input_=(), bounds=(), seed=None):

        self._gen = Generator(PCG64(seed))

        self._param = param

        if not input_:
            input_ = param  # input is in most cases equal to param

        self._input = input_
        self._bounds = bounds

    def cdf(self, x):
        raise NotImplementedError

    def reference(self):
        """
        Calculates the reference/default value for a given distribution. One of three values are picked, namely the
        mode, median or mean. With the priority being mode > median > mean.
        In case none of them are defined it will use the average of the support. If this fails (due to +-inf or similar)
        an error is thrown.

        Returns
        -------
        reference : float
            Reference/default value of the distribution
        """

        ref = None
        for f in (self.mode, self.median, self.mean):

            try:
                ref = f()
                break

            except NotImplementedError:
                pass

        if ref is None:
            a, b = self._bounds
            ref = (a + b) / 2.

        try:

            ref = float(ref)

        except ValueError:

            raise ValueError('Neither mode, median or mean are implemented for this distribution'
                             ' and the support is not finite.')

        return ref

    def ex_kurtosis(self):
        raise NotImplementedError

    def get_bounds(self):
        return self._bounds

    def get_input(self):
        return self._input

    def get_parameters(self):
        return self._param

    def kurtosis(self):
        return 3. + self.ex_kurtosis()

    def maximum(self):
        return self._bounds[1]

    def mean(self):
        raise NotImplementedError

    def median(self):
        return float(self.ppf(0.5))

    def minimum(self):
        return self._bounds[0]

    def mode(self):
        raise NotImplementedError

    def pdf(self, x):
        raise NotImplementedError

    def ppf(self, p):
        raise NotImplementedError

    def sample(self, size):
        raise NotImplementedError

    def skewness(self):
        raise NotImplementedError

    def std(self):
        return np.sqrt(self.variance())

    def variance(self):
        raise NotImplementedError


class NormalDist(Distribution):
    def __init__(self, mu=0., sigma=1., bounds=(-np.inf, np.inf), seed=None):

        _chk_normal_inp(mu, sigma)
        super().__init__((mu, sigma), bounds=bounds, seed=seed)

    def cdf(self, x):
        return normal_cdf(x, *self._param, self._bounds)

    def ex_kurtosis(self):
        return 0.

    def mean(self):
        return self._param[0]

    def median(self):
        return self._param[0]

    def mode(self):
        return self._param[0]

    def pdf(self, x):
        return normal_pdf(x, *self._param, self._bounds)

    def ppf(self, p):
        return normal_ppf(p, *self._param)

    def sample(self, size):
        return self._gen.normal(*self._param, size)

    def skewness(self):
        return 0.

    def variance(self):
        return self._param[1] ** 2.


class LogNormalDist(Distribution):
    def __init__(self, mu=0., sigma=1., bounds=(-np.inf, np.inf), base=EXP, seed=None):

        _chk_normal_inp(mu, sigma)
        m, sd = logn2n_par(mu, sigma, base=base)
        self._base = base

        super().__init__((m, sd), (mu, sigma), bounds=bounds, seed=seed)

    def cdf(self, x):
        return lognormal_cdf(x, *self._param, self._bounds, base=self._base)

    def ex_kurtosis(self):
        # TODO: WRONG - Function of base
        mu, sigma = self._param
        return np.exp(4. * sigma ** 2.) + 2 * np.exp(3. * sigma ** 2.) + 3. * np.exp(2. * sigma ** 2.) - 6.

    def mean(self):
        return self._base ** (self._param[0] + self._param[1] ** 2. / 2.)

    def median(self):
        return self._base ** self._param[0]

    def mode(self):
        return self._base ** (self._param[0] - self._param[1] ** 2.)

    def pdf(self, x):
        return lognormal_pdf(x, *self._param, self._bounds, base=self._base)

    def ppf(self, p):
        return lognormal_ppf(p, *self._param, base=self._base)

    def sample(self, size):
        return self._base ** self._gen.normal(*self._param, size)

    def skewness(self):
        # TODO: WRONG - Function of base
        mu, sigma, = self._param
        return (np.exp(sigma ** 2.) + 2) * np.sqrt(np.exp(sigma ** 2.) - 1.)

    def variance(self):
        # TODO: WRONG - Function of base
        mu, sigma = self._param
        return (np.exp(sigma ** 2.) - 1.) * np.exp(2. * mu + sigma ** 2.)


class RootNormalDist(Distribution):
    def __init__(self, mu=0., sigma=1., bounds=(-np.inf, np.inf), seed=None):

        _chk_normal_inp(mu, sigma)
        m, sd = sqrtn2n_par(mu, sigma)

        super().__init__((m, sd), (mu, sigma), bounds=bounds, seed=seed)

    def cdf(self, x):
        return rootnormal_cdf(x, *self._param, self._bounds)

    def ex_kurtosis(self):
        raise NotImplementedError

    def mean(self):
        mu, sigma = self._param
        return mu ** 2. + sigma ** 2.

    def median(self):
        return self._param[0] ** 2.

    def mode(self):
        raise NotImplementedError

    def pdf(self, x):
        return rootnormal_pdf(x, *self._param, self._bounds)

    def ppf(self, p):
        return rootnormal_ppf(p, *self._param)

    def sample(self, size):
        return self._gen.normal(*self._param, size) ** 2.

    def skewness(self):
        raise NotImplementedError

    def variance(self):
        mu, sigma = self._param
        return 2 * sigma ** 4. + 4. * mu ** 2. * sigma ** 2.


class TruncatedNormalDist(Distribution):
    def __init__(self, mu=0., sigma=1., a=-np.inf, b=np.inf, bounds=(-np.inf, np.inf), seed=None):

        _chk_normal_inp(mu, sigma)
        _chk_mmm_inp(a, b)

        if not bounds:
            bounds = (a, b)

        super().__init__((mu, sigma, a, b), bounds=bounds, seed=seed)

    def cdf(self, x):
        return trunc_normal_cdf(x, *self._param, self._bounds)

    def ex_kurtosis(self):
        return 0.

    def mean(self):
        mu, sigma, a, b = self._param
        alpha, beta_ = _tn2tstdn_par(mu, sigma, a, b)

        return mu + (normal_pdf(alpha) - normal_pdf(beta_)) / (normal_cdf(beta_) - normal_cdf(alpha)) * sigma

    def median(self):
        mu, sigma, a, b = self._param
        alpha, beta_ = _tn2tstdn_par(mu, sigma, a, b)

        return mu + normal_ppf((normal_cdf(alpha) + normal_cdf(beta_)) / 2.) * sigma

    def mode(self):
        mu, _, a, b = self._param
        if mu < a:
            return a
        elif a <= mu <= b:
            return mu
        else:
            return b

    def pdf(self, x):
        return trunc_normal_pdf(x, *self._param, self._bounds)

    def ppf(self, p):
        return trunc_normal_ppf(p, *self._param)

    def sample(self, size):
        return trunc_normal_sample(*self._param, size=size, gen=self._gen)

    def skewness(self):
        return 0.

    def variance(self):
        mu, sigma, a, b = self._param
        alpha, beta_ = _tn2tstdn_par(mu, sigma, a, b)

        phi_alpha = normal_pdf(alpha)
        phi_beta = normal_pdf(beta_)
        Z = normal_cdf(beta_) - normal_cdf(alpha)

        return sigma ** 2. * (1. + (alpha * phi_alpha - beta_ * phi_beta) / Z - ((phi_alpha - phi_beta) / Z) ** 2.)


class TruncatedLogNormalDist(Distribution):
    def __init__(self, mu, sigma, a, b, bounds=(-np.inf, np.inf), base=EXP, seed=None):

        _chk_normal_inp(mu, sigma)
        _chk_mmm_inp(a, b)

        m, sd = logn2n_par(mu, sigma, base=base)

        if not bounds:
            bounds = (a, b)

        self._base = base

        super().__init__((m, sd, a, b), (mu, sigma, a, b), bounds=bounds, seed=seed)

    def cdf(self, x):
        return trunc_lognormal_cdf(x, *self._param, self._bounds)

    def ex_kurtosis(self):
        raise NotImplementedError

    def mean(self):
        raise NotImplementedError

    def mode(self):
        raise NotImplementedError

    def pdf(self, x):
        return trunc_lognormal_pdf(x, *self._param, self._bounds)

    def ppf(self, p):
        return trunc_lognormal_ppf(p, *self._param)

    def sample(self, size):
        m, sd, a, b = self._param
        loga = log(a, base=self._base)
        logb = log(b, base=self._base)

        return self._base ** trunc_normal_sample(m, sd, loga, logb, size=size, gen=self._gen)

    def skewness(self):
        raise NotImplementedError

    def variance(self):
        raise NotImplementedError


class TruncatedRootNormalDist(Distribution):
    def __init__(self, mu, sigma, a, b, bounds=(-np.inf, np.inf), seed=None):

        _chk_normal_inp(mu, sigma)
        _chk_mmm_inp(a, b)

        m, sd = sqrtn2n_par(mu, sigma)

        if not bounds:
            bounds = (a, b)

        super().__init__((m, sd, a, b), (mu, sigma, a, b), bounds=bounds, seed=seed)

    def cdf(self, x):
        return trunc_rootnormal_cdf(x, *self._param, self._bounds)

    def ex_kurtosis(self):
        raise NotImplementedError

    def mean(self):
        raise NotImplementedError

    def mode(self):
        raise NotImplementedError

    def pdf(self, x):
        return trunc_rootnormal_pdf(x, *self._param, self._bounds)

    def ppf(self, p):
        return trunc_rootnormal_ppf(p, *self._param)

    def sample(self, size):
        m, sd, a, b = self._param
        sqrta = np.sqrt(a)
        sqrtb = np.sqrt(b)

        return trunc_normal_sample(m, sd, sqrta, sqrtb, size=size, gen=self._gen) ** 2.

    def skewness(self):
        raise NotImplementedError

    def variance(self):
        raise NotImplementedError


class UniformDist(Distribution):
    def __init__(self, a=0., b=1., bounds=(), seed=None):

        _chk_mmm_inp(a, b)

        if not bounds:
            bounds = (a, b)

        super().__init__((a, b), bounds=bounds, seed=seed)

    def cdf(self, x):
        return uniform_cdf(x, *self._param, self._bounds)

    def ex_kurtosis(self):
        return -6. / 5.

    def maximum(self):
        return np.minimum(self._param[1], self._bounds[1])

    def mean(self):
        a, b = self._param
        return (a + b) / 2.

    def median(self):
        a, b = self._param
        return (a + b) / 2.

    def minimum(self):
        return np.maximum(self._param[0], self._bounds[0])

    def mode(self):
        a, b = self._param
        return (a + b) / 2.  # any value in (a, b) is the mode, mean chosen as default

    def pdf(self, x):
        return uniform_pdf(x, *self._param, self._bounds)

    def ppf(self, p):
        return uniform_ppf(p, *self._param)

    def sample(self, size):
        return self._gen.uniform(*self._param, size)

    def skewness(self):
        return 0.

    def variance(self):
        a, b = self._param
        return 1. / 12. * (b - a) ** 2.


class LogUniformDist(Distribution):
    def __init__(self, a=0., b=1., bounds=(), seed=None):

        _chk_log_mmm_inp(a, b)

        if not bounds:
            bounds = (a, b)

        super().__init__((a, b), bounds=bounds, seed=seed)

    def ex_kurtosis(self):
        raise NotImplementedError

    def maximum(self):
        return np.minimum(self._param[1], self._bounds[1])

    def mean(self):
        a, b = self._param
        return (b - a) / np.log(b / a)

    def mode(self):
        return self._param[0]

    def minimum(self):
        return np.maximum(self._param[0], self._bounds[0])

    def pdf(self, x):
        return loguniform_pdf(x, *self._param, self._bounds)

    def cdf(self, x):
        return loguniform_cdf(x, *self._param, self._bounds)

    def ppf(self, p):
        return loguniform_ppf(p, *self._param)

    def sample(self, size):
        return self.ppf(self._gen.uniform(*(0., 1.), size))

    def skewness(self):
        raise NotImplementedError

    def variance(self):
        a, b = self._param
        return (b ** 2. - a ** 2.) / (2. * np.log(b / a)) - ((b - a) / np.log(b / a)) ** 2.


class RootUniformDist(Distribution):
    def __init__(self, a=0., b=1., bounds=(), root=2., seed=None):

        _chk_root_mmm_inp(a, b)

        if not bounds:
            bounds = (a, b)

        self._root = root

        super().__init__((a, b), bounds=bounds, seed=seed)

    def ex_kurtosis(self):
        raise NotImplementedError

    def maximum(self):
        return np.minimum(self._param[1], self._bounds[1])

    def mean(self):
        raise NotImplementedError

    def minimum(self):
        return np.maximum(self._param[0], self._bounds[0])

    def mode(self):
        return self._param[0]

    def pdf(self, x):
        return rootuniform_pdf(x, *self._param, self._bounds, self._root)

    def cdf(self, x):
        return rootuniform_cdf(x, *self._param, self._bounds, self._root)

    def ppf(self, p):
        return rootuniform_ppf(p, *self._param, self._root)

    def sample(self, size):
        return self.ppf(self._gen.uniform(*(0., 1.), size))

    def skewness(self):
        raise NotImplementedError

    def variance(self):
        raise NotImplementedError


class TriangularDist(Distribution):
    def __init__(self, a=0., b=1., c=None, bounds=(), seed=None):

        if c is None:
            c = (a + b) / 2.

        _chk_mmm_inp(a, b, c)

        if not bounds:
            bounds = (a, b)

        super().__init__((a, c, b), bounds=bounds, seed=seed)

    def ex_kurtosis(self):
        raise - 3. / 5.

    def maximum(self):
        return np.minimum(self._param[2], self._bounds[1])

    def mean(self):
        a, c, b = self._param
        return (a + c + b) / 3.

    def median(self):
        a, c, b = self._param

        if c >= (a + b) / 2.:
            return a + np.sqrt((b - a) * (c - a) / 2.)
        else:
            return b - np.sqrt((b - a) * (b - c) / 2.)

    def minimum(self):
        return np.maximum(self._param[0], self._bounds[0])

    def mode(self):
        return self._param[1]

    def pdf(self, x):
        return triangular_pdf(x, *self._param, self._bounds)

    def cdf(self, x):
        return triangular_cdf(x, *self._param, self._bounds)

    def ppf(self, p):
        return triangular_ppf(p, *self._param)

    def sample(self, size):
        return self._gen.triangular(*self._param, size)

    def skewness(self):
        a, c, b = self._param
        np.sqrt(2.) * (a + b - 2. * c) * (2. * a - b - c) * (a - 2. * b + c) / (5. * (self.variance() * 18.) ** (3. / 2.))

    def variance(self):
        a, c, b = self._param
        return (a ** 2. + b ** 2. + c ** 2. - a * b - a * c - b * c) / 18.


class LogTriangularDist(Distribution):
    def __init__(self, a=0., b=1., c=None, bounds=(), seed=None):

        if c is None:
            c = (a + b) / 2.

        _chk_log_mmm_inp(a, b, c)

        if not bounds:
            bounds = (a, b)

        super().__init__((a, c, b), bounds=bounds, seed=seed)

    def maximum(self):
        return np.minimum(self._param[2], self._bounds[1])

    def minimum(self):
        return np.maximum(self._param[0], self._bounds[0])

    def mode(self):
        return self._param[1]

    def pdf(self, x):
        return logtriangular_pdf(x, *self._param, self._bounds)

    def cdf(self, x):
        return logtriangular_cdf(x, *self._param, self._bounds)

    def ppf(self, p):
        return logtriangular_ppf(p, *self._param)

    def sample(self, size):
        return self.ppf(self._gen.uniform(0., 1., size))


class RootTriangularDist(Distribution):
    def __init__(self, a=0., b=1., c=None, bounds=(), root=2., seed=None):

        if c is None:
            c = (a + b) / 2.

        _chk_root_mmm_inp(a, b, c)

        if not bounds:
            bounds = (a, b)

        self._root = root

        super().__init__((a, c, b), bounds=bounds, seed=seed)

    def maximum(self):
        return np.minimum(self._param[2], self._bounds[1])

    def minimum(self):
        return np.maximum(self._param[0], self._bounds[0])

    def mode(self):
        return self._param[1]

    def pdf(self, x):
        return roottriangular_pdf(x, *self._param, self._bounds, self._root)

    def cdf(self, x):
        return roottriangular_cdf(x, *self._param, self._bounds, self._root)

    def ppf(self, p):
        return roottriangular_ppf(p, *self._param, self._root)

    def sample(self, size):
        return self.ppf(self._gen.uniform(0., 1., size))


class ExponentialDist(Distribution):
    def __init__(self, lam, bounds=(0., np.inf), seed=None):
        super().__init__((lam,), bounds=bounds, seed=seed)

    def ex_kurtosis(self):
        return 6.

    def mean(self):
        return 1. / self._param[0]

    def median(self):
        return np.log(2.) / self._param[0]

    def mode(self):
        return 0.

    def pdf(self, x):
        return exponential_pdf(x, *self._param, self._bounds)

    def cdf(self, x):
        return exponential_cdf(x, *self._param, self._bounds)

    def ppf(self, p):
        return exponential_ppf(p, *self._param)

    def sample(self, size):
        return self._gen.exponential(1. / self._param[0], size)

    def skewness(self):
        return 2.

    def variance(self):
        return 1. / self._param[0] ** 2.


class GammaDist(Distribution):
    def __init__(self, alpha=2., beta_=5, bounds=(0., np.inf), seed=None):
        super().__init__((alpha, beta_), bounds=bounds, seed=seed)

    def ex_kurtosis(self):
        return 6. / self._param[0]

    def mean(self):
        return self._param[0] / self._param[1]

    def mode(self):
        alpha, beta_ = self._param

        if alpha >= 1.:
            return (alpha - 1.) / beta_
        else:
            return 0.

    def pdf(self, x):
        return gamma_pdf(x, *self._param, self._bounds)

    def cdf(self, x):
        return gamma_cdf(x, *self._param, self._bounds)

    def ppf(self, p):
        return gamma_ppf(p, *self._param)

    def sample(self, size):
        alpha, beta_ = self._param
        return self._gen.gamma(alpha, 1. / beta_, size)

    def skewness(self):
        return 2. / np.sqrt(self._param[0])

    def variance(self):
        return self._param[0] / self._param[1] ** 2.


class BetaDist(Distribution):
    def __init__(self, alpha=2., beta_=5, a=0., b=1., bounds=(), seed=None):

        _chk_beta_inp(alpha, beta_)
        _chk_mmm_inp(a, b)

        if not bounds:
            bounds = (a, b)

        super().__init__((alpha, beta_, a, b), bounds=bounds, seed=seed)

    def maximum(self):
        return np.minimum(self._param[3], self._bounds[1])

    def mean(self):
        alpha, beta_, a, b = self._param
        m = alpha / (alpha + beta_)
        return scale_translate(m, a, b)

    def minimum(self):
        return np.maximum(self._param[2], self._bounds[0])

    def mode(self):
        alpha, beta_, a, b = self._param

        if (alpha > 1.) and (beta_ > 1.):

            m = (alpha - 1.) / (alpha + beta_ - 2.)
            return scale_translate(m, a, b)

        elif (alpha < 1.) and (beta_ < 1.):

            return a, b  # both min and max are modes

        elif ((alpha < 1.) and beta_ >= 1.) or ((alpha == 1. and beta_ > 1.)):

            return a

        elif ((alpha >= 1.) and (beta_ < 1.)) or ((alpha > 1.) and (beta_ == 1.)):

            return b

        else:

            raise ValueError('The mode is not uniquely defined for alpha=1 and beta=1.')

    def ex_kurtosis(self):
        alpha, beta_, _, _ = self._param
        return 6. * ((alpha - beta_) ** 2. * (alpha + beta_ + 1.) - alpha * beta_ * (alpha + beta_ + 2.)) / (alpha * beta_ * (alpha + beta_ + 2.) * (alpha + beta_ + 3.))

    def pdf(self, x):
        return beta_pdf(x, *self._param, self._bounds)

    def cdf(self, x):
        return beta_cdf(x, *self._param, self._bounds)

    def ppf(self, p):
        return beta_ppf(p, *self._param)

    def sample(self, size):
        return scale_translate(self._gen.beta(*self._param[:2], size), *self._param[2:])

    def skewness(self):
        # https://www.vosesoftware.com/riskwiki/Beta4distribution.php
        alpha, beta_, _, _ = self._param
        return 2. * (beta_ - alpha) / (alpha + beta_ + 2.) * np.sqrt((alpha + beta_ + 1.) / (alpha * beta_))

    def variance(self):
        # https://www.vosesoftware.com/riskwiki/Beta4distribution.php
        alpha, beta_, a, b = self._param
        return alpha * beta_ / ((alpha + beta_) ** 2. * (alpha + beta_ + 1.)) * (b - a) ** 2.


class LogBetaDist(Distribution):
    def __init__(self, alpha=2., beta_=5, a=0., b=1., bounds=(), seed=None):

        _chk_beta_inp(alpha, beta_)
        _chk_log_mmm_inp(a, b)

        if not bounds:
            bounds = (a, b)

        super().__init__((alpha, beta_, a, b), bounds=bounds, seed=seed)

    def maximum(self):
        return np.minimum(self._param[3], self._bounds[1])

    def minimum(self):
        return np.maximum(self._param[2], self._bounds[0])

    def pdf(self, x):
        return logbeta_pdf(x, *self._param, self._bounds)

    def cdf(self, x):
        return logbeta_cdf(x, *self._param, self._bounds)

    def ppf(self, p):
        return logbeta_ppf(p, *self._param)

    def sample(self, size):
        return self.ppf(self._gen.uniform(0., 1., size))


class RootBetaDist(Distribution):
    def __init__(self, alpha=2., beta_=5, a=0., b=1., bounds=(), root=2., seed=None):

        _chk_beta_inp(alpha, beta_)
        _chk_root_mmm_inp(a, b)

        if not bounds:
            bounds = (a, b)

        self._root = root

        super().__init__((alpha, beta_, a, b), bounds=bounds, seed=seed)

    def maximum(self):
        return np.minimum(self._param[3], self._bounds[1])

    def minimum(self):
        return np.maximum(self._param[2], self._bounds[0])

    def pdf(self, x):
        return rootbeta_pdf(x, *self._param, self._bounds, self._root)

    def cdf(self, x):
        return rootbeta_cdf(x, *self._param, self._bounds, self._root)

    def ppf(self, p):
        return rootbeta_ppf(p, *self._param, self._root)

    def sample(self, size):
        return self.ppf(self._gen.uniform(0., 1., size))


class PertDist(BetaDist):
    def __init__(self, a=0., b=1., c=None, bounds=(), seed=None):

        if c is None:
            c = (a + b) / 2.

        _chk_mmm_inp(a, b, c)

        # calculate beta parameters
        alpha = 1. + 4. * (c - a) / (b - a)
        beta_ = 1. + 4. * (b - c) / (b - a)

        super().__init__(alpha, beta_, a, b, bounds=bounds, seed=seed)
        self._input = (a, c, b)

    def mean(self):
        a, c, b = self._param
        return (a + 4. * c + b) / 6.

    def mode(self):
        return self._param[1]


class LogPertDist(LogBetaDist):
    def __init__(self, a=0., b=1., c=None, bounds=(), seed=None):

        if c is None:
            c = (a + b) / 2.

        _chk_log_mmm_inp(a, b, c)

        # calculate beta parameters
        alpha = 1. + 4. * (c - a) / (b - a)
        beta_ = 1. + 4. * (b - c) / (b - a)

        super().__init__(alpha, beta_, a, b, bounds=bounds, seed=seed)
        self._input = (a, c, b)

    def mode(self):
        return self._param[1]


class RootPertDist(RootBetaDist):
    def __init__(self, a=0., b=1., c=None, bounds=(), root=2., seed=None):

        if c is None:
            c = (a + b) / 2.

        _chk_root_mmm_inp(a, b, c)

        # calculate beta parameters
        alpha = 1. + 4. * (c - a) / (b - a)
        beta_ = 1. + 4. * (b - c) / (b - a)

        super().__init__(alpha, beta_, a, b, bounds=bounds, root=root, seed=seed)
        self._input = (a, c, b)

    def mode(self):
        return self._param[1]
