import time
import numpy as np

from modpy.optimize.benchmarks.benchmark_plots import plot_convergence_single_algorithm, plot_unconstrained_2D,\
    plot_constrained_2D, plot_unconstrained_3D, plot_performance_comparison, plot_surfaces_2D
from modpy.optimize.benchmarks.benchmark_suite import UNCON_ALL, CON_ALL, UNCON_QUAD, GLO_OPT_ALL, UNCON_QUAD_CONVEX,\
    UNCON_QUAD_NON_CONVEX

from modpy.optimize import cma_es, nlprog


class Algorithm:
    def __init__(self, tol=1e-6, xtol=1e-4, maxiter=1000):

        self.name = ''
        self.algorithm = None
        self.stochastic = False
        self.kwargs = {}

        self.barrier = False
        self.cma = False

        self.tol = tol
        self.xtol = xtol
        self.maxiter = maxiter

    def run(self, test, x0):
        res = self.algorithm(**test.get(x0, self.stochastic), **self.kwargs, tol=self.tol,
                             maxiter=self.maxiter, keep_path=True)

        if not test.check(res, self.tol, self.xtol):
            res.success = False

        return res


class NL_SQP(Algorithm):
    def __init__(self, tol=1e-6, xtol=1e-4, maxiter=1000):
        super().__init__(tol=tol, xtol=xtol, maxiter=maxiter)

        self.name = 'SQP'
        self.kwargs = {'method': 'SQP'}

        self.algorithm = nlprog


class NL_IP(Algorithm):
    def __init__(self, tol=1e-6, xtol=1e-4, maxiter=1000):
        super().__init__(tol=tol, xtol=xtol, maxiter=maxiter)

        self.name = 'Interior Point'
        self.barrier = True
        self.kwargs = {'method': 'IP'}

        self.algorithm = nlprog


class CMA_ES_mu_lam(Algorithm):
    def __init__(self, tol=1e-6, xtol=1e-4, maxiter=None):
        super().__init__(tol=tol, xtol=xtol, maxiter=maxiter)

        self.name = '(mu-lam)-CMA-ES'
        self.cma = True
        self.stochastic = True
        self.kwargs = {'method': 'mu-lam',
                       'seed': 1234}

        self.algorithm = cma_es


class CMA_ES_IPOP(Algorithm):
    def __init__(self, tol=1e-6, xtol=1e-4, maxiter=None):
        super().__init__(tol=tol, xtol=xtol, maxiter=maxiter)

        self.name = 'IPOP-CMA-ES'
        self.cma = True
        self.stochastic = True
        self.kwargs = {'method': 'IPOP',
                       'seed': 1234}

        self.algorithm = cma_es


class CMA_ES_1p1(Algorithm):
    def __init__(self, tol=1e-6, xtol=1e-4, maxiter=None):
        super().__init__(tol=tol, xtol=xtol, maxiter=maxiter)

        self.name = '(1+1)-CMA-ES'
        self.cma = True
        self.stochastic = True
        self.kwargs = {'method': '1+1'}

        self.algorithm = cma_es


def run_test_suite(algorithm, suite, dim=2, x0=None, bounds=None):

    print('Testing algorithm: {}'.format(algorithm.name))

    results = []

    for test in suite:

        if bounds:
            test_init = test(dim=dim, bounds=bounds)
        else:
            test_init = test(dim=dim)

        print('Testing function: {}'.format(test_init.name))

        res = algorithm.run(test_init, x0)

        if not res.success:
            print('Failed to converge: {}'.format(res.message))

        results.append((test_init, res))

    return tuple(results)


def run_test_performance(algorithms, suite, dim=2, loops=20, bounds=None):

    results = {alg.name: None for alg in algorithms}
    x0s = [np.random.randn(dim) for _ in range(loops)]
    tests = [test(dim=dim, bounds=bounds) if bounds else test(dim=dim) for test in suite]

    for algorithm in algorithms:

        print('Testing algorithm: {}'.format(algorithm.name))

        duration = [None for _ in suite]
        converged = [None for _ in suite]
        iterations = [None for _ in suite]
        status = [None for _ in suite]

        for i, test in enumerate(tests):

            print('Testing function: {}'.format(test.name))

            dur = 0.
            conv = 0
            it = 0
            sta = [None for _ in range(loops)]

            for j in range(loops):

                t0 = time.time()
                res = algorithm.run(test, x0s[j])
                t1 = time.time()

                dur += t1 - t0
                conv += res.success
                it += res.nit
                sta[j] = res.status

            duration[i] = dur / loops
            converged[i] = conv / loops
            iterations[i] = it / loops

            status[i] = np.bincount(np.abs(np.array(sta) - 1)).argmax()

        results[algorithm.name] = (duration, converged, iterations, status)

    return results, tuple(tests)


def test_nl_opt_conv(dim, x0, tol, xtol, maxiter):
    print('Testing NL OPT convergence.')

    nl_sqp = NL_SQP(tol, xtol, maxiter)
    nl_ip = NL_IP(tol, xtol, maxiter)
    cma_es_ml = CMA_ES_mu_lam(tol, xtol, None)
    cma_es_ipop = CMA_ES_IPOP(tol, xtol, None)
    cma_es_1p1 = CMA_ES_1p1(tol, xtol, None)

    # unconstrained optimization
    alg_uncon = (nl_sqp, cma_es_ml, cma_es_ipop)

    for algorithm in alg_uncon:
        results = run_test_suite(algorithm, UNCON_ALL, dim, x0)
        plot_convergence_single_algorithm(algorithm, results)
        plot_unconstrained_2D(algorithm, results, identifier='unconstrained')
        plot_unconstrained_3D(algorithm, results)

    # bounded optimization
    alg_bound = (nl_sqp, nl_ip, cma_es_ml, cma_es_ipop)
    bounds = tuple([(-.3, 3.) for _ in range(dim)])

    for algorithm in alg_bound:
        results = run_test_suite(algorithm, UNCON_QUAD, dim, x0, bounds=bounds)
        plot_convergence_single_algorithm(algorithm, results, bounded=True)
        plot_unconstrained_2D(algorithm, results, bounded=True, identifier='bounded')

    # equality and inequality optimization
    alg_con = (nl_sqp, nl_ip)

    for algorithm in alg_con:
        results = run_test_suite(algorithm, CON_ALL, dim, x0)
        plot_convergence_single_algorithm(algorithm, results, constrained=True)
        plot_constrained_2D(algorithm, results)

    # global optimization (bounded)
    alg_glo = (nl_sqp, nl_ip, cma_es_ml, cma_es_ipop)

    for algorithm in alg_glo:
        results = run_test_suite(algorithm, GLO_OPT_ALL, dim, x0)
        #plot_convergence_single_algorithm(algorithm, results, bounded=True)
        plot_unconstrained_2D(algorithm, results, bounded=True, identifier='global')


def test_nl_opt_perm(dim, loops, tol, xtol, maxiter):
    print('Testing NL OPT performance.')

    nl_sqp = NL_SQP(tol, xtol, maxiter)
    nl_ip = NL_IP(tol, xtol, maxiter)
    cma_es_ml = CMA_ES_mu_lam(tol, xtol, None)
    cma_es_ipop = CMA_ES_IPOP(tol, xtol, None)
    cma_es_1p1 = CMA_ES_1p1(tol, xtol, None)

    alg_uncon = (nl_sqp, cma_es_ml, cma_es_ipop)
    results, tests = run_test_performance(alg_uncon, UNCON_QUAD, dim=dim, loops=loops, bounds=None)
    plot_performance_comparison(alg_uncon, tests, results, dim, fig_name='unconstrained')

    alg_bound = (nl_sqp, nl_ip, cma_es_ml, cma_es_ipop)
    bounds = tuple([(-.3, 3.) for _ in range(dim)])
    results, tests = run_test_performance(alg_bound, UNCON_QUAD, dim=dim, loops=loops, bounds=bounds)
    plot_performance_comparison(alg_bound, tests, results, dim, fig_name='bounded')


if __name__ == '__main__':

    # test settings
    dim_ = 2
    loops_ = 20
    xtol_ = 1e-1
    tol_ = 1e-6
    maxiter_ = 1000

    x0_ = np.array([-5., 5.])#np.random.randn(dim_)

    test_nl_opt_conv(dim_, x0_, tol_, xtol_, maxiter_)
    #test_nl_opt_perm(dim_, loops_, tol_, xtol_, maxiter_)

    # if dim_ == 2:
    #     convex = tuple([test(dim_) for test in UNCON_QUAD_CONVEX])
    #     plot_surfaces_2D(convex, 'Convex')
    #
    #     quad = tuple([test(dim_) for test in UNCON_QUAD_NON_CONVEX])
    #     plot_surfaces_2D(quad, 'Quadratic')
    #
    #     glob = tuple([test(dim_) for test in GLO_OPT_ALL])
    #     plot_surfaces_2D(glob, 'Global')
