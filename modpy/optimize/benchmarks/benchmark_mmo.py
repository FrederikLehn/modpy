import numpy as np

from modpy.random import UniformDist, JointDistribution
from modpy.optimize.benchmarks.benchmark_suite import FiveUnevenPeakTrap, EqualMaxima, UnevenDecreasingMaxima,\
    Himmelblau, SixHumpCamelBack, ModifiedRastrigin, CrossInTray, HolderTable, Shubert, Vincent

from modpy.optimize import mmo, nlprog, cma_es
from modpy.optimize.benchmarks.benchmark_plots import plot_convergence_mmo, plot_mixed_dimensions


class AlgorithmMMO:
    def __init__(self, tol=1e-6, xtol=1e-4, maxiter=100):

        self.name = ''
        self.opt = None  # core optimizer
        self.stochastic = False
        self.kwargs = {}
        self.opt_kwargs = {}

        self.tol = tol
        self.xtol = xtol
        self.maxiter = maxiter

        self.seed = None

    def run(self, test):

        if self.stochastic:

            # define core optimizer
            def opt_core(obj_, x0_, bounds, constraints, sigma0=1., C0=None):
                return self.opt(obj_, x0_, bounds=bounds, constraints=constraints, sigma0=sigma0, C0=C0, tol=self.tol, **self.opt_kwargs)

        else:

            # define core optimizer
            def opt_core(obj_, x0_, bounds, constraints, sigma0=1., C0=None):
                return self.opt(obj_, x0_, bounds=bounds, constraints=constraints, tol=self.tol, **self.opt_kwargs)

        # define population sampler
        marginals = [UniformDist(*bound, seed=self.seed) for bound in test.bounds]
        population = JointDistribution(marginals)

        def pop(N):
            return population.sample(N)

        # if isinstance(test.f_opt, float):
        #     lbound = test.f_opt * 0.5
        # else:
        #     lbound = test.f_opt[0] * 0.5
        #
        # if lbound == 0:
        #     lbound += 1.

        lbound = test.get_lbound()

        # run MMO
        res = mmo(test.obj, opt_core, pop, test.dim, bounds=test.bounds, **self.kwargs, lbound=lbound, maxiter=self.maxiter, sigma0=test.start_deviation())

        #print('Before', len(res.results), [r.x for r in res.results])

        success, results, track = test.check_mmo(res, self.tol, self.xtol)
        res.success = success
        res.results = results
        res.track = track

        #print([r.f for r in res.results])
        #print('AFTER', len(res.results), [r.x for r in res.results])

        return res


class MMO_Hill_SQP(AlgorithmMMO):
    def __init__(self, tol=1e-6, xtol=1e-4, maxiter=1000):
        super().__init__(tol=tol, xtol=xtol, maxiter=maxiter)

        self.name = 'Hill-Valley (SQP)'
        self.kwargs = {'method': 'hill'}
        self.opt_kwargs = {'method': 'SQP'}

        self.opt = nlprog


class MMO_Hill_IP(AlgorithmMMO):
    def __init__(self, tol=1e-6, xtol=1e-4, maxiter=1000):
        super().__init__(tol=tol, xtol=xtol, maxiter=maxiter)

        self.name = 'Hill-Valley (IP)'
        self.kwargs = {'method': 'hill'}
        self.opt_kwargs = {'method': 'IP'}

        self.opt = nlprog


class MMO_Hill_CMA(AlgorithmMMO):
    def __init__(self, tol=1e-6, xtol=1e-4, maxiter=1000):
        super().__init__(tol=tol, xtol=xtol, maxiter=maxiter)

        self.name = 'Hill-Valley (CMA-ES)'
        self.stochastic = True
        self.kwargs = {'method': 'hill'}
        self.opt_kwargs = {'method': 'IPOP'}

        self.opt = cma_es


def _average_track(test, results, maxN):
    track = np.zeros((test.n_gopt + 2, 2))

    for i in range(1, test.n_gopt + 1):
        track[i, 0] = np.mean([res.track[i-1] if len(res.track) >= i else maxN for res in results])
        track[i, 1] = np.sum([1. for res in results if len(res.track) >= i])

    track[:, 1] = np.cumsum(track[:, 1]) / len(results) / test.n_gopt
    track[-1, 0] = maxN
    track[-1, 1] = track[-2, 1]

    # insert zero at first jump
    track = np.insert(track, 1, [track[1, 0], 0], axis=0)

    return track


def _max_samples(results):
    return np.amax([r.track[-1] for res in results for r in res if len(r.track)])


def run_test_suite(algorithm, suite, repeats):

    print('Testing algorithm: {}'.format(algorithm.name))

    results = []

    for test in suite:

        print('Testing function: {}'.format(test.name))

        results_repeat = []

        for _ in range(repeats):
            res = algorithm.run(test)

            if not res.success:
                print('Failed to converge: {}'.format(res.message))

            print(test.name, ':', _)

            results_repeat.append(res)

        results.append(results_repeat)

    return tuple(results)


def test_mmo_opt_conv(suite, tol, xtol, maxiter, repeats):
    hill_sqp = MMO_Hill_SQP(tol, xtol, maxiter)
    hill_ip = MMO_Hill_IP(tol, xtol, maxiter)
    hill_cma = MMO_Hill_CMA(tol, xtol, maxiter)

    algorithms = (hill_cma, hill_sqp, hill_ip)

    results = []
    for algorithm in algorithms:
        results.append(run_test_suite(algorithm, suite, repeats))

    # replace track by averaged track over repeats
    maxN = _max_samples([r for res in results for r in res])
    algorithm_tracks = []

    for (algorithm, result) in zip(algorithms, results):

        tracks = []

        for test, results_repeat in zip(suite, result):
            track = _average_track(test, results_repeat, maxN)
            tracks.append(track)

        algorithm_tracks.append(tracks)

    plot_convergence_mmo(algorithms, algorithm_tracks, suite, maxN, global_only=True)


if __name__ == '__main__':
    tol_ = 1e-5
    xtol_ = 1e-1
    maxiter_ = 1000
    repeats_ = 20

    suite_ = (EqualMaxima(), UnevenDecreasingMaxima(), Himmelblau(), SixHumpCamelBack(),
              ModifiedRastrigin(), CrossInTray(), HolderTable(), Vincent(2), Vincent(3),)# Shubert(2), Shubert(3))

    test_mmo_opt_conv(suite_, tol_, xtol_, maxiter_, repeats_)

    # suite_ = (FiveUnevenPeakTrap(), EqualMaxima(), UnevenDecreasingMaxima(), Himmelblau(), SixHumpCamelBack(),
    #          ModifiedRastrigin(), CrossInTray(), HolderTable(), Vincent(2))
    #
    # plot_mixed_dimensions(suite_, identifier='MMO')
