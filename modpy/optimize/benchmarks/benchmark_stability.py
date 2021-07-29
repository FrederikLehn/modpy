import time
import numpy as np

from modpy.optimize.benchmarks.benchmark_suite import QUAD_CONVEX, QUAD_NON_CONVEX
from modpy.optimize import cma_es, nlprog


class BenchmarkStability:
    def __init__(self, name='', algorithm=None, repeats=100):

        self.name = name            # name of algorithm (for plotting/saving)
        self.algorithm = algorithm  # callable, algorithm(fun, x0)
        self.repeats = repeats      # int, number of times to repeat test with new start-guess

        self.time = {}         # int, time [s] spent to terminate
        self.converged = {}    # bool, did the algorithm converge (to global or local optimum)
        self.status = {}       # int,
        self.iterations = {}   # int, number of iterations spent to converge

    def run(self, dim):
        print('RUNNING BENCHMARKS FOR: {}'.format(self.name))

        self.run_quad_convex(dim)
        self.run_quad_non_convex(dim)

    def run_quad_convex(self, dim):

        print('RUNNING QUADRATIC CONVEX TESTS')

        for key, fun in QUAD_CONVEX.items():

            print('RUNNING FUNCTION: {}'.format(str(key).capitalize()))

            self._pre_allocate(key)

            for i in range(self.repeats):

                x0 = np.random.randn(dim)

                t0 = time.time()
                res = self.algorithm(fun, x0)
                t1 = time.time()

                self._append_result(key, res, t1 - t0)

                if not res.success:
                    print('FAILED TO CONVERGE AT ITERATION: {}'.format(i))

    def run_quad_non_convex(self, dim):

        print('RUNNING QUADRATIC NON-CONVEX TESTS')

        for key, fun in QUAD_NON_CONVEX.items():

            print('RUNNING FUNCTION: {}'.format(str(key).capitalize()))

            self._pre_allocate(key)

            for i in range(self.repeats):

                x0 = np.random.randn(dim)

                t0 = time.time()
                res = self.algorithm(fun, x0)
                t1 = time.time()

                self._append_result(key, res, t1 - t0)

                if not res.success:
                    print('FAILED TO CONVERGE AT ITERATION: {}'.format(i))

    def _pre_allocate(self, key):
        # pre-allocate containers for saving results
        self.time[key] = []
        self.converged[key] = []
        self.status[key] = []
        self.iterations[key] = []

    def _append_result(self, key, res, dt):
        self.time[key].append(dt)
        self.converged[key].append(res.success)
        self.status[key].append(res.status)
        self.iterations[key].append(res.nit)


if __name__ == '__main__':

    # test SQP
    def sqp_wrap(fun, x0):
        return nlprog(fun, x0)

    seq_quad = BenchmarkStability('SQP', sqp_wrap)
    seq_quad.run(3)

    # test cma-es (mu/muw-lambda)
    def cma_es_wrap1(fun, x0):
        return cma_es(fun, x0, method='mu-lam', sigma0=30., maxit=20000)

    cma_es_mu_lam = BenchmarkStability('CMA-ES (mu-lam)', cma_es_wrap1)
    cma_es_mu_lam.run(3)

    # test cma-es (1+1)
    def cma_es_wrap2(fun, x0):
        return cma_es(fun, x0, method='1+1', sigma0=3., maxit=20000)

    cma_es_1p1 = BenchmarkStability('CMA-ES (1+1)', cma_es_wrap2)
    #cma_es_1p1.run(3)
