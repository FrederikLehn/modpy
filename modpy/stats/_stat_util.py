import numpy as np

from modpy.stats._core import auto_correlation_time


class MCMCResult:
    def __init__(self, x, f, samples, burn, success=False, status=0, message='', nit=0):

        # state-space
        self.x = x  # chain, array_like, shape (n, m)
        self.f = f  # likelihood, array_like, shape (n,)

        # statistical results
        self.mean = None            # mean
        self.std = None             # standard deviation
        self.var = None             # variance
        self.min = None             # minimum
        self.max = None             # maximum
        self.percentiles = None     # percentiles (multiple)
        self.tau = None             # auto-correlation time
        self.ess = None             # effective sample size

        # sampling
        self.samples = samples      # number of samples in chain
        self.burn = burn            # number of burned samples
        self.samples_eff = None     # number of effective samples

        # performance
        self.success = success  # true/false whether algorithm converged
        self.status = status    # termination cause (see individual MCMC sampler for description)
        self.message = message  # termination message
        self.nit = nit          # number of iterations performed

        # sampler path variables
        self.path = None

        # finalize
        self._finalize()

    def _finalize(self):
        self._calculate_statics()
        self._calculate_effectiveness()

    def _calculate_statics(self):
        self.mean = np.mean(self.x, axis=0)
        self.std = np.std(self.x, axis=0)
        self.var = self.std ** 2.
        self.min = np.amin(self.x, axis=0)
        self.max = np.amax(self.x, axis=0)
        self.percentiles = np.percentile(self.x, [2.5, 5., 10., 25., 50., 75., 90., 95., 97.5], axis=0)

    def _calculate_effectiveness(self):
        self.tau = np.array([auto_correlation_time(self.x[:, i]) for i in range(self.x.shape[1])])
        self.ess = self.x.shape[0] / self.tau
        self.samples_eff = (self.samples / self.ess).astype(np.int64)

    def get_thinned(self, tau=None):
        if tau is None:
            tau = np.amin(self.tau)

        return self.x[::tau], self.f[::tau]


class MCMCPath:
    def __init__(self, keep=False):

        self.keep = keep  # whether to keep results when functions are called

        self.accept = []        # acceptance rate
        self.steps = []         # integration steps (HMC)
        self.step_size = []     # step-size (HMC)

    def finalize(self):
        self.accept = np.array(self.accept)
        self.steps = np.array(self.steps, dtype=np.int64)
        self.step_size = np.array(self.step_size)

    def append(self, accept, steps=None, step_size=None):
        if not self.keep:
            return

        self.accept.append(float(accept))

        if steps is not None:
            self.steps.append(int(steps))

        if step_size is not None:
            self.step_size.append(float(step_size))

