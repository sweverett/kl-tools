import numpy as np
from multiprocessing import Pool

class MetropolisSampler(object):
    '''
    A simple Metropolis Hastings implementation for debugging
    purposes
    '''

    def __init__(self, nwalkers, ndims, log=True,
                 posterior=None, likelihood=None, prior=None,
                 post_args=None, post_kwargs=None,
                 like_args=None, like_kwargs=None,
                 prior_args=None, prior_kwargs=None,
                 pool=None):
        '''
        Must pass *either* a posterior or likelihood + prior

        nwalkers: int
            The number of walkers to initialize
        ndims: int
            The number of sampled parameters
        log: bool
            Whether the passed probability densities are their log versions
        pool: Pool
            The multiprocessing pool, if desired

        NOTE: The following are sets of callable functions & their args/kwargs
        needed to compute a (possibly log) probablility density. The user
        should provide *either* a posterior or likelihood + prior, not both
        {func} is one of {post, like, prior}

        {func}: function or callable()
            Callable function to sample from the posterior, likelihood,
            or prior
        {func}_args: list
            List of additional args needed to evaluate corresponding
            distribution, such as the data vector, covariance matrix, etc.
        {func}_kwargs: dict (or subclass, such as Pars, MetaPars, etc.)
            Dictionary of additional kwargs needed to evaluate corresponding
            distribution, such as the meta parameters
        '''

        if posterior is None:
            if (likelihood is None) or (prior is None):
                return ValueError('Must pass both a likelihood and prior ' +
                                  'if a posterior is not passed!')
        else:
            if (likelihood is not None) or (prior is not None):
                return ValueError('Cannot pass a likelihood or prior ' +
                                  'if a posterior is passed!')

        self.nwalkers = nwalkers
        self.ndims = ndims
        self.log = log
        self.pool = pool

        self.posterior = posterior
        self.likelihood = likelihood
        self.prior = prior
        self.like_args = like_args
        self.like_kwargs = like_kwargs
        self.prior_args = prior_args
        self.prior_kwargs = prior_kwargs

        self.blob = None

        return

    def posterior(theta):
        '''
        theta: list
            The list of sampled parameters, in order

        returns: tuple (P(theta), blob)
            Returns the tuple of posterior probability and the blob
        '''

        prior = self.prior(theta)

        if prior == -np.inf:
            return -np.inf, (-np.inf, -np.inf)

        likelihood = self.likelihood(theta)

        if self.log is True:
            posterior = likelihood + prior
        else:
            posterior = likelihood * prior

        if return_blob is True:
            return posterior, (likelihood, prior)

    def proposal(theta0, sigma=None):
        '''
        Generates proposal parameter values given the current state theta0

        theta0: list
            A list of current model paremeters, in order
        sigma: list of floats
            The standard deviation of the proposal distribution (all
            gaussian for now) for each parameter
        '''

        N = len(theta0)

        if sigma is None:
            sigma = np.ones(N)

        theta = np.zeros(N)
        for i, t, s in enumerate(zip(theta0, sigma)):
            theta[i](np.normal(t, s, 1).item())

        return theta

    def accept(self, p_theta, p_theta0):
        '''
        Decide whether to accept the proposal theta or reject
        and use theta0 instead

        p_theta: float
            The probability of sample theta
        p_theta0: float
            The probability of sample theta0
        '''

        if self.log is True:
            alpha = np.exp(p_theta - p_theta0)
        else:
            alpha = p_theta / p_theta0

        r = np.random.uniform(0, 1, 1)

        if alpha > r:
            return True
        else:
            return False

    def run_mcmc(self, start, nsteps, progress=True):
        '''
        Manage the MCMC run for all walkers

        start: list / np.ndarray
            The starting value of each sampled parameter for each walker;
            shape should be (nwalkers, ndims)
        nsteps: int
            The number of samples to draw
        progress: bool
            Set to display the approximate progress during the run
        '''

        self.blob = np.zeros((self.nwalkers, 2, nsteps))
        self.chain = np.zeros((self.nwalkers, self.ndims, nsteps))

        if self.pool is None:
            map_fn = map
        else:
            map_fn = self.pool.map

        if progress is True:
            print('Starting MCMC run')
        for i in range(self.nwalkers):
            if progress is True:
                print(f'Starting walker {i}')
            chain, blob = self._run_walker(start[i], nsteps)
            self.chain[i, :, :] = chain
            self.blob[i, :, :] = blob

        return

    def _run_walker(self, start, nsteps, progress=True):
        '''
        start: list / np.ndarray
            The starting value of each sampled parameter for this walker
        nsteps: int
            The number of samples to draw
        progress: bool
            Set to display the approximate progress during the run
        '''

        chain = np.zeros((self.ndims, nsteps))
        blobs = np.zeros((2, nsteps))

        theta0 = start
        p_theta0, b = self.posterior(theta0, return_blob=True)

        for i in range(nsteps):
            import pudb
            pudb.set_trace()
            if ((100*i/nsteps) % 1 == 0) and (progress is True):
                print(f'Starting walker {i}')
            theta = self.proposal(theta0)
            p_theta, b = self.posterior(theta, return_blob=True)

            if self.accept(p_theta, p_theta0):
                chain[:, i] = theta
                blobs[:, i] = b
                theta0 = theta
                b0 = b
            else:
                chain[:, i] = theta0
                blobs[:, i] = b0

        return chain, blob
