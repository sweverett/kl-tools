# This is to ensure that numpy doesn't have
# OpenMP optimizations clobber our own multiprocessing
import os
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
import ipdb

def main():

    # Define the dimensionality of our problem.
    ndim = 10

    # Define our 10-D Rosenbrock log-likelihood.
    def log_likelihood(x):
        # print('x = ',x)
        ipdb.set_trace()
        return -np.sum(10.0*(x[:,::2]**2.0 - x[:,1::2])**2.0 \
                + (x[:,::2] - 1.0)**2.0, axis=1)

    # Define our uniform prior.
    def log_prior(x):
        if np.any(x < -10.0) or np.any(x > 10.0):
            return -np.inf
        else:
            return 0.0

    # Number of particles to use
    nparticles = 1000

    # Initialise sampler
    import pocomc as pc
    sampler = pc.Sampler(n_particles,
                         n_dim,
                         log_likelihood=log_likelihood,
                         log_prior=log_prior,
                         vectorize_likelihood=True,
                         bounds=(-10.0, 10.0))

    # Initialise particles' positions using samples from the prior (this is very important, other initialisation will not work).
    prior_samples = np.random.uniform(low=-10.0, high=10.0, size=(nparticles, ndim))

    # Start sampling
    sampler.run(prior_samples)

    # We can add more samples at the end
    sampler.add_samples(1000)

    # Get results
    results = sampler.results

    return

if __name__ == '__main__':
    main()
