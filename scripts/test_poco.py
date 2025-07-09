# This is to ensure that numpy doesn't have
# OpenMP optimizations clobber our own multiprocessing
import os
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
from scipy.stats import uniform, norm
import numpy as np
import ipdb
try:
    import schwimmbad
    import mpi4py
    from mpi4py import MPI
except:
    print("Can not import MPI, use single process")
try:
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
except:
    rank = 0
    size = 1
from schwimmbad import MPIPool
pool = schwimmbad.choose_pool(mpi=True, processes=1)


def main():
    import pocomc as pc
    # Define the dimensionality of our problem.
    ndim = 10

    # Define the prior
    prior = pc.Prior(ndim * [uniform(loc=-10.0, scale=20.0)] )

    # Define our 10-D Rosenbrock log-likelihood.
    def log_likelihood(x, flag=True):
        # print('x = ',x)
        #ipdb.set_trace()
        return -np.sum(10.0*(x[:,::2]**2.0 - x[:,1::2])**2.0 \
                + (x[:,::2] - 1.0)**2.0, axis=1)

    # Number of particles to use
    nparticles = 512

    # Initialise sampler
    #sampler = pc.Sampler(nparticles,
    #                     ndim,
    #                     log_likelihood=log_likelihood,
    #                     log_prior=log_prior,
    #                     vectorize_likelihood=True,
    #                     bounds=(-10.0, 10.0))
    with pool:
        if isinstance(pool, schwimmbad.MPIPool):
            if not pool.is_master():
                pool.wait()
                sys.exit(0)   
        sampler = pc.Sampler(n_dim=ndim, 
                         n_effective=nparticles,
                         n_active = nparticles//2,
                         prior=prior, 
                         likelihood=log_likelihood,
                         likelihood_kwargs={'flag': False},
                         vectorize=True,
                         output_dir="../tests/test_poco",
                         output_label="checkpoint",
                         pool=pool,
                         )
        # Start sampling
        #sampler.run(prior_samples)
        sampler.run(n_total=4096, save_every=10)

    # Save state
    sampler.save_state("../tests/test_poco/final_state")
    # Get results
    #results = sampler.results
    results = sampler.posterior()
    np.save("../tests/test_poco/samples.npy", results[0])
    np.save("../tests/test_poco/weights.npy", results[1])
    np.save("../tests/test_poco/lglikes.npy", results[2])
    np.save("../tests/test_poco/lgposts.npy", results[3])
    ipdb.set_trace()
    return

if __name__ == '__main__':
    main()

