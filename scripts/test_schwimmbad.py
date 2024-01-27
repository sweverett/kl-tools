import os
import time
import sys
import numpy as np
import schwimmbad
from multiprocessing import Pool
import emcee
from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument('nsteps', type=int,
                    help='Number of mcmc iterations per walker')

group = parser.add_mutually_exclusive_group()
group.add_argument('-ncores', default=1, type=int,
                    help='Number of processes to use (outside of MPI)')
group.add_argument('--mpi', action='store_true', default=False,
                   help='Run with MPI.')

parser.add_argument('--use_builtin', action='store_true', default=False,
                    help='Turn on to use the builtin multiprocessing module')

def log_posterior(theta):
    # Do something slow
    N = int(1e6)
    s = 0
    for i in range(N):
        s += i

    # Now return a dummy value
    return -0.5 * np.sum(theta ** 2)

class EmceeRunner(object):
    def __init__(self, nwalkers, ndim, pfunc, args=None, kwargs=None):

        self.nwalkers = nwalkers
        self.ndim = ndim
        self.pfunc = pfunc
        self.args = args
        self.kwargs = kwargs

        return

    def _initialize_walkers(self):
        self.start = np.random.randn(nwalkers, ndims)

        return

    def _initialize_sampler(self, pool):
        sampler = emcee.EnsembleSampler(
            self.nwalkers, self.ndim, self.pfunc,
            args=self.args, kwargs=self.kwargs, pool=pool
            )

        return sampler

    def run(self, nsteps, pool):
        sampler = self._initialize_sampler(pool)

        if not isinstance(pool, schwimmbad.SerialPool):
            # Otherwise some libraries can degrade performance
            os.environ['OMP_NUM_THREADS'] = '1'

        with pool:

            pt = type(pool)
            print(f'Pool type: {pt}')
            print(f'Pool: {pool}')

            if isinstance(pool, schwimmbad.MPIPool):
                if not pool.is_master():
                    pool.wait()
                    sys.exit(0)

            sampler = self._initialize_sampler(pool=pool)
            sampler.run_mcmc(
                start, nsteps, progress=progress
                )

        return

def main(args, pool):

    ndims = 8
    nwalkers = 2*ndims

    print('Beginning minimal test')

    if args.use_builtin is True:
        pool = Pool(args.ncores)
    else:
        pool = schwimmbad.choose_pool(
            mpi=args.mpi, processes=args.ncores
        )

    if not isinstance(pool, schwimmbad.SerialPool):
        os.environ['OMP_NUM_THREADS'] = '1'

    with pool:
        if isinstance(pool, schwimmbad.MPIPool):
            if not pool.is_master():
                pool.wait()
                sys.exit(0)

        sampler = emcee.EnsembleSampler(
            nwalkers, ndims, log_posterior, pool=pool
            )

        sampler.run_mcmc(start, args.nsteps, progress=True)

    print('Beginning class test')
    runner = EmceeRunner(nwalkers, ndims, log_posterior)
    runner.run(args.nsteps, pool)

    return 0

if __name__ == '__main__':
    args = parser.parse_args()

    pool = schwimmbad.choose_pool(
        mpi=args.mpi, processes=args.ncores
        )

    if isinstance(pool, schwimmbad.MPIPool):
        if not pool.is_master():
            pool.wait()
            sys.exit(0)

    print('Starting tests')
    rc = main(args, pool)

    if rc == 0:
        print('All tests ran succesfully')
    else:
        print(f'Tests failed with return code of {rc}')
