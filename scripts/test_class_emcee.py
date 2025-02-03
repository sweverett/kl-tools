# This is to ensure that numpy doesn't have
# OpenMP optimizations clobber our own multiprocessing
import os
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
import emcee
import schwimmbad
from schwimmbad import MPIPool
from argparse import ArgumentParser

from mcmc import KLensZeusRunner, KLensEmceeRunner
import likelihood
from likelihood import log_posterior as real_log_posterior
from parameters import PARS_ORDER

import pudb

parser = ArgumentParser()

parser.add_argument('-nsteps', type=int, default=10,
                    help='Number of mcmc iterations per walker')
parser.add_argument('-sampler', type=str, choices=['zeus', 'emcee'],
                    default='emcee',
                    help='Which sampler to use for mcmc')
parser.add_argument('--use_real', action='store_true',
                    help='Set to use real posterior func')
parser.add_argument('--v', action='store_true',
                    help='Turn on to display MCMC progress')

group = parser.add_mutually_exclusive_group()
group.add_argument('-ncores', default=1, type=int,
                    help='Number of processes (uses `multiprocessing`)')
group.add_argument('--mpi', dest='mpi', default=False, action='store_true',
                   help='Run with MPI.')

class TestLikelihood(object):
    '''
    Test likelihood class to see if we can pass
    a class method as the likelihood function to
    emcee & zeus
    '''

    def __init__(self, ivar):
        self.ivar = ivar

        return

    def log_posterior(self, theta, datavector, pars):
        return self.log_prior(theta, pars) + self.log_likelihood(theta, datavector, pars)

    def log_prior(self, theta, pars):
        return 0.

    def log_likelihood(self, theta, datavector, pars):
        return -0.5 * self.sum_chi2(theta)

    def simple_log_likelihood(self, x):
        return self.sum_chi2(x)

    def sum_chi2(self, theta):
        return np.sum(self.ivar * theta**2)

    def real_log_posterior(self, theta, data, pars):
        return real_log_posterior(theta, data, pars)

def main(args, pool):
    nsteps = args.nsteps
    sampler = args.sampler
    ncores = args.ncores
    mpi = args.mpi
    use_real = args.use_real
    vb = args.v

    ndims, nwalkers = 5, 10

    # instantiate likelihood w/ inverse variance set
    # during constructor
    ivar = 1. / np.random.rand(ndims)
    like = TestLikelihood(ivar)

    p0 = np.random.randn(nwalkers, ndims)

    print('Running simple test...')
    runner = emcee.EnsembleSampler(nwalkers, ndims, like.simple_log_likelihood)
    runner.run_mcmc(p0, nsteps, progress=vb)

    # Now a more complicated example ...

    if use_real is False:
        func = like.log_posterior
        datavector = None
        pars = {}
    else:
        func = like.real_log_posterior

        li, le, dl = 654, 657, 1
        lambdas = [(l, l+dl) for l in range(li, le, dl)]

        Nx, Ny = 30, 30
        Nspec = len(lambdas)
        shape = (Nspec, Nx, Ny)

        true_pars, pars = likelihood.setup_test_pars(Nx, Ny)

        print('Setting up test datacube and true Halpha image')
        datavector, sed, vmap, true_im = likelihood.setup_likelihood_test(
            true_pars, pars, shape, lambdas
            )

        pars['sed'] = sed

        ndims = len(PARS_ORDER)

    if sampler == 'zeus':
        print('Setting up KLensZeusRunner')
        nwalkers = 2*ndims
        runner = KLensZeusRunner(
            nwalkers, ndims, func, datavector, pars
        )

    if sampler == 'emcee':
        print('Setting up KLensEmceeRunner')
        nwalkers = 2*ndims

        runner = KLensEmceeRunner(
            nwalkers, ndims, func, datavector, pars
        )

    print(f'Running class test with {sampler}...')
    runner.run(nsteps, pool, vb=vb)

    return 0

if __name__ == '__main__':
    args = parser.parse_args()

    pool = schwimmbad.choose_pool(
        mpi=args.mpi, processes=args.ncores
        )

    if isinstance(pool, MPIPool):
        if not pool.is_master():
            pool.wait()
            sys.exit(0)

    print('Starting tests')
    rc = main(args, pool)

    if rc == 0:
        print('All tests ran succesfully')
    else:
        print(f'Tests failed with return code of {rc}')

