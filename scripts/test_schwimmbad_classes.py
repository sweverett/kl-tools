# This is to ensure that numpy doesn't have
# OpenMP optimizations clobber our own multiprocessing
import os
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import numpy
import time
import sys
from astropy.units import Unit
import schwimmbad
from multiprocessing import Pool
import emcee
from argparse import ArgumentParser

import priors
import likelihood
from parameters import PARS_ORDER
from mcmc import KLensEmceeRunner

from likelihood import log_posterior

import pudb

parser = ArgumentParser()

parser.add_argument('nsteps', type=int,
                    help='Number of mcmc iterations per walker')

group = parser.add_mutually_exclusive_group()
group.add_argument('-ncores', default=1, type=int,
                    help='Number of processes to use (outside of MPI)')
group.add_argument('--mpi', action='store_true', default=False,
                   help='Run with MPI.')
parser.add_argument('-sampler', type=str, choices=['zeus', 'emcee'],
                    default='emcee',
                    help='Which sampler to use for mcmc')
parser.add_argument('-run_name', type=str, default='',
                    help='Name of mcmc run')
parser.add_argument('--show', action='store_true', default=False,
                    help='Set to show test plots')

parser.add_argument('--use_builtin', action='store_true', default=False,
                    help='Turn on to use the builtin multiprocessing module')

# def log_posterior(theta, datacube=None, args=None, pars=None):
#     # Do something slow
#     N = int(1e6)
#     s = 0
#     for i in range(N):
#         s += i

#     # Now return a dummy value
#     return -0.5 * np.sum(theta ** 2)

class EmceeRunner(object):
    def __init__(self, nwalkers, ndim, pfunc, args=None, kwargs=None):

        self.nwalkers = nwalkers
        self.ndim = ndim
        self.pfunc = pfunc
        self.args = args
        self.kwargs = kwargs

        return

    def _initialize_walkers(self):
        self.start = np.random.randn(self.nwalkers, self.ndim)

        return

    def _initialize_sampler(self, pool):
        sampler = emcee.EnsembleSampler(
            self.nwalkers, self.ndim, self.pfunc,
            args=self.args, kwargs=self.kwargs, pool=pool
            )

        return sampler

    def run(self, nsteps, pool, start=None):

        if start is None:
            self._initialize_walkers()
            start = self.start

        sampler = self._initialize_sampler(pool)

        # if not isinstance(pool, schwimmbad.SerialPool):
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
                start, nsteps, progress=True
                )

        return

def main(args, pool):

    # Needed to setup likelihood
    true_pars = {
        'g1': 0.05,
        'g2': -0.025,
        'theta_int': np.pi / 3,
        'sini': 0.8,
        'v0': 10.,
        'vcirc': 200,
        'rscale': 5,
    }

    # additional args needed for prior / likelihood evaluation
    halpha = 656.28 # nm
    R = 5000.
    z = 0.3
    pars = {
        'Nx': 30, # pixels
        'Ny': 30, # pixels
        'true_flux': 1e5, # counts
        'true_hlr': 5, # pixels
        'v_unit': Unit('km / s'),
        'r_unit': Unit('kpc'),
        'z': z,
        'spec_resolution': R,
        # 'line_std': 0.17,
        'line_std': halpha * (1.+z) / R, # emission line SED std; nm
        'line_value': 656.28, # emission line SED std; nm
        'line_unit': Unit('nm'),
        'sed_start': 650,
        'sed_end': 660,
        'sed_resolution': 0.025,
        'sed_unit': Unit('nm'),
        'cov_sigma': 4, # pixel counts; dummy value
        'bandpass_throughput': '.2',
        'bandpass_unit': 'nm',
        'bandpass_zp': 30,
        'priors': {
            'g1': priors.GaussPrior(0., 0.1),#, clip_sigmas=2),
            'g2': priors.GaussPrior(0., 0.1),#, clip_sigmas=2),
            'theta_int': priors.UniformPrior(0., np.pi),
            # 'theta_int': priors.UniformPrior(np.pi/3, np.pi),
            'sini': priors.UniformPrior(0., 1.),
            # 'sini': priors.GaussPrior()
            'v0': priors.UniformPrior(0, 20),
            'vcirc': priors.GaussPrior(200, 10, clip_sigmas=2),
            # 'vcirc': priors.UniformPrior(190, 210),
            'rscale': priors.UniformPrior(0, 10),
        },
        'intensity': {
            # For this test, use truth info
            # 'type': 'inclined_exp',
            # 'flux': 1e5, # counts
            # 'hlr': 5, # pixels
            'type': 'basis',
            'basis_type': 'shapelets',
            'basis_kwargs': {
                'Nmax': 10,
                'plane': 'obs'
                }
        },
        # 'psf': gs.Gaussian(fwhm=3), # fwhm in pixels
        'use_numba': False,
    }

    # li, le, dl = 655.5, 657, 0.1
    li, le, dl = 655.8, 656.8, 0.1
    # li, le, dl = 655.9, 656.8, .1
    lambdas = [(l, l+dl) for l in np.arange(li, le, dl)]

    Nx, Ny = 30, 30
    Nspec = len(lambdas)
    shape = (Nx, Ny, Nspec)
    print('Setting up test datacube and true Halpha image')
    datacube, sed, vmap, true_im = likelihood.setup_likelihood_test(
        true_pars, pars, shape, lambdas
        )

    pars['sed'] = sed

    ndim = len(PARS_ORDER)
    nwalkers = 2*ndim

    print('Setting up KLensEmceeRunner')

    runner = KLensEmceeRunner(
        nwalkers, ndim, log_posterior, datacube, pars#, datacube, pars
        )

    print('Starting mcmc run')
    # start = np.random.randn(nwalkers, ndim)
    start = None
    # pudb.set_trace()
    runner.run(args.nsteps, pool, start=start)

    # print('Starting EmceeRunner')
    # runner = EmceeRunner(nwalkers, ndim, log_posterior)
    # print('Starting mcmc run')
    # runner.run(args.nsteps, pool)

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
