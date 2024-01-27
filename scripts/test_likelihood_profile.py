# This is to ensure that numpy doesn't have
# OpenMP optimizations clobber our own multiprocessing
import os
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
import sys
import pickle
import schwimmbad
import mpi4py
from schwimmbad import MPIPool
from argparse import ArgumentParser
from astropy.units import Unit
import galsim as gs
import matplotlib.pyplot as plt
import zeus

import kl_tools.utils as utils
from kl_tools.mcmc import KLensZeusRunner, KLensEmceeRunner
import kl_tools.priors as priors
from kl_tools.muse import MuseDataCube
import kl_tools.likelihood as likelihood
from parameters import Pars
from likelihood import LogPosterior
from velocity import VelocityMap

import ipdb

parser = ArgumentParser()

parser.add_argument('nsteps', type=int,
                    help='Number of mcmc iterations per walker')
parser.add_argument('-sampler', type=str, choices=['zeus', 'emcee'],
                    default='emcee',
                    help='Which sampler to use for mcmc')
parser.add_argument('-run_name', type=str, default='',
                    help='Name of mcmc run')
parser.add_argument('--show', action='store_true', default=False,
                    help='Set to show test plots')

group = parser.add_mutually_exclusive_group()
group.add_argument('-ncores', default=1, type=int,
                    help='Number of processes (uses `multiprocessing`)')
group.add_argument('--mpi', dest='mpi', default=False, action='store_true',
                   help='Run with MPI.')

def main(args, pool):

    # TODO: Try making datacube a global variable, as it may significantly
    # decrease execution time due to pickling only once vs. every model eval

    nsteps = args.nsteps
    sampler = args.sampler
    ncores = args.ncores
    mpi = args.mpi
    run_name = args.run_name
    show = args.show

    outdir = os.path.join(
        utils.TEST_DIR, 'muse-mcmc-run', run_name
        )
    utils.make_dir(outdir)

    sampled_pars = [
        'g1',
        'g2',
        'theta_int',
        'sini',
        'v0',
        'vcirc',
        'rscale',
        'z'
        #'beta'
        ]

    # additional args needed for prior / likelihood evaluation
    meta_pars = {
        'units': {
            'v_unit': Unit('km/s'),
            'r_unit': Unit('kpc')
            },
        'priors': {
            # 'g1': priors.GaussPrior(0., 0.3, clip_sigmas=2),
            # 'g2': priors.GaussPrior(0., 0.3, clip_sigmas=2),
            'g1': priors.UniformPrior(-.01, 0.01),
            'g2': priors.UniformPrior(-.01, 0.01),
            # 'theta_int': priors.UniformPrior(0., np.pi),
            'theta_int': priors.UniformPrior(0., np.pi),
            # 'theta_int': priors.UniformPrior(np.pi/3, np.pi),
            'sini': priors.UniformPrior(-1, 1.),
            'v0': priors.UniformPrior(0, 20),
            # 'vcirc': priors.GaussPrior(200, 20, zero_boundary='positive'),# clip_sigmas=2),
            'vcirc': priors.UniformPrior(0, 400),
            # 'vcirc': priors.GaussPrior(188, 2.5, zero_boundary='positive', clip_sigmas=2),
            # 'vcirc': priors.UniformPrior(190, 210),
            'rscale': priors.UniformPrior(0, 20),
            # 'beta': priors.UniformPrior(0, .2),
            # 'hlr': priors.UniformPrior(0, 8),
            # 'flux': priors.UniformPrior(5e3, 7e4),
            },
        'intensity': {
            # For this test, use truth info
            # 'type': 'inclined_exp',
            # 'flux': 1.8e4, # counts
            # 'hlr': 3.5,
            # 'flux': 'sampled', # counts
            # 'hlr': 'sampled', # pixels
            'type': 'basis',
            # 'basis_type': 'shapelets',
            # 'basis_type': 'sersiclets',
            'basis_type': 'exp_shapelets',
            'basis_kwargs': {
                'use_continuum_template': True,
                'Nmax': 21,
            #     # 'plane': 'disk',
                'plane': 'obs',
                'beta': 0.17,
                # 'beta': 'sampled',
            #     # 'index': 1,
            #     # 'b': 1,
                }
            },
        # 'marginalize_intensity': True,
        # 'psf': gs.Gaussian(fwhm=1), # fwhm in pixels
        'run_options': {
            'use_numba': False
            }
    }

    cube_dir = os.path.join(utils.TEST_DIR, 'test_data')

    cubefile = os.path.join(cube_dir, '102021103_objcube.fits')
    specfile = os.path.join(cube_dir, 'spectrum_102021103.fits')
    catfile = os.path.join(cube_dir, 'MW_1-24_main_table.fits')
    linefile = os.path.join(cube_dir, 'MW_1-24_emline_table.fits')

    print(f'Setting up MUSE datacube from file {cubefile}')
    datacube = MuseDataCube(
        cubefile, specfile, catfile, linefile
        )

    # default, but we'll make it explicit:
    datacube.set_line(line_choice='strongest')
    Nspec = datacube.Nspec
    lambdas = datacube.lambdas

    print(f'Strongest emission line has {Nspec} slices')

    #-----------------------------------------------------------------
    # Setup sampled posterior

    pars = Pars(sampled_pars, meta_pars)
    pars_order = pars.sampled.pars_order

    log_posterior = LogPosterior(pars, datacube, likelihood='datacube')

    #-----------------------------------------------------------------
    # Setup fake sampler

    # for n in range(nsteps):
    #     theta = np.random.rand(len(sampled_pars))
    #     log_posterior(theta, datacube, pars)

    #-----------------------------------------------------------------
    # Setup emcee sampler

    ndims = len(sampled_pars)
    # nwalkers = 2*ndims
    nwalkers = 2*ndims

    # using this part just to get an initial guess...
    runner = KLensEmceeRunner(
        nwalkers, ndims, log_posterior, datacube, pars
    )
    runner._initialize_walkers()
    initial = runner.start

    # now for base sampler
    import emcee
    args = [datacube]
    kwargs = {'pars':pars.meta.pars}
    sampler = emcee.EnsembleSampler(
        nwalkers, ndims, log_posterior, args=args, kwargs=kwargs
        # nwalkers, ndims, log_posterior, pool=pool, args=args, kwargs=kwargs
    )

    sampler.run_mcmc(initial, nsteps, progress=True)

    return 0

if __name__ == '__main__':
    args = parser.parse_args()

    pool = schwimmbad.choose_pool(
        mpi=args.mpi, processes=args.ncores
        )

    if isinstance(pool, MPIPool):
        if not pool.is_master():
            pool.wait()


    print('Starting tests')
    rc = main(args, pool)

    if rc == 0:
        print('All tests ran succesfully')
    else:
        print(f'Tests failed with return code of {rc}')
