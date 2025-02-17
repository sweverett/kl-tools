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

import utils
from mcmc import build_mcmc_runner
import priors
from muse import MuseDataCube
import likelihood
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
        'x0',
        'y0',
        'z',
        'R'
        # 'beta'
        ]

    # additional args needed for prior / likelihood evaluation
    mcmc_pars = {
        'units': {
            'v_unit': Unit('km/s'),
            'r_unit': Unit('kpc')
            },
        'priors': {
            'g1': priors.GaussPrior(0., 0.01, clip_sigmas=10),
            'g2': priors.GaussPrior(0., 0.01, clip_sigmas=10),
            # 'g1': priors.UniformPrior(-.01, 0.01),
            # 'g2': priors.UniformPrior(-.01, 0.01),
            # 'theta_int': priors.UniformPrior(0., np.pi),
            'theta_int': priors.UniformPrior(0., 2.*np.pi),
            # 'theta_int': priors.UniformPrior(np.pi/3, np.pi),
            'sini': priors.UniformPrior(0, 1.),
            'v0': priors.UniformPrior(0, 20),
            # 'vcirc': priors.GaussPrior(200, 20, zero_boundary='positive'),# clip_sigmas=2),
            'vcirc': priors.UniformPrior(0, 800),
            # 'vcirc': priors.GaussPrior(188, 2.5, zero_boundary='positive', clip_sigmas=2),
            # 'vcirc': priors.UniformPrior(190, 210),
            'rscale': priors.UniformPrior(0, 40),
            'x0': priors.GaussPrior(0, 2.5),
            'y0': priors.GaussPrior(0, 2.5),
            'z': priors.GaussPrior(0.2466, .00001),# clip_sigmas=3),
            'R': priors.GaussPrior(3200, 20),# clip_sigmas=4),
            # 'beta': priors.UniformPrior(0, .2),
            # 'hlr': priors.UniformPrior(0, 8),
            # 'flux': priors.UniformPrior(5e3, 7e4),
            },
        'velocity': {
            'model': 'offset'
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
            'basis_type': 'sersiclets',
            # 'basis_type': 'exp_shapelets',
            'basis_kwargs': {
                'use_continuum_template': True,
                'Nmax': 12,
            #     # 'plane': 'disk',
                'plane': 'obs',
                # 'beta': 0.17,
                'beta': 0.61,
                # 'beta': 'sampled',
                'index': 1,
                'b': 1,
                }
            },
        # 'marginalize_intensity': True,
        'psf': gs.Gaussian(fwhm=.8, flux=1.0), # fwhm in arcsec
        # 'psf': gs.Moffat(fwhm=.8, beta=2.5, flux=1.0), # fwhm in arcsec
        'run_options': {
            'remove_continuum': True,
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

    datacube.set_psf(mcmc_pars['psf'])

    print(f'Strongest emission line has {Nspec} slices')

    outfile = os.path.join(outdir, 'datacube-slices.png')
    print(f'Saving example datacube slice images to {outfile}')
    sqrt = int(np.ceil(np.sqrt(Nspec)))
    slice_indices = range(Nspec)

    k = 1
    for i in slice_indices:
        plt.subplot(sqrt, sqrt, k)
        plt.imshow(datacube.slices[i]._data, origin='lower')
        plt.colorbar()
        l, r = lambdas[i]
        plt.title(f'lambda=({l:.1f}, {r:.1f})')
        k += 1
    plt.gcf().set_size_inches(12,12)
    plt.tight_layout()
    plt.savefig(outfile, bbox_inches='tight', dpi=300)
    if show is True:
        plt.show()
    else:
        plt.close()

    #-----------------------------------------------------------------
    # Setup sampled posterior

    pars = Pars(sampled_pars, mcmc_pars)
    pars_order = pars.sampled.pars_order

    log_posterior = LogPosterior(pars, datacube, likelihood='datacube')

    #-----------------------------------------------------------------
    # Setup sampler

    ndims = log_posterior.ndims
    nwalkers = 2*ndims

    print(f'Setting up {sampler} MCMCRunner')
    args = [nwalkers, ndims]
    if sampler in ['zeus', 'emcee']:
        args += [log_posterior, datacube, pars]
        kwargs = {}

    elif sampler == 'poco':
        kwargs = {
            'datacube': datacube,
            'pars': pars,
            'loglike': log_posterior.log_likelihood,
            'logprior': log_posterior.log_prior,
        }

    runner = build_mcmc_runner(sampler, args, kwargs)

    #-----------------------------------------------------------------
    # Run MCMC

    print('Starting mcmc run')
    runner.run(pool, nsteps=nsteps)

    runner.burn_in = nsteps // 2

    if (sampler == 'zeus') and ((ncores > 1) or (mpi == True)):
        # The sampler isn't pickleable for some reason in this scenario
        # Save whole chain
        outfile = os.path.join(outdir, 'test-mcmc-chain.pkl')
        chain = runner.sampler.get_chain(flat=True)
        print(f'pickling chain to {outfile}')
        with open(outfile, 'wb') as f:
            pickle.dump(chain, f)
    else:
        outfile = os.path.join(outdir, 'test-mcmc-sampler.pkl')
        print(f'Pickling sampler to {outfile}')
        with open(outfile, 'wb') as f:
            pickle.dump(runner.sampler, f)

        outfile = os.path.join(outdir, 'test-mcmc-runner.pkl')
        print(f'Pickling runner to {outfile}')
        with open(outfile, 'wb') as f:
            pickle.dump(runner, f)

    outfile = os.path.join(outdir, 'chains.png')
    print(f'Saving chain plots to {outfile}')
    runner.plot_chains(
        outfile=outfile, show=show
        )

    runner.compute_MAP()
    map_medians = runner.MAP_medians
    print('(median) MAP values:')
    for name, indx in pars_order.items():
        m = map_medians[indx]
        print(f'{name}: {m:.4f}')

    outfile = os.path.join(outdir, 'compare-data-to-map.png')
    print(f'Plotting MAP comparison to data in {outfile}')
    runner.compare_MAP_to_data(outfile=outfile, show=show)

    outfile = os.path.join(outdir, 'corner-map.png')
    print(f'Saving corner plot compare to MAP in {outfile}')
    title = 'Reference lines are param MAP values'
    runner.plot_corner(
        outfile=outfile, reference=runner.MAP_medians, title=title, show=show
        )

    if sampler == 'emcee':
        blobs = runner.sampler.blobs
    elif sampler == 'zeus':
        blobs = runner.sampler.get_blobs()

    outfile = os.path.join(outdir, 'chain-blob.pkl')
    print(f'Saving prior & likelihood values to {outfile}')
    data = {
        'prior': blobs[:,:,0],
        'likelihood': blobs[:,:,1],
        # TODO: generalize or remove after debugging!
        # 'image': blobs[:,:,2],
        # 'cont_template': blobs[:,:,3],
        # 'mle_coeff': blobs[:,:,4],
    }
    with open(outfile, 'wb') as f:
        pickle.dump(data, f)

    outfile = os.path.join(outdir, 'chain-probabilities.png')
    print(f'Saving prior & likelihood value plot to {outfile}')
    prior = blobs[:,indx,0]
    like = blobs[:,indx,1]
    indx = np.random.randint(0, high=nwalkers)
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 4))
    plt.subplot(131)
    plt.plot(prior, label='prior', c='tab:blue')
    plt.xlabel('Sample')
    plt.ylabel('Log probability')
    plt.legend()
    plt.subplot(132)
    plt.plot(like, label='likelihood', c='tab:orange')
    plt.xlabel('Sample')
    plt.ylabel('Log probability')
    plt.legend()
    plt.subplot(133)
    plt.plot(prior, label='prior', c='tab:blue')
    plt.plot(like, label='likelihood', c='tab:orange')
    plt.xlabel('Sample')
    plt.ylabel('Log probability')
    plt.legend()

    plt.savefig(outfile, bbox_inches='tight', dpi=300)

    if show is True:
        plt.show()
    else:
        plt.close()

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
