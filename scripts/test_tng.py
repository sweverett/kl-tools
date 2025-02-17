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
import cube
import mocks
import likelihood
from parameters import Pars
from likelihood import LogPosterior
from velocity import VelocityMap

import ipdb

parser = ArgumentParser()

parser.add_argument('nsteps', type=int,
                    help='Number of mcmc iterations per walker')
parser.add_argument('-sampler', type=str, choices=['zeus', 'emcee', 'poco'],
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
        utils.TEST_DIR, 'test-tng', run_name
        )
    utils.make_dir(outdir)

    # true_pars = {
    #     'g1': 0.025,
    #     'g2': -0.0125,
    #     # 'g1': 0.0,
    #     # 'g2': 0.0,
    #     'theta_int': np.pi / 6,
    #     # 'theta_int': 0.,
    #     'sini': 0.7,
    #     'v0': 5,
    #     'vcirc': 200,
    #     'rscale': 3,
    #     # 'beta': np.NaN
    #     # 'flux': 1.8e4,
    #     # 'hlr': 3.5,
    # }

    sampled_pars = ['g1',
                    'g2',
                    'theta_int',
                    'sini',
                    'v0',
                    'vcirc',
                    'rscale',
                    'x0',
                    'y0',
                    'beta'
                    ]

    mcmc_pars = {
        'units': {
            'v_unit': Unit('km / s'),
            'r_unit': Unit('kpc'),
        },
        'priors': {
            'g1': priors.GaussPrior(0., 0.1, clip_sigmas=2),
            'g2': priors.GaussPrior(0., 0.1, clip_sigmas=2),
            # 'theta_int': priors.UniformPrior(0., np.pi),
            'theta_int': priors.UniformPrior(0., np.pi),
            # 'theta_int': priors.UniformPrior(np.pi/3, np.pi),
            'sini': priors.UniformPrior(0., 1.),
            'v0': priors.UniformPrior(0, 20),
            'vcirc': priors.GaussPrior(200, 20, clip_sigmas=3),
            # 'vcirc': priors.GaussPrior(188, 2.5, zero_boundary='positive', clip_sigmas=2),
            # 'vcirc': priors.UniformPrior(190, 210),
            'rscale': priors.UniformPrior(0, 20),
            'x0': priors.UniformPrior(-20, 20.),
            'y0': priors.UniformPrior(-20, 20.),
            'beta': priors.UniformPrior(0, 0.5),
            # 'hlr': priors.UniformPrior(0, 8),
            # 'flux': priors.UniformPrior(5e3, 7e4),
        },
        'intensity': {
            # For this test, use truth info
            # 'type': 'inclined_exp',
            # 'flux': 3.8e4, # counts
            # 'hlr': 3.5,
            # 'flux': 'sampled', # counts
            # 'hlr': 'sampled', # pixels
            'type': 'basis',
            # 'basis_type': 'shapelets',
            'basis_type': 'sersiclets',
            # 'basis_type': 'exp_shapelets',
            'basis_kwargs': {
                'Nmax': 7,
                # 'plane': 'disk',
                'plane': 'obs',
                # 'beta': 0.35,
                'beta': 'sampled',
                # 'index': 1,
                # 'b': 1,
                }
        },
        'velocity': {
            'model': 'offset'
        },
        # 'marginalize_intensity': True,
        # 'psf': gs.Gaussian(fwhm=.5), # fwhm in pixels
        'run_options': {
            'use_numba': False,
            }
    }

    with open('../.cache/TNG50-1_subhalo_2_snap67_MUSEcube.pickle', 'rb') as p:
        datacube = pickle.load(p)

    # plt.imshow(datacube.stack())
    # plt.colorbar()
    # plt.show()
    # ipdb.set_trace()

    datacube.set_line()

    Nspec, Nx, Ny = datacube.shape
    lambdas = datacube.lambdas

    outfile = os.path.join(outdir, 'datacube.fits')
    print(f'Saving TNG datacube to {outfile}')
    datacube.write(outfile)

    outfile = os.path.join(outdir, 'datacube-slices.png')
    print(f'Saving example datacube slice images to {outfile}')
    # if Nspec < 10:
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

    print(f'Setting up {sampler} MCMCRunner')
    kwargs = {}
    if sampler in ['zeus', 'emcee']:
        nwalkers = 2*ndims
        args = [nwalkers, ndims, log_posterior, datacube, pars]

    elif sampler == 'poco':
        nwalkers = 1000
        args = [
            nwalkers,
            ndims,
            log_posterior.log_likelihood,
            log_posterior.log_prior,
            datacube,
            pars
            ]

    runner = build_mcmc_runner(sampler, args, kwargs)

    #-----------------------------------------------------------------
    # Run MCMC

    print('Starting mcmc run')
    # try:
    runner.run(pool, nsteps=nsteps)
    # except Exception as e:
    #     g1 = runner.start[:,0]
    #     g2 = runner.start[:,1]
    #     print('Starting ball for (g1, g2):')
    #     print(f'g1: {g1}')
    #     print(f'g2: {g2}')
    #     val = np.sqrt(g1**2+g2**2)
    #     print(f' |g1+ig2| = {val}')
    #     raise e

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

    if sampler == 'emcee':
        blobs = runner.sampler.blobs
    elif sampler == 'zeus':
        blobs = runner.sampler.get_blobs()

    outfile = os.path.join(outdir, 'chain-probabilities.pkl')
    print(f'Saving prior & likelihood values to {outfile}')
    blob_data = {
        'prior': blobs[:,:,0],
        'likelihood': blobs[:,:,1]
    }
    with open(outfile, 'wb') as f:
        pickle.dump(blob_data, f)

    outfile = os.path.join(outdir, 'chain-probabilities.png')
    print(f'Saving prior & likelihood value plot to {outfile}')
    indx = np.random.randint(0, high=nwalkers)
    prior = blobs[:,indx,0]
    like = blobs[:,indx,1]
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

    runner.compute_MAP(loglike=blob_data['likelihood'])
    map_vals = runner.MAP_true
    print('MAP values:')
    for name, indx in pars_order.items():
        m = map_vals[indx]
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
