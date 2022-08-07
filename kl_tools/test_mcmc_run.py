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
        utils.TEST_DIR, 'test-mcmc-run', run_name
        )
    utils.make_dir(outdir)

    # for exp gal fits
    true_flux = 1.8e4
    true_hlr = 3.5

    true_pars = {
        'g1': 0.025,
        'g2': -0.0125,
        # 'g1': 0.0,
        # 'g2': 0.0,
        'theta_int': np.pi / 6,
        # 'theta_int': 0.,
        'sini': 0.7,
        'v0': 5,
        'vcirc': 200,
        'rscale': 3,
        # 'beta': np.NaN,
        # 'flux': true_flux,
        # 'hlr': true_hlr,
    }

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
            'rscale': priors.UniformPrior(0, 10),
            # 'beta': priors.UniformPrior(0, 0.5),
            # 'hlr': priors.UniformPrior(0, 8),
            # 'flux': priors.UniformPrior(5e3, 7e4),
        },
        'intensity': {
            # For this test, use truth info
            # 'type': 'inclined_exp',
            # 'flux': true_flux, # counts
            # 'hlr': true_hlr, # counts
            # 'flux': 'sampled', # counts
            # 'hlr': 'sampled', # pixels
            'type': 'basis',
            # 'basis_type': 'shapelets',
            'basis_type': 'sersiclets',
            # 'basis_type': 'exp_shapelets',
            'basis_kwargs': {
                'Nmax': 8,
            #     # 'plane': 'disk',
                'plane': 'obs',
                'beta': 0.28,
            #     'beta': 'sampled',
                'index': 1,
                'b': 1,
                }
        },
        'velocity': {
            'model': 'centered'
        },
        # 'marginalize_intensity': True,
        'run_options': {
            'use_numba': False,
            }
    }

    datacube_pars = {
        # image meta pars
        'Nx': 40, # pixels
        'Ny': 40, # pixels
        'pix_scale': 0.5, # arcsec / pixel
        # intensity meta pars
        'true_flux': true_flux, # counts
        'true_hlr': true_hlr, # pixels
        # velocty meta pars
        'v_model': mcmc_pars['velocity']['model'],
        'v_unit': mcmc_pars['units']['v_unit'],
        'r_unit': mcmc_pars['units']['r_unit'],
        # emission line meta pars
        'wavelength': 656.28, # nm; halpha
        'lam_unit': 'nm',
        'z': 0.3,
        'R': 5000.,
        'sky_sigma': 0.5, # pixel counts for mock data vector
        'psf': gs.Gaussian(fwhm=1., flux=1.)
    }

    print('Setting up test datacube and true Halpha image')
    datacube, vmap, true_im = mocks.setup_likelihood_test(
        true_pars, datacube_pars
        )
    Nspec, Nx, Ny = datacube.shape
    lambdas = datacube.lambdas
    datacube.set_psf(datacube_pars['psf'])

    outfile = os.path.join(outdir, 'true-im.png')
    print(f'Saving true intensity profile in obs plane to {outfile}')
    plt.imshow(true_im, origin='lower')
    plt.colorbar()
    plt.title('True Halpha profile in obs plane')
    plt.savefig(outfile, bbox_inches='tight', dpi=300)
    if show is True:
        plt.show()
    else:
        plt.close()

    outfile = os.path.join(outdir, 'vmap.png')
    print(f'Saving true vamp in obs plane to {outfile}')
    plt.imshow(vmap, origin='lower')
    plt.colorbar(label='v')
    plt.title('True velocity map in obs plane')
    plt.savefig(outfile, bbox_inches='tight', dpi=300)
    if show is True:
        plt.show()
    else:
        plt.close()

    outfile = os.path.join(outdir, 'datacube.fits')
    print(f'Saving test datacube to {outfile}')
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

    sampled_pars = list(true_pars)
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

    truth = np.zeros(ndims)
    for name, indx in pars_order.items():
        truth[indx] = true_pars[name]
    outfile = os.path.join(outdir, 'test-mcmc-truth.pkl')
    print(f'Pickling truth to {outfile}')
    with open(outfile, 'wb') as f:
        pickle.dump(truth, f)

    outfile = os.path.join(outdir, 'chains.png')
    print(f'Saving chain plots to {outfile}')
    reference = pars.pars2theta(true_pars)
    runner.plot_chains(
        outfile=outfile, reference=reference, show=show
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

    outfile = os.path.join(outdir, 'corner-truth.png')
    print(f'Saving corner plot to {outfile}')
    title = 'Reference lines are param truth values'
    runner.plot_corner(
        outfile=outfile, reference=truth, title=title, show=show
        )

    runner.compute_MAP(loglike=blob_data['likelihood'])
    map_vals = runner.MAP_true
    print('MAP values:')
    for name, indx in pars_order.items():
        m = map_vals[indx]
        print(f'{name}: {m:.4f}')

    outfile = os.path.join(outdir, 'compare-data-to-map.png')
    print(f'Plotting MAP comparison to data in {outfile}')
    runner.compare_MAP_to_data(outfile=outfile, show=show)

    outfile = os.path.join(outdir, 'compare-vmap-to-map.png')
    print(f'Plotting MAP comparison to velocity map in {outfile}')
    vmap_pars = true_pars
    vmap_pars['r_unit'] = mcmc_pars['units']['r_unit']
    vmap_pars['v_unit'] = mcmc_pars['units']['v_unit']
    vmap_true = VelocityMap('default', vmap_pars)
    runner.compare_MAP_to_truth(vmap_true, outfile=outfile, show=show)

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
