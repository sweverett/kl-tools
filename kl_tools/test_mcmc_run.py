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
from mcmc import KLensZeusRunner, KLensEmceeRunner
import priors
import cube
import likelihood
from parameters import PARS_ORDER
from likelihood import log_posterior
from velocity import VelocityMap

import pudb

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
        utils.TEST_DIR, 'test-mcmc-run', run_name
        )
    utils.make_dir(outdir)

    true_pars = {
        'g1': 0.05,
        'g2': -0.025,
        # 'g1': 0.0,
        # 'g2': 0.0,
        'theta_int': np.pi / 3,
        # 'theta_int': 0.,
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
        'pix_scale': 1., # arcsec / pixel
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
        'sed_start': 655,
        'sed_end': 657.5,
        'sed_resolution': 0.025,
        'sed_unit': Unit('nm'),
        'cov_sigma': 3, # pixel counts; dummy value
        'bandpass_throughput': '.2',
        'bandpass_unit': 'nm',
        'bandpass_zp': 30,
        'priors': {
            'g1': priors.GaussPrior(0., 0.3),#, clip_sigmas=2),
            'g2': priors.GaussPrior(0., 0.3),#, clip_sigmas=2),
            # 'theta_int': priors.UniformPrior(0., np.pi),
            'theta_int': priors.UniformPrior(0., np.pi),
            # 'theta_int': priors.UniformPrior(np.pi/3, np.pi),
            'sini': priors.UniformPrior(0., 1.),
            'v0': priors.UniformPrior(0, 20),
            'vcirc': priors.GaussPrior(200, 10, zero_boundary='positive'),# clip_sigmas=2),
            # 'vcirc': priors.GaussPrior(188, 2.5, zero_boundary='positive', clip_sigmas=2),
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
                'Nmax': 14,
                'plane': 'disk'
                }
        },
        # 'marginalize_intensity': True,
        # 'psf': gs.Gaussian(fwhm=3), # fwhm in pixels
        'use_numba': False,
    }

    li, le, dl = 655.8, 656.8, 0.1
    # li, le, dl = 655.6, 657, 0.1
    # li, le, dl = 655.4, 657.1, 0.1
    lambdas = [(l, l+dl) for l in np.arange(li, le, dl)]

    bandpasses = cube.setup_simple_bandpasses(
        li, le, dl,
        throughput=pars['bandpass_throughput'],
        zp=pars['bandpass_zp'],
        unit=pars['bandpass_unit']
        )

    Nx, Ny = 30, 30
    Nspec = len(lambdas)
    shape = (Nx, Ny, Nspec)
    print('Setting up test datacube and true Halpha image')
    datacube, sed, vmap, true_im = likelihood.setup_likelihood_test(
        true_pars, pars, shape, lambdas
        )

    # Update pars w/ SED, bandapsses, etc. as can no longer assume
    # this will be available in the datavector
    pars['sed'] = sed
    pars['lambdas'] = lambdas
    pars['bandpasses'] = bandpasses

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

    if sampler == 'zeus':
        print('Setting up KLensZeusRunner')
        ndims = len(PARS_ORDER)
        nwalkers = 2*ndims
        runner = KLensZeusRunner(
            nwalkers, ndims, log_posterior, datacube, pars
            )

    elif sampler == 'emcee':
        print('Setting up KLensEmceeRunner')
        ndims = len(PARS_ORDER)
        nwalkers = 2*ndims

        runner = KLensEmceeRunner(
            nwalkers, ndims, log_posterior, datacube, pars
            )

    print('Starting mcmc run')
    # try:
    runner.run(nsteps, pool)
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
        print('pickling chain to {outfile}')
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

    truth = np.zeros(len(PARS_ORDER))
    for name, indx in PARS_ORDER.items():
        truth[indx] = true_pars[name]
    outfile = os.path.join(outdir, 'test-mcmc-truth.pkl')
    print(f'Pickling truth to {outfile}')
    with open(outfile, 'wb') as f:
        pickle.dump(truth, f)

    outfile = os.path.join(outdir, 'chains.png')
    print(f'Saving chain plots to {outfile}')
    reference = likelihood.pars2theta(true_pars)
    runner.plot_chains(
        outfile=outfile, reference=reference, show=show
        )

    outfile = os.path.join(outdir, 'corner-truth.png')
    print(f'Saving corner plot to {outfile}')
    title = 'Reference lines are param truth values'
    runner.plot_corner(
        outfile=outfile, reference=truth, title=title, show=show
        )

    runner.compute_MAP()
    map_medians = runner.MAP_medians
    print('(median) MAP values:')
    for name, indx in PARS_ORDER.items():
        m = map_medians[indx]
        print(f'{name}: {m:.4f}')

    outfile = os.path.join(outdir, 'compare-data-to-map.png')
    print(f'Plotting MAP comparison to data in {outfile}')
    runner.compare_MAP_to_data(outfile=outfile, show=show)

    outfile = os.path.join(outdir, 'compare-vmap-to-map.png')
    print(f'Plotting MAP comparison to velocity map in {outfile}')
    vmap_pars = true_pars
    vmap_pars['r_unit'] = pars['r_unit']
    vmap_pars['v_unit'] = pars['v_unit']
    vmap_true = VelocityMap('default', vmap_pars)
    runner.compare_MAP_to_truth(vmap_true, outfile=outfile, show=show)

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

    outfile = os.path.join(outdir, 'chain-probabilities.pkl')
    print(f'Saving prior & likelihood values to {outfile}')
    data = {
        'prior': blobs[:,:,0],
        'likelihood': blobs[:,:,1]
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
