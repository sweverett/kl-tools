# This is to ensure that numpy doesn't have
# OpenMP optimizations clobber our own multiprocessing
import os
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
import sys, copy
sys.path.insert(0, './grism_modules')
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
from muse import MuseDataCube
import likelihood
from parameters import Pars
from likelihood import LogPosterior, GrismLikelihood
from velocity import VelocityMap
from grism import GrismDataVector

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
        utils.TEST_DIR, 'grism-mcmc-run', run_name
        )
    utils.make_dir(outdir)

    ### Initialization
    sampled_pars = [
        'g1',
        'g2',
        'theta_int',
        'sini',
        'v0',
        'vcirc',
        'rscale',
        #'hlr',
        ]
    sampled_pars_value = [0.0, 0.0, -1.04, 0.86, 0.0, 300.0, 0.5]
    sampled_pars_value_dict = {k:v for k,v in zip(sampled_pars, sampled_pars_value)}
    meta_pars = {
        'g1': 'sampled',
        'g1': 'sampled',
        'theta_int': 'sampled',
        'sini': 'sampled',
        'units': {
            'v_unit': Unit('km/s'),
            'r_unit': Unit('arcsec')
            },
        'priors': {
            'g1': priors.GaussPrior(0., 0.1, clip_sigmas=2),
            'g2': priors.GaussPrior(0., 0.1, clip_sigmas=2),
            'theta_int': priors.UniformPrior(-np.pi, np.pi),
            'sini': priors.UniformPrior(0.001, 0.999),
            'v0': priors.GaussPrior(0.0, 30, clip_sigmas=5),
            #'vcirc': priors.UniformPrior(0, 800),
            'vcirc': priors.GaussPrior(300.0, 30, clip_sigmas=5),
            'rscale': priors.UniformPrior(0.001, 4),
            #'hlr': priors.UniformPrior(0, 2),
            },
        'velocity': {
            'model': 'default',
            'v0': 'sampled',
            'vcirc': 'sampled',
            'rscale': 'sampled',
        },
        'intensity': {
            # For this test, use truth info
            'type': 'inclined_exp',
            'flux': 1.0, # counts
            'hlr': 0.5,
            #'hlr': 'sampled',
            },
        # 'marginalize_intensity': True,
        'psf': gs.Gaussian(fwhm=0.13),
        'run_options': {
            #'remove_continuum': True,
            'use_numba': False
            },
        'model_dimension':{
            'Nx': 100,
            'Ny': 100,
            'scale': 0.0325, # arcsec
            'lambda_range': [1236.6, 1325.1], # 1190-1370
            'lambda_res': 0.5, # nm
            'lambda_unit': 'nm',
        },
        'sed':{
            'template': '../data/Simulation/GSB2.spec',
            'wave_type': 'Ang',
            'flux_type': 'flambda',
            'z': 0.9513,
            'wave_range': [500., 3000.], # nm
            # obs-frame continuum normalization (nm, erg/s/cm2/nm)
            'obs_cont_norm': [614, 2.6e-17],
            # a dict of line names and obs-frame flux values (erg/s/cm2)
            'lines':{
                'Ha': 1.25e-16,
                'O2': [1.0e-15, 1.2e-15],
                'O3_1': 1.0e-15,
                'O3_2': 1.2e-15,
            },
            # intrinsic linewidth in nm
            'line_sigma_int':{
                'Ha': 0.5,
                'O2': [0.2, 0.2],
                'O3_1': 0.2,
                'O3_2': 0.2,
            },
        },
    }
    pars = Pars(sampled_pars, meta_pars)

    cube_dir = os.path.join(utils.TEST_DIR, 'test_data')

    ### Loading data vector 
    datafile = "/Users/jiachuanxu/Workspace/KL_measurement/kl-tools_spencer/data/simudata_3_noisy.fits"
    datavector = GrismDataVector(file=datafile)

    #-----------------------------------------------------------------
    # Setup sampled posterior

    pars_order = pars.sampled.pars_order

    log_posterior = LogPosterior(pars, datavector, likelihood='grism')

    #-----------------------------------------------------------------
    # Setup sampler

    ndims = log_posterior.ndims
    nwalkers = 2*ndims

    if sampler == 'zeus':
        print('Setting up KLensZeusRunner')

        runner = KLensZeusRunner(
            nwalkers, ndims, log_posterior, datavector, pars
            )

    elif sampler == 'emcee':
        print('Setting up KLensEmceeRunner')

        runner = KLensEmceeRunner(
            nwalkers, ndims, log_posterior, datavector, pars
            )

    print('>>>>>>>>>> Starting mcmc run <<<<<<<<<<')
    runner.run(pool, nsteps, vb=True)

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
    
    exit(0)
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

def _callback_():
    print("Worker exit!!!!!")

if __name__ == '__main__':
    args = parser.parse_args()
    pool = schwimmbad.choose_pool(
        mpi=args.mpi, processes=args.ncores
        )

    if isinstance(pool, MPIPool):
        if not pool.is_master():
            pool.wait(callback=_callback_)
            sys.exit(0)
    rc = main(args, pool)

    if rc == 0:
        print('All tests ran succesfully')
    else:
        print(f'Tests failed with return code of {rc}')
