# This is to ensure that numpy doesn't have
# OpenMP optimizations clobber our own multiprocessing
import os
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
import sys, copy, os
sys.path.insert(0, './grism_modules')
import pickle
import schwimmbad
import mpi4py
from mpi4py import MPI
try:
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
except:
    rank = 0
    size = 1
from schwimmbad import MPIPool
from argparse import ArgumentParser
from astropy.units import Unit
import galsim as gs
import matplotlib.pyplot as plt
import zeus
import astropy.io.fits as fits
import astropy.units as u

import utils
from mcmc import KLensZeusRunner, KLensEmceeRunner
import priors
from muse import MuseDataCube
import likelihood
from parameters import Pars
from emission import LINE_LAMBDAS
from likelihood import LogPosterior, GrismLikelihood
from velocity import VelocityMap
from grism import GrismDataVector

import ipdb

parser = ArgumentParser()

parser.add_argument('yaml', type=str, help='Input YAML file')
parser.add_argument('-ID', type=int, default=181661, help='ID of the object to fit')
parser.add_argument('-sampler', type=str, choices=['zeus', 'emcee'],
                    default='emcee',
                    help='Which sampler to use for mcmc')
parser.add_argument('-run_name', type=str, default='',
                    help='Name of mcmc run')
parser.add_argument('--show', action='store_true', default=False,
                    help='Set to show test plots')
parser.add_argument('-nsteps', type=int, default=-1,
                    help='Number of mcmc iterations per walker')

group = parser.add_mutually_exclusive_group()
group.add_argument('-ncores', default=1, type=int,
                    help='Number of processes (uses `multiprocessing`)')
group.add_argument('--mpi', dest='mpi', default=False, action='store_true',
                   help='Run with MPI.')

def main(args, pool):
    ''' Fit KL model for JWST objects
    # TODO: Try making datacube a global variable, as it may significantly
    # decrease execution time due to pickling only once vs. every model eval
    '''
    # read arguments
    global rank, size
    ########################### Initialization #################################
    # First, parse the YAML file
    sampled_pars, fidvals, latex_labels, derived, meta_dict, mcmc_dict, ball_mean, ball_std, ball_proposal = utils.parse_yaml(args.yaml)
    if rank==0:
        print(f'Sampled parameters: {sampled_pars}')
    nsteps = mcmc_dict["nsteps"] if args.nsteps==-1 else args.nsteps
    objID = mcmc_dict["objid"] if args.ID == -1 else args.ID
    ncores = args.ncores
    mpi = args.mpi
    run_name = args.run_name
    show = args.show
    outdir = os.path.join(utils.TEST_DIR, 'jwst-mcmc-run', run_name)
    utils.make_dir(outdir)

    ### Step 1: load JWST data and figure out source information
    data_root = "../data/jwst"
    data_file = os.path.join(data_root, "data_compile_short_withmask_GDS_ID%d.fits"%objID)
    print("Reading JWST observation ID-%d"%(objID))
    data_hdul = fits.open(data_file)
    
    ''' read important information from fits file and overwrite YAML file
    Information to be overwrite
        - redshift
        - image hlr
        - image flux
        - emission line flux
        - continuum flux?
        - 
    '''
    eml_name = data_hdul[0].header["name_line_exp"] # emission line name
    src_z = data_hdul[0].header["fit_center_um"]*u.um/LINE_LAMBDAS[eml_name]-1
    meta_dict['sed']['z'] = src_z.to('1').value

    pars = Pars(sampled_pars, meta_dict)

    cube_dir = os.path.join(utils.TEST_DIR, 'test_data')

    ### Loading data vector 
    datafile = "/home/u17/jiachuanxu/kl-tools/data/jwst/data_compile_short_withmask_GDS_ID%d.fits"%(objID)
    #datafile = "/Users/jiachuanxu/Workspace/KL_measurement/kl-tools_spencer/data/jwst/data_compile_short_withmask_GDS_ID%d.fits"%(objID)
    datavector = GrismDataVector(file=datafile)

    #-----------------------------------------------------------------
    # Setup sampled posterior

    pars_order = pars.sampled.pars_order
    print(pars_order)

    log_posterior = LogPosterior(pars, datavector, likelihood='grism')

    #-----------------------------------------------------------------
    # Setup sampler

    ndims = log_posterior.ndims
    nwalkers = 2*ndims

    if args.sampler == 'zeus':
        print('Setting up KLensZeusRunner')

        runner = KLensZeusRunner(
            nwalkers, ndims, log_posterior, datavector, pars
            )

    elif args.sampler == 'emcee':
        print('Setting up KLensEmceeRunner')

        runner = KLensEmceeRunner(
            nwalkers, ndims, log_posterior, datavector, pars
            )

    print('>>>>>>>>>> Starting mcmc run <<<<<<<<<<')
    runner.run(pool, nsteps, vb=True)

    runner.burn_in = nsteps // 2

    if (args.sampler == 'zeus') and ((ncores > 1) or (mpi == True)):
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

    if args.sampler == 'emcee':
        blobs = runner.sampler.blobs
    elif args.sampler == 'zeus':
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
