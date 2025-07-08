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
try:
    import schwimmbad
    import mpi4py
    from mpi4py import MPI
except:
    print("Can not import MPI, use single process")
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

import kl_tools.utils as utils
from kl_tools.mcmc import KLensZeusRunner, KLensEmceeRunner
import kl_tools.priors as priors
import kl_tools.likelihood as likelihood
from kl_tools.parameters import Pars
from kl_tools.emission import LINE_LAMBDAS
from kl_tools.likelihood import LogPosterior, GrismLikelihood
from kl_tools.velocity import VelocityMap
from kl_tools.grism_modules.grism import GrismDataVector

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
parser.add_argument('--not_overwrite', action='store_true', default=False,
                    help='Not Overwrite existing outputs')
parser.add_argument('-nsteps', type=int, default=-1,
                    help='Number of mcmc iterations per walker')

group = parser.add_mutually_exclusive_group()
group.add_argument('-ncores', default=1, type=int,
                    help='Number of processes (uses `multiprocessing`)')
group.add_argument('--mpi', dest='mpi', default=False, action='store_true',
                   help='Run with MPI.')

def main(args):
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
        for i in range(len(sampled_pars)):
            key = sampled_pars[i]
            print(f'- {key}: fiducial = {fidvals[key]:.2e}; init = {ball_mean[key]:.2e} +- {ball_std[key]:.2e}')
    nsteps = mcmc_dict["nsteps"] if args.nsteps==-1 else args.nsteps
    objID = mcmc_dict["objid"] if args.ID == -1 else args.ID
    ncores = args.ncores
    mpi = args.mpi
    run_name = args.run_name
    show = args.show

    ### Step 1: load JWST data and figure out source information
    data_root = "../data/jwst"
    data_file = os.path.join(data_root, mcmc_dict["data_filename_fmt"]%objID)
    if rank==0:
        print("Reading JWST observation ID-%d"%(objID))
    data_hdul = fits.open(data_file)
    outdir = mcmc_dict["output"]
    utils.make_dir(outdir)

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

    pars = Pars(sampled_pars, meta_dict, derived)

    cube_dir = os.path.join(utils.TEST_DIR, 'test_data')

    ### Loading data vector
    datavector = GrismDataVector(file=data_file)

    #-----------------------------------------------------------------
    # Setup sampled posterior

    pars_order = pars.sampled.pars_order
    if rank==0:
        print(pars_order)

    log_posterior = LogPosterior(pars, datavector, likelihood='grism')
    # test pickle dump to see the size of the posterior function
    # with open("test_JWST_post_func.pkl", 'wb') as f:
    #         pickle.dump(log_posterior, f)

    #-----------------------------------------------------------------
    # Setup sampler

    ndims = log_posterior.ndims
    nwalkers = 2*ndims

    if args.sampler == 'zeus':
        if rank==0:
            print('Setting up KLensZeusRunner')
        outfile = os.path.join(outdir, f'zeus-mcmc-runner.pkl')
        if os.path.exists(outfile) and args.not_overwrite:
            print(f'Continue from {outfile}')
            with open(outfile, mode='rb') as fp:
                runner = pickle.load(fp)
        else:
            runner = KLensZeusRunner(
                nwalkers, ndims, log_posterior, None, pars
            )

    elif args.sampler == 'emcee':
        if rank==0:
            print('Setting up KLensEmceeRunner')
        outfile = os.path.join(outdir, f'emcee-mcmc-runner.pkl')
        if os.path.exists(outfile) and args.not_overwrite:
            print(f'Continue from {outfile}')
            with open(outfile, mode='rb') as fp:
                runner = pickle.load(fp)
        else:
            runner = KLensEmceeRunner(
                nwalkers, ndims, log_posterior, None, pars
            )
    elif args.sampler == 'pocomc':
        if rank==0:
            print('Setting up KLensPocoRunner')
        outfile = os.path.join(outdir, f'poco-mcmc-runner.pkl')
        if os.path.exists(outfile) and args.not_overwrite:
            print(f'Continue from {outfile}')
            with open(outfile, mode='rb') as fp:
                runner = pickle.load(fp)
        else:
            runner = KLensPocoRunner(
                nwalkers, ndims, 
                log_posterior.log_likelihood, log_posterior.log_prior,
                None, pars,
                loglike_kwargs={'return_derived_lglike': True},
                loglike_args=[pars],
                output_dir=outdir, output_label="poco_checkpoint",
            )
    else:
        print(f'sampler {args.sampler} is not supported!')
        exit(-1)

    pool = schwimmbad.choose_pool(
        mpi=args.mpi, processes=args.ncores
        )

    # if isinstance(pool, MPIPool):
    #     if not pool.is_master():
    #         pool.wait(callback=_callback_)
    #         sys.exit(0)
    # with open("test_JWST_runner_class.pkl", 'wb') as f:
    #         pickle.dump(runner, f)

    #-----------------------------------------------------------------
    # Run sampler

    print('>>>>>>>>>> Starting mcmc run <<<<<<<<<<')
    # get initial starting points
    p0_mean = np.array([ball_mean[_key] for _key in sampled_pars])
    p0_std = np.array([ball_std[_key] for _key in sampled_pars])
    p0 = p0_mean + np.random.randn(nwalkers, ndims)*p0_std
    
    ### run sampler
    runner.run(pool, nsteps, start=p0, vb=True)
    runner.burn_in = nsteps // 2

    ### save results
    if (args.sampler == 'zeus'):
        # The sampler isn't pickleable for some reason in this scenario
        # Save whole chain
        outfile = os.path.join(outdir, f'zeus-mcmc-chain.pkl')
        chain = runner.sampler.get_chain(flat=True)
        print(f'pickling chain to {outfile}')
        with open(outfile, 'wb') as f:
            pickle.dump(chain, f)
    elif (args.sampler == 'emcee'):
        outfile = os.path.join(outdir, f'emcee-mcmc-sampler.pkl')
        print(f'Pickling sampler to {outfile}')
        with open(outfile, 'wb') as f:
            pickle.dump(runner.sampler, f)

        outfile = os.path.join(outdir, f'emcee-mcmc-runner.pkl')
        print(f'Pickling runner to {outfile}')
        with open(outfile, 'wb') as f:
            pickle.dump(runner, f)

        chain = runner.sampler.get_chain(flat=True)
        post = runner.sampler.get_log_prob(flat=True)
        blobs = np.array(runner.sampler.get_blobs(flat=True))
        blobs_name = ['logprior', 'loglike', ]
        if pars.derived.keys() is not None:
            blobs_name += pars.derived.keys()
        np.savetxt(os.path.join(outdir, "emcee_chain.txt"), 
            np.hstack([chain, post[:,np.newaxis], blobs]), 
                  header = "# " + " ".join(sampled_pars) + " logprob " + " ".join(blobs_name), 
                  comments="")
    elif (args.sampler == 'pocomc'):
        outfile = os.path.join(outdir, f'poco-mcmc-sampler.pkl')
        print(f'Pickling sampler to {outfile}')
        with open(outfile, 'wb') as f:
            pickle.dump(runner.sampler, f)

        outfile = os.path.join(outdir, f'poco-mcmc-runner.pkl')
        print(f'Pickling runner to {outfile}')
        with open(outfile, 'wb') as f:
            pickle.dump(runner, f)

        ### retrieve MCMC samples, weights, lglikes, lgposts, blobs
        results = runner.sampler.posterior(return_blobs=True)
        data_block = [results[0], 
                    results[1][:,np.newaxis],
                    results[2][:,np.newaxis],
                    results[3][:,np.newaxis]]
        header = "# " + " ".join(sampled_pars) + " weight loglike logpost"
        # blobs: derived parameters
        if pars.derived.keys() is not None:
            for dev_par in pars.derived.keys():
                header = header + " " + dev_par
                data_block.append(results[4][dev_par][:,np.newaxis])
        data_block = np.hstack(data_block)
        # save chain
        np.savetxt(os.path.join(outdir, "poco_chain.txt"), data_block, 
                  header = header, comments="")

    return 0

def _callback_():
    print("Worker exit!!!!!")

if __name__ == '__main__':
    args = parser.parse_args()
    # pool = schwimmbad.choose_pool(
    #     mpi=args.mpi, processes=args.ncores
    #     )

    # if isinstance(pool, MPIPool):
    #     if not pool.is_master():
    #         pool.wait(callback=_callback_)
    #         sys.exit(0)
    # rc = main(args, pool)
    rc = main(args)

    if rc == 0:
        print('All tests ran succesfully')
    else:
        print(f'Tests failed with return code of {rc}')
