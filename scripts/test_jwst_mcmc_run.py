# This is to ensure that numpy doesn't have
# OpenMP optimizations clobber our own multiprocessing
import os
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
import sys, copy, os, glob, re
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
    print(f'[{rank}/{size}] Calling MPI')
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
from kl_tools.mcmc import KLensZeusRunner, KLensEmceeRunner, KLensPocoRunner, KLensUltranestRunner
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
parser.add_argument('-ID', type=int, default=-1, help='ID of the object to fit')
parser.add_argument('-sampler', type=str, choices=['zeus', 'emcee', 'poco', 'ultranest'],
                    default='emcee',
                    help='Which sampler to use for mcmc')
parser.add_argument('--not_overwrite', action='store_true', default=False,
                    help='Not Overwrite existing outputs')
parser.add_argument('-nsteps', type=int, default=-1,
                    help='Number of mcmc iterations per walker')
parser.add_argument('-nparticles', type=int, default=512,
                    help='[pocoMC] Number of effective particles')
parser.add_argument('-n_total', type=int, default=4096,
                    help='[pocoMC] Total number of effectively independent samples')
parser.add_argument('--run_from_params', action='store_true', default=False,
                    help='Run mock analysis from fiducial parameters rather than from data file')

group = parser.add_mutually_exclusive_group()
group.add_argument('-ncores', default=1, type=int,
                    help='Number of processes (uses `multiprocessing`)')
group.add_argument('--mpi', dest='mpi', default=False, action='store_true',
                   help='Run with MPI.')

def get_dummy_data_vector(sampled_theta_fid, meta_dict):
    ''' Get a placeholder datavector object from obs_conf '''
    dummy_header = {k:v for k,v in sampled_theta_fid.items()}
    dummy_header["z_spec"] = meta_dict["sed"]["z"]
    Nobs = len(meta_dict["obs_conf"])
    dummy_header["OBSNUM"] = Nobs
    dummy_data = []
    dummy_data_header = []
    dummy_noise = []
    dummy_psf = []
    dummy_mask = []
    for i in range(Nobs):
        _conf = meta_dict["obs_conf"][i]
        data_shape = [_conf["NAXIS2"], _conf["NAXIS1"]]
        dummy_data.append(np.zeros(data_shape))
        dummy_noise.append(np.ones(data_shape))
        dummy_data_header.append(copy.deepcopy(_conf))
        if _conf["PSFTYPE"] == "data":
            with fits.open(_conf["PSFDATA"]) as hdul:
                psf_data = gs.Image(hdul[1].data/hdul[1].data.sum(), 
                    scale=hdul[1].header["PIXELSCL"], copy=True)
            dummy_psf.append(gs.InterpolatedImage(psf_data, flux=1))
        else:
            dummy_psf.append(None)
        if "MASKDATA" in _conf:
            with fits.open(_conf["MASKDATA"]) as hdul:
                dummy_mask.append(hdul[1].data)
        else:
            dummy_mask.append(np.ones(data_shape))
    dummy_datavector = GrismDataVector(header=dummy_header,
        data_header=dummy_data_header, data=dummy_data, 
        noise=dummy_noise, psf=dummy_psf, mask=dummy_mask)
    return dummy_datavector

def main(args):
    ''' Fit KL model for JWST objects
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
            print(f'- {key}: ref = {fidvals[key]:.2e}; init = {ball_mean[key]:.2e} +- {ball_std[key]:.2e}')
    nsteps = mcmc_dict["nsteps"] if args.nsteps==-1 else args.nsteps
    objID = mcmc_dict["objid"] if args.ID == -1 else args.ID
    ncores = args.ncores
    mpi = args.mpi
    outdir = mcmc_dict["output"]
    utils.make_dir(outdir)

    ######################### Build Data Vector ################################
    if args.run_from_params:
        #-----------------------------------------------------------------
        # Build data vector from fiducial parameters
        #-----------------------------------------------------------------
        # fiducial parameter of the evaluation
        sampled_theta_fid = mcmc_dict["sampled_theta_fid"]
        # generate mock data and write into disk ONLY IN THE MAIN PROCESS
        # such that all the processes share the same noise
        if rank==0:
            # prepare a dummy data vector, to read real PSF and image mask
            dummyvector = get_dummy_data_vector(sampled_theta_fid, meta_dict)
            pars = Pars(sampled_pars, meta_dict, derived)
            log_posterior = LogPosterior(pars, dummyvector, likelihood='grism', 
                sampled_theta_fid=sampled_theta_fid,
                write_mock_data_to=mcmc_dict["mockdata_output"])
            if size>1:
                comm.Barrier()
        else:
            if size>1:
                comm.Barrier()
        datavector = GrismDataVector(file=mcmc_dict["mockdata_output"])
    else:
        #-----------------------------------------------------------------
        # Load data vector from existing data
        #-----------------------------------------------------------------
        ### Step 1: load JWST data and figure out source information
        if os.path.exists(mcmc_dict["data_filename_fmt"]%objID):
            data_file = mcmc_dict["data_filename_fmt"]%objID
        else:
            data_file = os.path.join("../data/jwst", 
                        mcmc_dict["data_filename_fmt"]%objID)
        assert os.path.exists(data_file), f'Input data {data_file} not exists!'
        if rank==0:
            print("Reading JWST observation ID-%d"%(objID))
        data_hdul = fits.open(data_file)
        # read redshift from FITS data file and overwrite the default z
        eml_name = data_hdul[0].header["name_line_exp"] # emission line name
        eml_obscen = data_hdul[0].header["fit_center_um"]*u.um
        data_hdul.close()
        src_z = eml_obscen/LINE_LAMBDAS[eml_name] - 1
        meta_dict['sed']['z'] = src_z.to('1').value
        ### Loading data vector
        datavector = GrismDataVector(file=data_file)
    # Then, initialize Pars object
    pars = Pars(sampled_pars, meta_dict, derived)
    pars_order = pars.sampled.pars_order
    if rank==0:
        print(pars_order)

    ########################### Setup Posterior ################################
    #if args.run_from_params:
    #    log_posterior = LogPosterior(pars, datavector, likelihood='grism', 
    #        sampled_theta_fid=sampled_theta_fid,
    #        write_mock_data_to=mcmc_dict["mockdata_output"])
    #else:
    prior_type = likelihood.get_sampler_prior_type(args.sampler)
    log_posterior = LogPosterior(pars, datavector, likelihood='grism', prior=prior_type)
    # Example posterior evaluation
    if rank==0:
        #ipdb.set_trace()
        print("---------------------------------------------")
        print("Example likelihood evaluation: ")
        try:
            example_theta = np.array([sampled_theta_fid[key] for key in sampled_pars])
            print(f'Use `sampled_theta_fid`')
            print("Evaluated at: ", sampled_theta_fid)
        except:
            example_theta = np.array([fidvals[key] for key in sampled_pars])
            print(f'Can not find `sampled_theta_fid`, use reference dist center')
            print("Evaluated at: ", fidvals)
        loglike_test = log_posterior.log_likelihood(example_theta, None)
        print(f'log likelihood = {loglike_test}')
        if args.sampler != 'ultranest':
            logpost_test = log_posterior(example_theta, None, pars)
            print(f'log posterior  = {logpost_test}')
        print("---------------------------------------------")
        #ipdb.set_trace()
        comm.Barrier()
    else:
        comm.Barrier()
    # test pickle dump to see the size of the posterior function
    # with open("test_JWST_post_func.pkl", 'wb') as f:
    #         pickle.dump(log_posterior, f)

    ############################## Setup Runner ################################
    ndims = log_posterior.ndims
    if (args.sampler != 'poco') and (args.sampler != 'ultranest'):
        nwalkers = 2*ndims
    else:
        nwalkers = args.nparticles
        nsteps = args.n_total
    #-----------------------------------------------------------------
    # zeus
    #-----------------------------------------------------------------
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
    #-----------------------------------------------------------------
    # emcee
    #-----------------------------------------------------------------
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
    #-----------------------------------------------------------------
    # pocoMC
    #-----------------------------------------------------------------
    elif args.sampler == 'poco':
        if rank==0:
            print('Setting up KLensPocoRunner')
        ### resume pocoMC run from previous state
        checkpoint_root = os.path.join(outdir, "pmc")
        statefiles = glob.glob(checkpoint_root + "_*.state")
        max_checkpoint = 0
        for statefile in statefiles:
            match = re.match(checkpoint_root + "_(\S*).state", statefile)
            pattern = match.groups(0)[0]
            if pattern=='final':
                max_checkpoint = 'final'
                break
            else:
                max_checkpoint = max(max_checkpoint, int(pattern))
        if max_checkpoint==0:
            resume_state_path = None
            if rank==0:
                print(f'Start new pocoMC run...')
        else:
            resume_state_path = os.path.join(outdir, 
                            f'pmc_{max_checkpoint}.state')
            if rank==0:
                print(f'Resume pocoMC from {resume_state_path}!')
        ### init PocoMC runner
        runner = KLensPocoRunner(
                nwalkers, ndims, 
                log_posterior.log_likelihood, log_posterior.log_prior,
                None, pars,
                loglike_kwargs={'return_derived_lglike': True},
                output_dir=outdir, # output_label=None,
                resume_state_path=resume_state_path,
            )
    #-----------------------------------------------------------------
    # UltraNest
    #-----------------------------------------------------------------
    elif args.sampler == 'ultranest':
        if rank==0:
            print('Setting up KLensUltranestRunner')
        ### init UltraNest runner
        if args.not_overwrite:
            resume = True
        else:
            resume = "subfolder"
        runner = KLensUltranestRunner(
                nwalkers, ndims, 
                log_posterior.log_likelihood._call_no_args,
                log_posterior.log_prior,
                None, 
                pars,
                loglike_kwargs={'return_derived_lglike': True},
                out_dir=outdir, 
                resume=resume,
            )
    else:
        print(f'sampler {args.sampler} is not supported!')
        exit(-1)
    print(f'[{rank}/{size}] We arrive here')
    pool = schwimmbad.choose_pool(
        mpi=args.mpi, processes=args.ncores
        )

    ############################### Run Sampler ################################
    print('>>>>>>>>>> Starting mcmc run <<<<<<<<<<')
    # get initial starting points
    p0_mean = np.array([ball_mean[_key] for _key in sampled_pars])
    p0_std = np.array([ball_std[_key] for _key in sampled_pars])
    p0 = p0_mean + np.random.randn(nwalkers, ndims)*p0_std
    
    #-----------------------------------------------------------------
    # Run sampler
    #-----------------------------------------------------------------
    #ipdb.set_trace()
    runner.run(pool, nsteps, start=p0, vb=True)
    runner.burn_in = nsteps // 2

    #-----------------------------------------------------------------
    # Save chains
    #-----------------------------------------------------------------
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
    elif (args.sampler == 'poco'):
        ### retrieve MCMC samples, weights, lglikes, lgposts, blobs
        results = runner.sampler.posterior(return_blobs=True, resample=False)
        logZ, logZerr = runner.sampler.evidence()
        data_block = [results[0], 
                    results[1][:,np.newaxis],
                    results[2][:,np.newaxis],
                    results[3][:,np.newaxis]]
        header = f'# logZ {logZ}\n# logZerr {logZerr}'
        header += "\n# " + " ".join(sampled_pars) + " weight loglike logpost"
        # blobs: derived parameters
        if pars.derived.keys() is not None:
            for dev_par in pars.derived.keys():
                header = header + " " + dev_par
                data_block.append(results[4][dev_par][:,np.newaxis])
        data_block = np.hstack(data_block)
        # save chain
        np.savetxt(os.path.join(outdir, "poco_chain.txt"), data_block, 
                  header = header, comments="")
    elif (args.sampler == 'ultranest'):
        pass

    ############################### Produce MAP ################################


    return 0

def _callback_():
    print("Worker exit!!!!!")

if __name__ == '__main__':
    args = parser.parse_args()
    rc = main(args)

    if rc == 0:
        print('All tests ran succesfully')
    else:
        print(f'Tests failed with return code of {rc}')
