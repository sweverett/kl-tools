# This is to ensure that numpy doesn't have
# OpenMP optimizations clobber our own multiprocessing
_USER_RUNNER_CLASS_ = False
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
from mpi4py import MPI
comm = MPI.COMM_WORLD
from schwimmbad import MPIPool
from argparse import ArgumentParser
from astropy.units import Unit
import galsim as gs
import matplotlib.pyplot as plt
import zeus
import emcee

import utils
from mcmc import KLensZeusRunner, KLensEmceeRunner
import priors
from muse import MuseDataCube
import likelihood
from parameters import Pars
from likelihood import LogPosterior, GrismLikelihood, get_GlobalDataVector
from velocity import VelocityMap
from grism import GrismDataVector
from datavector import FiberDataVector

import ipdb

fiber_blur = 3.4 # pixels
atm_psf_fwhm = 1.0 # arcsec
fiber_rad = 0.75 # arcsec
fiber_offset_x = 1.5 # arcsec
fiber_offset_y = 1.5 # arcsec
exptime_nominal = 600 # seconds
ADD_NOISE = False

default_obs_conf = [
     { # observation 1: fiber position 1
         'OBSINDEX': 0,
         'INSTNAME': "DESI",
         'OBSTYPE': 1,
         'BANDPASS': "../data/Bandpass/DESI/z.dat",
         'SKYMODEL': "../data/Skyspectra/skysb_grey.dat",
         'NAXIS': 2,
         'NAXIS1': 32,# Nx 
         'NAXIS2': 30,# Ny
         'PIXSCALE': 0.22,
         'RSPEC': 5332,
         'PSFTYPE': "airy_fwhm",
         'PSFFWHM': atm_psf_fwhm,
         'DIAMETER': 332.42,
         'EXPTIME': exptime_nominal,
         'GAIN': 1.0,
         'NOISETYP': 'ccd',
         'NOISESIG': 1.0,
         'RDNOISE': 8.6,
         'ADDNOISE': ADD_NOISE,
         'FIBERDX': fiber_offset_x,
         'FIBERDY': 0.0,
         'FIBERRAD': fiber_rad,
         'FIBRBLUR': fiber_blur,
     },
     { # observation 2: fiber position 2
         'OBSINDEX': 1,
         'INSTNAME': "DESI",
         'OBSTYPE': 1,
         'BANDPASS':"../data/Bandpass/DESI/z.dat",
         'SKYMODEL': "../data/Skyspectra/skysb_grey.dat",
         'NAXIS': 2,
         'NAXIS1': 32,# Nx 
         'NAXIS2': 30,# Ny
         'PIXSCALE': 0.22,
         'RSPEC': 5332,
         'PSFTYPE': "airy_fwhm",
         'PSFFWHM': atm_psf_fwhm,
         'DIAMETER': 332.42,
         'EXPTIME': exptime_nominal,
         'GAIN': 1.0,
         'NOISETYP': 'ccd',
         'NOISESIG': 1.0,
         'RDNOISE': 8.6,
         'ADDNOISE': ADD_NOISE,
         'FIBERDX': -1*fiber_offset_x,
         'FIBERDY': 0.0,
         'FIBERRAD': fiber_rad,
         'FIBRBLUR': fiber_blur,
     },
     { # observation 3: fiber position 3
         'OBSINDEX': 2,
         'INSTNAME': "DESI",
         'OBSTYPE': 1,
         'BANDPASS': "../data/Bandpass/DESI/z.dat",
         'SKYMODEL': "../data/Skyspectra/skysb_grey.dat",
         'NAXIS': 2,
         'NAXIS1': 32,# Nx 
         'NAXIS2': 30,# Ny
         'PIXSCALE': 0.22,
         'RSPEC': 5332,
         'PSFTYPE': "airy_fwhm",
         'PSFFWHM': atm_psf_fwhm,
         'DIAMETER': 332.42,
         'EXPTIME': exptime_nominal,
         'GAIN': 1.0,
         'NOISETYP': 'ccd',
         'NOISESIG': 1.0,
         'RDNOISE': 8.6,
         'ADDNOISE': ADD_NOISE,
         'FIBERDX': 0, 
         'FIBERDY': fiber_offset_y,
         'FIBERRAD': fiber_rad,
         'FIBRBLUR': fiber_blur,
     },
    { # observation 4: fiber position 4
         'OBSINDEX': 3,
         'INSTNAME': "DESI",
         'OBSTYPE': 1,
         'BANDPASS': "../data/Bandpass/DESI/z.dat",
         'SKYMODEL': "../data/Skyspectra/skysb_grey.dat",
         'NAXIS': 2,
         'NAXIS1': 32,# Nx 
         'NAXIS2': 30,# Ny
         'PIXSCALE': 0.22,
         'RSPEC': 5332,
         'PSFTYPE': "airy_fwhm",
         'PSFFWHM': atm_psf_fwhm,
         'DIAMETER': 332.42,
         'EXPTIME': exptime_nominal,
         'GAIN': 1.0,
         'NOISETYP': 'ccd',
         'NOISESIG': 1.0,
         'RDNOISE': 8.6,
         'ADDNOISE': ADD_NOISE,
         'FIBERDX': 0, 
         'FIBERDY': -1.0*fiber_offset_y,
         'FIBERRAD': fiber_rad,
        'FIBRBLUR': fiber_blur,
     },
    { # observation 5: fiber position 5
         'OBSINDEX': 4,
         'INSTNAME': "DESI",
         'OBSTYPE': 1,
         'BANDPASS': "../data/Bandpass/DESI/z.dat",
         'SKYMODEL': "../data/Skyspectra/skysb_grey.dat",
         'NAXIS': 2,
         'NAXIS1': 32,# Nx 
         'NAXIS2': 30,# Ny
         'PIXSCALE': 0.22,
         'RSPEC': 5332,
         'PSFTYPE': "airy_fwhm",
         'PSFFWHM': atm_psf_fwhm,
         'DIAMETER': 332.42,
         'EXPTIME': 180,
         'GAIN': 1.0,
         'NOISETYP': 'ccd',
         'NOISESIG': 1.0,
         'RDNOISE': 8.6,
         'ADDNOISE': ADD_NOISE,
         'FIBERDX': 0, 
         'FIBERDY': 0,
         'FIBERRAD': fiber_rad,
        'FIBRBLUR': fiber_blur,
     },
    { # observation 6: photometry image
        'OBSINDEX': 5,
        'INSTNAME': "DESI",
        'OBSTYPE': 0,
        'BANDPASS': "../data/Bandpass/DESI/z.dat",
        'NAXIS': 2,
        'NAXIS1': 32,# Nx
        'NAXIS2': 30,# Ny
        'PIXSCALE': 0.22,
        'PSFTYPE': "airy_fwhm",
        'PSFFWHM': 1.0,
        'DIAMETER': 332.42,
        'EXPTIME': 180*2,
        'GAIN': 1.0,
        'NOISETYP': 'ccd',
        'NOISESIG': 1.0,
        'SKYLEVEL': 40.4*180*2,
        'RDNOISE': 2.6,
        'ADDNOISE': ADD_NOISE,
    }
]

def choose_fiber_conf(case):
    if case==0:
        # major+minor, 1+4 fibers
        obs_conf = copy.deepcopy(default_obs_conf)
    elif case==1:
        # major, 1+2 fibers
        obs_conf = copy.deepcopy([default_obs_conf[i] for i in [0,1,4,5]])
        for i,conf in enumerate(obs_conf):
            conf['OBSINDEX'] = i
            if i<2:
                conf['EXPTIME'] = 2 * conf['EXPTIME']
    elif case==2:
        # minor, 1+2 fibers
        obs_conf = copy.deepcopy([default_obs_conf[i] for i in [2,3,4,5]])
        for i,conf in enumerate(obs_conf):
            conf['OBSINDEX'] = i
            if i<2:
                conf['EXPTIME'] = 2 * conf['EXPTIME']
    elif case==3:
        # semi-major+semi-minor, 1+2 fibers
        obs_conf = copy.deepcopy([default_obs_conf[i] for i in [0,2,4,5]])
        for i,conf in enumerate(obs_conf):
            conf['OBSINDEX'] = i
            if i<2:
                conf['EXPTIME'] = 2 * conf['EXPTIME']
    else:
        print(f'Fiber configuration case {case} is not implemented yet!')
        exit(-1)
    return obs_conf

parser = ArgumentParser()

parser.add_argument('nsteps', type=int,
                    help='Number of mcmc iterations per walker')
parser.add_argument('-sampler', type=str, choices=['zeus', 'emcee'],
                    default='emcee',
                    help='Which sampler to use for mcmc')
parser.add_argument('-run_name', type=str, default='',
                    help='Name of mcmc run')
parser.add_argument('-Iflux', type=int, default=0,help='Flux bin index')
parser.add_argument('-sini', type=int, default=0, help='sin(i) bin index')
parser.add_argument('-hlr', type=int, default=1.5, help='image hlr bin index')
parser.add_argument('-fiberconf', type=int, default=0, help='fiber conf index')
parser.add_argument('--show', action='store_true', default=False,
                    help='Set to show test plots')

group = parser.add_mutually_exclusive_group()
group.add_argument('--mpi', dest='mpi', default=False, action='store_true',
                   help='Run with MPI.')
group.add_argument('-ncores', default=1, type=int,
                    help='Number of processes (uses `multiprocessing` sequencial pool).')

def main(args, pool):

    # TODO: Try making datacube a global variable, as it may significantly
    # decrease execution time due to pickling only once vs. every model eval
    nsteps = args.nsteps
    sampler = args.sampler
    ncores = args.ncores
    mpi = args.mpi
    run_name = args.run_name
    show = args.show

    flux_scaling_power = args.Iflux
    flux_scaling = 1.58489**flux_scaling_power
    sini = 0.05 + 0.1*args.sini
    assert (0<sini<1)
    hlr = 0.5 + 0.5*args.hlr 
    fiber_conf = args.fiberconf

    ### Initialization
    sampled_pars = [
        'g1',
        'g2',
        'theta_int',
        'sini',
        'v0',
        'vcirc',
        'rscale',
        'hlr',
        ]
    sampled_pars_value = [0.0, 0.0, 0, sini, 0.0, 300.0, hlr, hlr]
    sampled_pars_std = np.array(
        [0.01, 0.01, 0.01, 0.01, 2, 5, 0.05, 0.01]
        )/1000
    sampled_pars_value_dict = {k:v for k,v in zip(sampled_pars, sampled_pars_value)}
    meta_pars = {
        ### shear and alignment
        'g1': 'sampled',
        'g1': 'sampled',
        'theta_int': 'sampled',
        'sini': 'sampled',
        ### priors
        'priors': {
            'g1': priors.UniformPrior(-0.2, 0.2),
            'g2': priors.UniformPrior(-0.2, 0.2),
            'theta_int': priors.UniformPrior(-np.pi, np.pi),
            'sini': priors.UniformPrior(0, 1.),
            'v0': priors.GaussPrior(0, 10),
            #'vcirc': priors.UniformPrior(10, 800),
            'vcirc': priors.GaussPrior(300, 80, clip_sigmas=3),
            'rscale': priors.UniformPrior(0, 5),
            'hlr': priors.UniformPrior(0, 5),
        },
        ### velocity model
        'velocity': {
            'model': 'default',
            'v0': 'sampled',
            'vcirc': 'sampled',
            'rscale': 'sampled',
        },
        ### intensity model
        'intensity': {
            'type': 'inclined_exp',
            'flux': 1.0, # counts
            'hlr': 'sampled',
            #'hlr': 2.5
        },
        ### misc
        'units': {
            'v_unit': Unit('km/s'),
            'r_unit': Unit('arcsec')
        },
        'run_options': {
            'run_mode': 'ETC',
            #'remove_continuum': True,
            'use_numba': False
        },
        ### 3D underlying model dimension 
        'model_dimension':{
            'Nx': 64,
            'Ny': 62,
            'scale': 0.11, # arcsec
            'lambda_range': [851, 856], # 1190-1370          
            'lambda_res': 0.08, # nm
            'lambda_unit': 'nm',
        },
        ### SED model
        'sed':{
            'template': '../data/Simulation/GSB2.spec',
            'wave_type': 'Ang',
            'flux_type': 'flambda',
            'z': 0.3,
            'wave_range': [500., 3000.], # nm
            # obs-frame continuum normalization (nm, erg/s/cm2/nm)
            'obs_cont_norm': [850, 2.6e-15*flux_scaling],
            # a dict of line names and obs-frame flux values (erg/s/cm2)
            'lines':{
                'Ha': 1.25e-15*flux_scaling,
                'O2': [1.0e-14*flux_scaling, 1.2e-14*flux_scaling],
                'O3_1': 1.0e-14*flux_scaling,
                'O3_2': 1.2e-14*flux_scaling,
            },
            # intrinsic linewidth in nm
            'line_sigma_int':{
                'Ha': 0.05,
                'O2': [0.2, 0.2],
                'O3_1': 0.2,
                'O3_2': 0.2,
            },
        },
        ### observation configurations
        'obs_conf':  choose_fiber_conf(fiber_conf)
    }
    pars = Pars(sampled_pars, meta_pars)

    #cube_dir = os.path.join(utils.TEST_DIR, 'test_data')
    cube_dir = os.path.join("/xdisk/timeifler/jiachuanxu/kl_fiber")

    ### Loading data vector 
    #datafile = "/Users/jiachuanxu/Workspace/KL_measurement/kl-tools_spencer/data/simufiber_3.fits"
    #dv = FiberDataVector(file=datafile)

    #-----------------------------------------------------------------
    # Setup sampled posterior

    pars_order = pars.sampled.pars_order
    # log_posterior arguments: theta, data, pars
    log_posterior = LogPosterior(pars, None, likelihood='fiber', sampled_theta_fid=sampled_pars_value)

    #-----------------------------------------------------------------
    # Setup sampler

    ndims = log_posterior.ndims
    nwalkers = 2*ndims
    size = comm.Get_size()
    rank = comm.Get_rank()

    ### use the runner class approach
    if _USER_RUNNER_CLASS_:
        if sampler == 'zeus':
            print('Setting up KLensZeusRunner')

            runner = KLensZeusRunner(
                nwalkers, ndims, log_posterior, dv, pars
                )

        elif sampler == 'emcee':
            print('Setting up KLensEmceeRunner')

            runner = KLensEmceeRunner(
                nwalkers, ndims, log_posterior, dv, pars
                )

        if (not args.mpi) and args.ncores==1:
            print("Single process!")
            pool = None 
        else:
            print("Multi-process! MPI=%s, ncores=%d"%(args.mpi, args.ncores))
            pool = schwimmbad.choose_pool(
                mpi=args.mpi, processes=args.ncores
                )
        if isinstance(pool, MPIPool):
            if not pool.is_master():
                print("[%d/%d] wait"%(rank, size))
                pool.wait(callback=_callback_)
                sys.exit(0)
            else:
                print("[%d/%d] go"%(rank, size))
        print('>>>>>>>>>> [%d/%d] Starting mcmc run <<<<<<<<<<'%(rank, size))
        runner.run(pool, nsteps, vb=True)

        runner.burn_in = nsteps // 2

        outdir = os.path.join(
            "/xdisk/timeifler/jiachuanxu/kl_fiber", run_name
        )
        utils.make_dir(outdir)
        if (sampler == 'zeus') and ((ncores > 1) or (mpi == True)):
            # The sampler isn't pickleable for some reason in this scenario
            # Save whole chain
            outfile = os.path.join(outdir, 'test-mcmc-chain.pkl')
            chain = runner.sampler.get_chain(flat=True)
            print(f'pickling chain to {outfile}')
            with open(outfile, 'wb') as f:
                pickle.dump(chain, f)
        else:
            outfile = os.path.join(outdir, 'sampler_%d.pkl'%flux_scaling_power)
            print(f'Pickling sampler to {outfile}')
            with open(outfile, 'wb') as f:
                pickle.dump(runner.sampler, f)

            outfile = os.path.join(outdir, 'runner_%d.pkl'%flux_scaling_power)
            print(f'Pickling runner to {outfile}')
            with open(outfile, 'wb') as f:
                pickle.dump(runner, f)
    else:
        if (not args.mpi) and args.ncores==1:
            print("Single process!")
            pool = None 
        else:
            print("Multi-process! MPI=%s, ncores=%d"%(args.mpi, args.ncores))
            pool = schwimmbad.choose_pool(
                mpi=args.mpi, processes=args.ncores
                )
        if isinstance(pool, MPIPool):
            if not pool.is_master():
                print("[%d/%d] wait"%(rank, size))
                pool.wait(callback=_callback_)
                sys.exit(0)
            else:
                print("[%d/%d] go"%(rank, size))
        print('>>>>>>>>>> [%d/%d] Starting EMCEE run <<<<<<<<<<'%(rank, size))
        MCMCsampler = emcee.EnsembleSampler(
            nwalkers, ndims, log_posterior,
            args=[None, pars], pool=pool
            )
        p0 = emcee.utils.sample_ball(sampled_pars_value, sampled_pars_std, 
            size=nwalkers)
        
        MCMCsampler.run_mcmc(p0, nsteps, progress=True)

        outdir = os.path.join(
            "/xdisk/timeifler/jiachuanxu/kl_fiber", run_name
        )
        utils.make_dir(outdir)
        outfile = os.path.join(outdir, 
            'sampler_%s_sini%.2f_hlr%.2f_fiberconf%d.pkl'%(flux_scaling_power, sini, hlr, fiber_conf))
        print(f'Pickling sampler to {outfile}')
        with open(outfile, 'wb') as f:
            pickle.dump(MCMCsampler, f)
    outfile = os.path.join(outdir, 'dv_%s_sini%.2f_hlr%.2f_fiberconf%d.pkl'%(flux_scaling_power, sini, hlr, fiber_conf))
    print(f'Saving data vector to {outfile}')
    dv = get_GlobalDataVector(0)
    dv.to_fits(outfile, overwrite=True)
    
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
    
    rc = main(args, None)

    if rc == 0:
        print('All tests ran succesfully')
    else:
        print(f'Tests failed with return code of {rc}')
