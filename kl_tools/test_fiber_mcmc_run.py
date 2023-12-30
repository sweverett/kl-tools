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

########################### Parsing arguments ##################################
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
parser.add_argument('-hlr', type=int, default=0, help='image hlr bin index')
parser.add_argument('-sigma_int', type=int, default=1, help='Intrinsic eml sig')
parser.add_argument('-fiberconf', type=int, default=0, help='fiber conf index')
parser.add_argument('-EXPTIME', type=int, default=600, help='Exposure time in s')
parser.add_argument('--show', action='store_true', default=False,
                    help='Set to show test plots')
group = parser.add_mutually_exclusive_group()
group.add_argument('--mpi', dest='mpi', default=False, action='store_true',
                   help='Run with MPI.')
group.add_argument('-ncores', default=1, type=int,
                    help='Number of processes (uses `multiprocessing` sequencial pool).')
args = parser.parse_args()

fiber_blur = 3.4 # pixels
atm_psf_fwhm = 1.0 # arcsec
fiber_rad = 0.75 # arcsec
fiber_offset_x = 1.5 # arcsec
fiber_offset_y = 1.5 # arcsec
exptime_nominal = args.EXPTIME # seconds
ADD_NOISE = False

##################### Setting up observation configurations ####################
default_fiber_conf = {'INSTNAME': "DESI", 'OBSTYPE': 1,
     'SKYMODEL': "../data/Skyspectra/skysb_grey.dat", 'PSFTYPE': "airy_fwhm",
     'PSFFWHM': atm_psf_fwhm, 'DIAMETER': 332.42, 'EXPTIME': exptime_nominal,
     'GAIN': 1.0, 'NOISETYP': 'ccd', 'ADDNOISE': ADD_NOISE, 
     'FIBERRAD': fiber_rad, 'FIBRBLUR': fiber_blur,
}

default_photo_conf = {'INSTNAME': "CTIO/DECam", 'OBSTYPE': 0, 'NAXIS': 2,
    'NAXIS1': 32, 'NAXIS2': 30, 'PIXSCALE': 0.2637, 'PSFTYPE': "airy_fwhm",
    'PSFFWHM': 1.0, 'DIAMETER': 378.2856, 'EXPTIME': 150, 'GAIN': 4.0,
    'NOISETYP': 'ccd', 'RDNOISE': 7, 'ADDNOISE': ADD_NOISE,
}

default_obs_conf, _index_ = [], 0

### Fiber observations
emlines = ['O2', 'O3_1', 'O3_2', 'Ha']
blockids = [0, 2, 3, 4]
channels = ['b', 'r', 'r', 'z']
rdnoise = [3.41, 2.6, 2.6, 2.6]
### Choose fiber configurations
if args.fiberconf==0:
    offsets = [(fiber_offset_x, 0), (-fiber_offset_x, 0), 
               (0, fiber_offset_y), (0, -fiber_offset_y), (0,0)]
    OFFSETX = 1
elif args.fiberconf==1:
    offsets = [(fiber_offset_x, 0), (-fiber_offset_x, 0), (0,0)]
    OFFSETX = 2
elif args.fiberconf==2:
    offsets = [(0, fiber_offset_y), (0, -fiber_offset_y), (0,0)]
    OFFSETX = 2
elif args.fiberconf==3:
    offsets = [(fiber_offset_x, 0), (0, fiber_offset_y), (0,0)]
    OFFSETX = 2
else:
    print(f'Fiber configuration case {args.fiberconf} is not implemented yet!')
    exit(-1)

for eml, bid, chn, rdn in zip(emlines, blockids, channels, rdnoise):
    _bp = "../data/Bandpass/DESI/%s.dat"%(chn)
    for (dx, dy) in offsets:
        _conf = copy.deepcopy(default_fiber_conf)
        _conf.update({'OBSINDEX': _index_, 'SEDBLKID': bid, 'BANDPASS': _bp, 
            'RDNOISE': rdn, 'FIBERDX': dx, 'FIBERDY': dy})
        if np.abs(dx)>1e-3 and np.abs(dy)>1e-3:
            _conf.update({'EPTIME': exptime_nominal*OFFSETX})
        default_obs_conf.append(_conf)
        _index_+=1

### Photometry observations
# Assume 150s exptime for all exposures
# tune the sky level such that the 5-sigma limit are
# 𝑔=24.0, 𝑟=23.4 and 𝑧=22.5
# photometry_band = ['r', 'g', 'z']
# sky_levels = [40.4/4*150/15, 19.02/4*150/40, 40.4/4*150/5]
photometry_band = ['r', ]
sky_levels = [40.4/4*150/15,]

for chn, sky in zip(photometry_band, sky_levels):
    _bp = "../data/Bandpass/CTIO/DECam.%s.dat"%chn
    _conf = copy.deepcopy(default_photo_conf)
    _conf.update({"OBSINDEX": _index_, 'BANDPASS': _bp, "SKYLEVEL": sky})
    default_obs_conf.append(_conf)

####################### Main function ##########################################
def main(args, pool):

    # TODO: Try making datacube a global variable, as it may significantly
    # decrease execution time due to pickling only once vs. every model eval
    nsteps = args.nsteps
    sampler = args.sampler
    ncores = args.ncores
    mpi = args.mpi
    run_name = args.run_name
    sigma_int = 0.05*args.sigma_int
    show = args.show

    flux_scaling_power = args.Iflux
    #flux_scaling = 1.58489**flux_scaling_power
    flux_scaling = 1.2**flux_scaling_power
    sini = 0.05 + 0.1*args.sini
    assert (0<sini<1)
    hlr = 0.5 + 0.5*args.hlr 
    fiber_conf = args.fiberconf

    ########################### Initialization #################################
    redshift = 0.3
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
        [0.01, 0.01, 0.01, 0.01, 2, 5, 0.01, 0.01]
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
            'g1': priors.UniformPrior(-0.5, 0.5),
            'g2': priors.UniformPrior(-0.5, 0.5),
            'theta_int': priors.UniformPrior(-np.pi, np.pi),
            'sini': priors.UniformPrior(0, 1.),
            'v0': priors.GaussPrior(0, 10),
            #'vcirc': priors.UniformPrior(10, 800),
            'vcirc': priors.GaussPrior(300, 80, clip_sigmas=3),
            'rscale': priors.UniformPrior(0.1, 5),
            'hlr': priors.UniformPrior(0.1, 5),
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
            'lblue': 300,
            'lred': 1200,
            'resolution': 500000,
            'scale': 0.11, # arcsec
            'lambda_range': [[482.3, 487.11], [629.7, 634.51], [642.4, 647.21], [648.7, 653.51], [851, 855.81]], 
            'lambda_res': 0.08, # nm
            'super_sampling': 4,
            'lambda_unit': 'nm',
        },
        ### SED model
        # typical values: cont 4e-16, emline 1e-16-1e-15 erg/s/cm2/nm
        'sed':{
            'z': redshift,
            'continuum_type': 'temp',
            'restframe_temp': '../data/Simulation/GSB2.spec',
            'temp_wave_type': 'Ang',
            'temp_flux_type': 'flambda',
            'cont_norm_method': 'flux',
            'obs_cont_norm_wave': 850,
            'obs_cont_norm_flam': 3.0e-17*flux_scaling,
            'em_Ha_flux': 1.2e-16*flux_scaling,
            #'em_Ha_sigma': 0.26,
            'em_Ha_sigma': sigma_int*(1+redshift),
            'em_O2_flux': 8.8e-17*flux_scaling*1,
            'em_O2_sigma': (sigma_int*(1+redshift), sigma_int*(1+redshift)),
            'em_O2_share': (0.45, 0.55),
            'em_O3_1_flux': 2.4e-17*flux_scaling*1,
            'em_O3_1_sigma': sigma_int*(1+redshift),
            'em_O3_2_flux': 2.8e-17*flux_scaling*1,
            'em_O3_2_sigma': sigma_int*(1+redshift),
            'em_Hb_flux': 1.2e-17*flux_scaling,
            'em_Hb_sigma': sigma_int*(1+redshift),
            # 'template': '../data/Simulation/GSB2.spec',
            # 'wave_type': 'Ang',
            # 'flux_type': 'flambda',
            # 'z': 0.3,
            # 'wave_range': [500., 3000.], # nm
            # # obs-frame continuum normalization (nm, erg/s/cm2/nm)
            # 'obs_cont_norm': [850, 2.6e-15*flux_scaling],
            # # a dict of line names and obs-frame flux values (erg/s/cm2)
            # 'lines':{
            #     'Ha': 1.25e-15*flux_scaling,
            #     'O2': [1.0e-14*flux_scaling, 1.2e-14*flux_scaling],
            #     'O3_1': 1.0e-14*flux_scaling,
            #     'O3_2': 1.2e-14*flux_scaling,
            # },
            # # intrinsic linewidth in nm
            # 'line_sigma_int':{
            #     'Ha': 0.05,
            #     'O2': [0.2, 0.2],
            #     'O3_1': 0.2,
            #     'O3_2': 0.2,
            # },
        },
        ### observation configurations
        'obs_conf': default_obs_conf,
    }
    pars = Pars(sampled_pars, meta_pars)

    ### Outputs
    cube_dir = os.path.join(utils.TEST_DIR, 'test_data')
    #cube_dir = os.path.join("/xdisk/timeifler/jiachuanxu/kl_fiber")
    outdir = os.path.join(cube_dir, run_name)
    outfile_sampler = os.path.join(outdir, 'sampler/%s_sini%.2f_hlr%.2f_intsig%.3f_fiberconf%d.pkl'%(flux_scaling_power, sini, hlr, args.sigma_int, fiber_conf))
    outfile_dv = os.path.join(outdir, 'dv/%s_sini%.2f_hlr%.2f_intsig%.3f_fiberconf%d.pkl'%(flux_scaling_power, sini, hlr, args.sigma_int, fiber_conf))

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

    ######################### Run MCMC sampler #################################
    if (not args.mpi) and args.ncores==1:
        pool = None 
    else:
        pool = schwimmbad.choose_pool(mpi=args.mpi, processes=args.ncores)
    if isinstance(pool, MPIPool):
        if not pool.is_master():
            pool.wait(callback=_callback_)
            sys.exit(0)
        else:
            if not os.path.exists(outdir):
                utils.make_dir(outdir)
            if not os.path.exists(os.path.join(outdir, "dv")):
                utils.make_dir(os.path.join(outdir, "dv"))
            if not os.path.exists(os.path.join(outdir, "sampler")):
                utils.make_dir(os.path.join(outdir, "sampler"))
    print('>>>>>>>>>> [%d/%d] Starting EMCEE run <<<<<<<<<<'%(rank, size))
    MCMCsampler = emcee.EnsembleSampler(nwalkers, ndims, log_posterior,
        args=[None, pars], pool=pool)
    p0 = emcee.utils.sample_ball(sampled_pars_value, sampled_pars_std, 
        size=nwalkers)
    MCMCsampler.run_mcmc(p0, nsteps, progress=True)

    ########################## Save MCMC outputs ###############################
    print(f'Pickling sampler to {outfile_sampler}')
    with open(outfile_sampler, 'wb') as f:
        pickle.dump(MCMCsampler, f)
    
    print(f'Saving data vector to {outfile_dv}')
    dv = get_GlobalDataVector(0)
    dv.to_fits(outfile_dv, overwrite=True)

    ######################### Analysis the chains ##############################
    

    return 0

def _callback_():
    print("Worker exit!!!!!")

if __name__ == '__main__':
    
    rc = main(args, None)

    if rc == 0:
        print('All tests ran succesfully')
    else:
        print(f'Tests failed with return code of {rc}')
