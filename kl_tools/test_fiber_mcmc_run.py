# This is to ensure that numpy doesn't have
# OpenMP optimizations clobber our own multiprocessing
_USER_RUNNER_CLASS_ = False
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
comm = MPI.COMM_WORLD
from schwimmbad import MPIPool
from argparse import ArgumentParser
from astropy.units import Unit
import galsim as gs
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from matplotlib.patches import Circle
import getdist
from getdist import plots, MCSamples
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
from emission import LINE_LAMBDAS

import ipdb

def return_lse_func(xs, ys, ivars):
    xc = np.sum(xs*ivars)/np.sum(ivars)
    yc = np.sum(ys*ivars)/np.sum(ivars)
    alpha = np.sum(ys*(xs-xc)*ivars)/np.sum((xs-xc)*(xs-xc)*ivars)
    return lambda x: alpha*(x-xc) + yc

def fit_cont(flux, wave, noise, emline_name, redshift):
    emline_obs_wave = np.mean(LINE_LAMBDAS[emline_name].to('Angstrom').value) * (1+redshift)
    #print(f'{emline_name} at z={redshift} is {emline_obs_wave} A')
    cont_wave = ((emline_obs_wave-20 < wave) & (wave < emline_obs_wave-10)) | \
                    ((emline_obs_wave+10 < wave) & (wave < emline_obs_wave+20))
    xs, ys, ivars = wave[cont_wave], flux[cont_wave], 1/noise[cont_wave]**2
    return return_lse_func(xs, ys, ivars)

def get_emline_snr(flux, wave, noise, emline_name, redshift, subtract_cont=False):
    emline_obs_wave = np.mean(LINE_LAMBDAS[emline_name].to('Angstrom').value) * (1+redshift)
    #print(f'{emline_name} at z={redshift} is {emline_obs_wave} A')
    if not subtract_cont:
        wave_range = (emline_obs_wave-10 < wave) & (wave < emline_obs_wave+10)
        SNR = flux[wave_range].sum() / np.sqrt((noise[wave_range]**2).sum())
    else:
        cont_fit = fit_cont(flux, wave, noise, emline_name, redshift)
        wave_range = (emline_obs_wave-10 < wave) & (wave < emline_obs_wave+10)
        SNR = np.sum(flux[wave_range]-cont_fit(wave[wave_range])) / \
            np.sqrt((noise[wave_range]**2).sum()) #aproximate
    return SNR

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
parser.add_argument('-EXP_OFFSET', type=int, default=600, 
    help='Exposure time of offset fibers, in second')
parser.add_argument('-EXP_PHOTO', type=int, default=-1, 
    help='Exposure time of photometry image, in second')
parser.add_argument('-PA', type=int, default=0, help='Position angle')
parser.add_argument('-NPHOT', type=int, default=1, help='# of photometry image')
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
fiber_offset = 1.5 # arcsec
#fiber_offset_y = 1.5 # arcsec
exptime_offset = args.EXP_OFFSET # seconds
exptime_photo = args.EXP_PHOTO
PA = 10/180*np.pi*args.PA
while PA>np.pi:
    PA -= 2*np.pi
ADD_NOISE = False

##################### Setting up observation configurations ####################
default_fiber_conf = {'INSTNAME': "DESI", 'OBSTYPE': 1,
     'SKYMODEL': "../data/Skyspectra/skysb_grey.dat", 'PSFTYPE': "airy_fwhm",
     'PSFFWHM': atm_psf_fwhm, 'DIAMETER': 332.42, 'EXPTIME': 180,
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
blockids = [0, 1, 2, 3]
channels = ['b', 'r', 'r', 'z']
rdnoise = [3.41, 2.6, 2.6, 2.6]
### Choose fiber configurations
if args.fiberconf==0:
    offsets = [(fiber_offset*np.cos(0),         fiber_offset*np.sin(0)),
               (fiber_offset*np.cos(np.pi/2),   fiber_offset*np.sin(np.pi/2)),
               (fiber_offset*np.cos(np.pi),   fiber_offset*np.sin(np.pi)),
               (fiber_offset*np.cos(3*np.pi/2), fiber_offset*np.sin(3*np.pi/2)),
               (0,0)]
    OFFSETX = 1
elif args.fiberconf==1:
    offsets = [(fiber_offset*np.cos(0),         fiber_offset*np.sin(0)),
               (fiber_offset*np.cos(np.pi),   fiber_offset*np.sin(np.pi)),
               (0,0)]
    OFFSETX = 2
elif args.fiberconf==2:
    offsets = [(fiber_offset*np.cos(np.pi/2),   fiber_offset*np.sin(np.pi/2)),
               (fiber_offset*np.cos(3*np.pi/2), fiber_offset*np.sin(3*np.pi/2)),
               (0,0)]
    OFFSETX = 2
elif args.fiberconf==3:
    offsets = [(fiber_offset*np.cos(0),         fiber_offset*np.sin(0)),
               (fiber_offset*np.cos(np.pi/2),   fiber_offset*np.sin(np.pi/2)),
               (0,0)]
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
            _conf.update({'EXPTIME': exptime_offset*OFFSETX})
        print(f'offset = ({dx}, {dy}); EXPTIME = {_conf["EXPTIME"]}')
        default_obs_conf.append(_conf)
        _index_+=1

### Photometry observations
# Assume 150s exptime for all exposures
# tune the sky level such that the 5-sigma limit are
# 𝑔=24.0, 𝑟=23.4 and 𝑧=22.5
photometry_band = ['r', 'g', 'z']
#sky_levels = [40.4/4*150/15, 19.02/4*150/40, 40.4/4*150/5]
sky_levels = [44.54, 19.02, 168.66]
LS_DR9_exptime = [50, 90, 80]
#photometry_band = ['r', ]
#sky_levels = [40.4/4*150/15,]

for i in range(args.NPHOT):
    _bp = "../data/Bandpass/CTIO/DECam.%s.dat"%photometry_band[i]
    _conf = copy.deepcopy(default_photo_conf)
    _conf.update({"OBSINDEX": _index_, 'BANDPASS': _bp, "SKYLEVEL": sky_levels[i],
        "EXPTIME": exptime_photo if exptime_photo>0 else LS_DR9_exptime[i]})
    default_obs_conf.append(_conf)
    _index_+=1

####################### Main function ##########################################
def main(args, pool):

    # TODO: Try making datacube a global variable, as it may significantly
    # decrease execution time due to pickling only once vs. every model eval
    nsteps = args.nsteps
    sampler = args.sampler
    ncores = args.ncores
    mpi = args.mpi
    sigma_int = 0.05*args.sigma_int
    show = args.show

    flux_scaling_power = args.Iflux
    #flux_scaling = 1.58489**flux_scaling_power
    #flux_scaling = 1.2**flux_scaling_power
    flux_scaling = 10**((22-16)/2.5/9) ** flux_scaling_power
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
        'em_Ha_flux',
        'em_O2_flux',
        'em_O3_1_flux',
        'em_O3_2_flux',
        'obs_cont_norm_flam',
        ]
    print(f'Sampled parameters: {sampled_pars}')
    sampled_pars_label = ['g_1', 'g_2', r'{\theta}_{\mathrm{int}}', 
    r'\mathrm{sin}(i)', 'v_0', 'v_\mathrm{circ}', 'r_\mathrm{scale}', 
    r'\mathrm{hlr}', r'F_\mathrm{H\alpha}', r'F_\mathrm{[OII]}', r'F_\mathrm{[OIII]4960}', r'F_\mathrm{[OIII]5008}', r'F_\mathrm{cont}']
    param_limit = [
        [-0.2, 0.2],
        [-0.2, 0.2],
        [-1.0, 1.0],
        [0.0, 1.0],
        [-100, 100],
        [0.0, 800],
        [0, 5],
        [0, 5],
        [1e-21, 1e-11],[1e-21, 1e-11],[1e-21, 1e-11],[1e-21, 1e-11],[1e-21, 1e-11]
    ]
    sampled_pars_value = [
        0.0, 0.0, 
        PA, 
        sini, 
        0.0, 300.0, hlr, hlr, 
        1.2e-16*flux_scaling, 8.8e-17*flux_scaling, 
        2.4e-17*flux_scaling, 2.8e-17*flux_scaling, 3.0e-17*flux_scaling]
    sampled_pars_std = np.array(
        [0.01, 0.01, 0.01, 0.01, 
         2, 5, 0.01, 
         0.01, 1e-18, 1e-19, 1e-19, 1e-19, 1e-19]
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
            #'v0': 0.0,
            #'vcirc': 300.0,
            #'rscale': 1.5,
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
            'lambda_range': [[482.3, 487.11], #[629.7, 634.51], 
                             [642.4, 647.21], [648.7, 653.51], [851, 855.81]], 
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
            # 'obs_cont_norm_flam': 3.0e-17*flux_scaling,
            'obs_cont_norm_flam': 'sampled',
            # 'em_Ha_flux': 1.2e-16*flux_scaling,
            'em_Ha_flux': 'sampled',
            'em_Ha_sigma': sigma_int*(1+redshift),
            # 'em_O2_flux': 8.8e-17*flux_scaling*1,
            'em_O2_flux': 'sampled',
            'em_O2_sigma': (sigma_int*(1+redshift), sigma_int*(1+redshift)),
            'em_O2_share': (0.45, 0.55),
            # 'em_O3_1_flux': 2.4e-17*flux_scaling*1,
            'em_O3_1_flux': 'sampled',
            'em_O3_1_sigma': sigma_int*(1+redshift),
            # 'em_O3_2_flux': 2.8e-17*flux_scaling*1,
            'em_O3_2_flux': "sampled",
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
    #outdir = os.path.join(utils.TEST_DIR, 'test_data', args.run_name)
    outdir = os.path.join("/xdisk/timeifler/jiachuanxu/kl_fiber", args.run_name)

    fig_dir = os.path.join(outdir, "figs")
    sum_dir = os.path.join(outdir, "summary_stats")
    
    filename_fmt = "%s_sini%.2f_hlr%.2f_intsig%.3f_PA%d_TPHOT%d_fiberconf%d"%\
        (flux_scaling_power, sini, hlr, sigma_int, args.PA, args.EXP_PHOTO, fiber_conf)
    outfile_sampler = os.path.join(outdir, "sampler", filename_fmt+".pkl")
    outfile_dv = os.path.join(outdir,"dv", filename_fmt)

    #-----------------------------------------------------------------
    # Setup sampled posterior

    pars_order = pars.sampled.pars_order
    # log_posterior arguments: theta, data, pars
    log_posterior = LogPosterior(pars, None, likelihood='fiber', sampled_theta_fid=sampled_pars_value)

    #-----------------------------------------------------------------
    # Setup sampler

    ndims = log_posterior.ndims
    nwalkers = 2*ndims
    print(f'Param space dimension = {ndims}; Number of walkers = {nwalkers}')
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
            if not os.path.exists(fig_dir):
                utils.make_dir(fig_dir)
            if not os.path.exists(os.path.join(fig_dir, "trace")):
                utils.make_dir(os.path.join(fig_dir, "trace"))
            if not os.path.exists(os.path.join(fig_dir, "posterior")):
                utils.make_dir(os.path.join(fig_dir, "posterior"))
            if not os.path.exists(os.path.join(fig_dir, "image")):
                utils.make_dir(os.path.join(fig_dir, "image"))
            if not os.path.exists(os.path.join(fig_dir, "spectra")):
                utils.make_dir(os.path.join(fig_dir, "spectra"))
            if not os.path.exists(sum_dir):
                utils.make_dir(sum_dir)
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
    ### Read chains
    chains = MCMCsampler.get_chain(flat=False)
    chains_flat = MCMCsampler.get_chain(flat=True)
    # get blobs (priors, like)
    blobs = MCMCsampler.get_blobs(flat=False)
    blobs_flat = MCMCsampler.get_blobs(flat=True)

    ### build getdist.MCSamples object from the chains
    goodwalkers = np.where(blobs[-1,:,1]>-100)[0]
    print(f'Failed walkers {blobs.shape[1]-len(goodwalkers)}/{blobs.shape[1]}')
    samples = MCSamples(samples=[chains[nsteps//2:,gw,:] for gw in goodwalkers],
        loglikes=[-1*blobs[nsteps//2:,gw,:].sum(axis=1) for gw in goodwalkers],
        names = sampled_pars, labels = sampled_pars_label)

    ### 1. plot trace
    ### =============
    fig, axes = plt.subplots(ndims+1,1,figsize=(8,12), sharex=True)
    for i in range(ndims):
        for j in range(nwalkers):
            axes[i].plot(chains[:,j,i])
        axes[i].set(ylabel=r'$%s$'%sampled_pars_label[i])
        axes[i].axhline(sampled_pars_value_dict[sampled_pars[i]], ls='--', color='k')
    for j in range(nwalkers):
        axes[ndims].semilogy(-blobs[:,j,1])
    #axes[ndims].set(ylim=[0.5,1e8])
    plt.savefig(os.path.join(fig_dir, "trace", filename_fmt+".png"))
    plt.close(fig)

    ### 2. triangle plot
    ### ================
    g = plots.get_subplot_plotter()
    g.settings.title_limit_fontsize = 14
    g.triangle_plot([samples,], filled=True, 
                    markers=sampled_pars_value_dict,
                    marker_args = {'lw':2, 'ls':'--', 'color':'k'},
                    #param_limits={k:v for k,v in zip(param_names, param_limit)}
                    title_limit = 1,
                   )
    g.export(os.path.join(fig_dir, "posterior", filename_fmt+".png"))

    ### 3. shape noise
    ### ============== 
    ms = samples.getMargeStats()
    g1, eg1 = ms.parWithName('g1').mean, ms.parWithName('g1').err
    g2, eg2 = ms.parWithName('g2').mean, ms.parWithName('g2').err
    sigma_e_rms = np.sqrt(eg1**2+eg2**2)
    print(f'r.m.s. shape noise = {sigma_e_rms}')

    ### 4. best-fitting v.s. data
    ### =========================
    sampled_pars_bestfit = chains_flat[np.argmax(np.sum(blobs_flat, axis=1)), :]
    sampled_pars_bestfit_dict = {k:v for k,v in zip(sampled_pars, sampled_pars_bestfit)}
    loglike = log_posterior.log_likelihood
    theory_cube, gal, sed = loglike._setup_model(sampled_pars_bestfit_dict, dv)
    
    rmag = sed.calculateMagnitude("../data/Bandpass/CTIO/DECam.r.dat")
    #wave = likelihood.get_GlobalLambdas()
    #wave = get_Cube(0).lambdas.mean(axis=1)*10 # Angstrom
    images_bestfit = loglike.get_images(sampled_pars_bestfit)

    ### 5. fiber spectra
    ### ================
    fig, axes = plt.subplots(len(offsets),len(emlines), figsize=(2*len(emlines),2*len(offsets)))

    _obs_id_, SNR_best = 0, []
    for j, (emline, bid) in enumerate(zip(emlines, blockids)):
        wave = likelihood.get_GlobalLambdas(bid).mean(axis=1)
        emline_cen = np.mean(LINE_LAMBDAS[emline].to('Angstrom').value) * (1+pars.meta['sed']['z'])
        for i, (dx, dy) in enumerate(offsets):
            ax = axes[i,j]
            snr = get_emline_snr(dv.get_data(_obs_id_), wave*10, 
                                 dv.get_noise(_obs_id_), emline, 
                                 pars.meta['sed']['z'], subtract_cont=True)
            ax.plot(wave*10, dv.get_data(_obs_id_)+dv.get_noise(_obs_id_), color="grey", drawstyle="steps")
            ax.text(0.05,0.05, "SNR=%.3f"%snr, transform=ax.transAxes, color='red', weight='bold')
            ax.text(0.05,0.9, "(%.1f, %.1f)"%(dx, dy), transform=ax.transAxes, color='red', weight='bold')
            ax.plot(wave*10, images_bestfit[_obs_id_], ls='-', color="k")
            if j==0:
                ax.set(ylabel='Flux [ADU]')
            if (np.abs(dx)<1e-3) & (np.abs(dy)<1e-3):
                SNR_best.append(snr)
            _obs_id_+=1
        axes[len(offsets)-1, j].set(xlabel="Wavelength [A]")
        axes[0, j].set(title=f'{emline}')
    plt.xlabel('wavelength [A]')
    plt.ylabel('ADU')
    plt.savefig(os.path.join(fig_dir, "spectra", filename_fmt+".png"))
    plt.close(fig)

    ### 6. broad-band image
    ### ===================
    fig, axes = plt.subplots(args.NPHOT,3,figsize=(9,3*args.NPHOT), sharey=True)
    for i in range(args.NPHOT):
        row_axes = axes[:] if args.NPHOT==1 else axes[i,:]
        ax1, ax2, ax3 = row_axes[0], row_axes[1], row_axes[2]
        noisy_data = dv.get_data(_obs_id_)+dv.get_noise(_obs_id_)
        dchi2 = (((dv.get_data(_obs_id_)-images_bestfit[_obs_id_])/np.std(dv.get_noise(_obs_id_)))**2).sum()

        Ny, Nx = noisy_data.shape
        extent = np.array([-Nx/2, Nx/2, -Ny/2, Ny/2])*dv.get_config(_obs_id_)['PIXSCALE']

        cb = ax1.imshow(noisy_data, origin='lower', extent=extent)
        vmin, vmax = cb.get_clim()
        ax2.imshow(images_bestfit[_obs_id_], origin='lower', 
                            vmin=vmin, vmax=vmax, extent=extent)
        ax3.imshow(noisy_data-images_bestfit[_obs_id_], origin='lower',
                            vmin=vmin, vmax=vmax, extent=extent)
        plt.colorbar(cb, ax=row_axes.ravel().tolist(), location='right', 
            fraction=0.0135, label='ADU', pad=0.005)
        ax1.text(0.05, 0.9, 'Data (noise-free)', color='white', transform=ax1.transAxes)
        ax2.text(0.05, 0.9, 'Bestfit', color='white', transform=ax2.transAxes)
        ax3.text(0.05, 0.9, 'Redisuals', color='white', transform=ax3.transAxes)
        ax3.text(0.75, 0.9, r'$\Delta\chi^2=$%.1e'%(dchi2), color='white',
                        ha='center', transform=ax3.transAxes)


        for (dx, dy) in offsets:
            rad = fiber_rad
            #conf = dv.get_config(i)
            #dx, dy, rad = conf['FIBERDX'],conf['FIBERDY'],conf['FIBERRAD']
            circ = Circle((dx, dy), rad, fill=False, ls='-.', color='red')
            ax1.add_patch(circ)
            ax1.text(dx, dy, "+", ha='center', va='center', color='red')

        ax1.set(ylabel="Y [arcsec]")
        if i==args.NPHOT-1:
            for ax in row_axes:
                ax.set(xlabel="X [arcsec]")
        _obs_id_ += 1

    plt.savefig(os.path.join(fig_dir,"image", filename_fmt+".png"))
    plt.close(fig)

    ### 7. save summary stats
    ### =====================
    with open(os.path.join(sum_dir, filename_fmt+".dat"), "w") as fp:
        res1 = "%d %f %.2f %.2f %le %d %d %le %le"%(args.Iflux, rmag, sini, hlr, PA, args.EXP_PHOTO, fiber_conf, sigma_e_rms, np.max(SNR_best))
        pars_bias = [sampled_pars_bestfit_dict[key]-sampled_pars_value_dict[key] for key in sampled_pars]
        pars_errs = [ms.parWithName(key).err for key in sampled_pars]
        res2 = ' '.join("%le"%bias for bias in pars_bias)
        res3 = ' '.join("%le"%err for err in pars_errs)
        fp.write(' '.join([res1, res2, res3])+'\n')
    #if (args.Iflux==0) and (args.sini==0) and (args.hlr==0) and (args.fiberconf==0):
    colname_fn = os.path.join(sum_dir,"colnames.dat")
    if not os.path.exists(colname_fn):
        with open(colname_fn, "w") as fp:
            hdr1 = "# flux_bin rmag sini hlr PA EXPTIME_PHOTO fiberconf sn_rms snr_best"
            hdr2 = ' '.join("%s_bias"%key for key in sampled_pars)
            hdr3 = ' '.join("%s_std"%key for key in sampled_pars)
            fp.write(' '.join([hdr1, hdr2, hdr3])+'\n')

    return 0

def _callback_():
    print("Worker exit!!!!!")

if __name__ == '__main__':
    
    rc = main(args, None)

    if rc == 0:
        print('All tests ran succesfully')
    else:
        print(f'Tests failed with return code of {rc}')
