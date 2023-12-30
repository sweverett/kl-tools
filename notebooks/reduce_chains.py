HOME_DIR = "/home/u17/jiachuanxu/kl-tools"
import numpy as np
import galsim as gs
from galsim.angle import Angle, radians, degrees
import sys
import os
from pathlib import Path
from copy import deepcopy
import astropy.io.fits as fits
from astropy.units import Unit
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from matplotlib.patches import Circle
import cProfile
import pickle
import getdist
from getdist import plots, MCSamples
from argparse import ArgumentParser
sys.path.append(HOME_DIR+"/kl_tools")
sys.path.append(HOME_DIR+"/kl_tools/grism_modules")

import kltools_grism_module_2 as m
from parameters import Pars, SampledPars, MetaPars, MCMCPars
import priors
import utils
import likelihood
from likelihood import LogPosterior, FiberLikelihood, get_Cube
from datavector import DataVector, FiberDataVector
import emission
from emission import LINE_LAMBDAS

parser = ArgumentParser()
parser.add_argument('-run_name', type=str, default='',
                    help='Name of mcmc run')
parser.add_argument('-Iflux', type=int, default=0,help='Flux bin index')
parser.add_argument('-sini', type=int, default=0, help='sin(i) bin index')
parser.add_argument('-hlr', type=int, default=0, help='image hlr bin index')
parser.add_argument('-sigma_int', type=int, default=1, help='Intrinsic eml sig')
parser.add_argument('-fiberconf', type=int, default=0, help='fiber conf index')
parser.add_argument('-EXPTIME', type=int, default=600, help='Exposure time in s')
args = parser.parse_args()

flux_bin = args.Iflux
flux_scaling = 1.2**flux_bin
sini = args.sini*0.1 + 0.05
hlr = 0.5+0.5*args.hlr 
sigma_int = 0.05*args.sigma_int
fiberconf = args.fiberconf

fiber_blur = 3.4 # pixels
atm_psf_fwhm = 1.0 # arcsec
fiber_rad = 0.75 # arcsec
fiber_offset_x = 1.5 # arcsec
fiber_offset_y = 1.5 # arcsec
exptime_nominal = args.EXPTIME # seconds
ADD_NOISE = False

emlines = ['O2', 'O3_1', 'O3_2', 'Ha']
blockids = [0, 2, 3, 4]
if args.fiberconf==0:
    offsets = [(fiber_offset_x, 0), (-fiber_offset_x, 0), 
               (0, fiber_offset_y), (0, -fiber_offset_y), (0,0)]
    Nspec = 5
elif args.fiberconf==1:
    offsets = [(fiber_offset_x, 0), (-fiber_offset_x, 0), (0,0)]
    Nspec = 3
elif args.fiberconf==2:
    offsets = [(0, fiber_offset_y), (0, -fiber_offset_y), (0,0)]
    Nspec = 3
elif args.fiberconf==3:
    offsets = [(fiber_offset_x, 0), (0, fiber_offset_y), (0,0)]
    Nspec = 3
else:
    print(f'Fiber configuration case {args.fiberconf} is not implemented yet!')
    exit(-1)



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

### (Default) Meta parameters used in the chains setting
redshift = 0.3
default_meta = {
    ### shear and alignment
    'g1': 'sampled',
    'g1': 'sampled',
    'theta_int': 'sampled',
    'sini': 'sampled',
    ### oriors
    'priors': {
        'g1': priors.UniformPrior(-0.5, 0.5),
        'g2': priors.UniformPrior(-0.5, 0.5),
        'theta_int': priors.UniformPrior(-np.pi, np.pi),
        'sini': priors.UniformPrior(0, 1.),
        'v0': priors.GaussPrior(0, 10),
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
#        'template': '../data/Simulation/GSB2.spec',
#        'wave_type': 'Ang',
#        'flux_type': 'flambda',
#        'z': 0.3,
#        'wave_range': [500., 3000.], # nm
#        # obs-frame continuum normalization (nm, erg/s/cm2/nm)
#        'obs_cont_norm': [850, 2.6e-15*flux_scaling],
#        # a dict of line names and obs-frame flux values (erg/s/cm2)
#        'lines':{
#            'Ha': 1.25e-15*flux_scaling,
#            'O2': [1.0e-14*flux_scaling, 1.2e-14*flux_scaling],
#            'O3_1': 1.0e-14*flux_scaling,
#            'O3_2': 1.2e-14*flux_scaling,
#        },
#        # intrinsic linewidth in nm
#        'line_sigma_int':{
#            'Ha': 0.05,
#            'O2': [0.2, 0.2],
#            'O3_1': 0.2,
#            'O3_2': 0.2,
#        },
    },
}

### Parameter Truth
### ===============
### state which parameters are being sampled and their fiducial values
### remember to set those parameters to "sampled" in meta_pars
sampled_pars = ['g1', 'g2', 'theta_int', 'sini', 'v0', 'vcirc', 'rscale', 'hlr']
sampled_pars_value = [0.0, 0.0, 0, sini, 0.0, 300.0, hlr, hlr]
sampled_pars_value_dict = {key:val for key,val in zip(sampled_pars, sampled_pars_value)}
# for the convenience of plotting
sampled_pars_label = ['g_1', 'g_2', r'{\theta}_{\mathrm{int}}', r'\mathrm{sin}(i)', 'v_0', 
                      'v_\mathrm{circ}', 'r_\mathrm{scale}', 
                      r'\mathrm{hlr}'
                     ]
param_limit = [
    [-0.2, 0.2],
    [-0.2, 0.2],
    [-1.0, 1.0],
    [0.0, 1.0],
    [-100, 100],
    [0.0, 800],
    [0, 5],
    [0, 5],
]

### Pickled runner and sampler
DATA_DIR = os.path.join("/xdisk/timeifler/jiachuanxu/kl_fiber", args.run_name)
#DATA_DIR = os.path.join("../tests/test_data", args.run_name)
FIG_DIR = os.path.join(DATA_DIR, "figs")
SUM_DIR = os.path.join(DATA_DIR, "summary_stats")
Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
Path(FIG_DIR).mkdir(parents=True, exist_ok=True)
Path(SUM_DIR).mkdir(parents=True, exist_ok=True)
Path(os.path.join(FIG_DIR, "image")).mkdir(parents=True, exist_ok=True)
Path(os.path.join(FIG_DIR, "spectra")).mkdir(parents=True, exist_ok=True)
Path(os.path.join(FIG_DIR, "trace")).mkdir(parents=True, exist_ok=True)
Path(os.path.join(FIG_DIR, "posterior")).mkdir(parents=True, exist_ok=True)

sampler_fn = os.path.join(DATA_DIR, 
    "sampler/%d_sini%.2f_hlr%.2f_intsig%.3f_fiberconf%d.pkl"%(flux_bin, sini, hlr, args.sigma_int, fiberconf))
### Data vector used in the code
datafile = os.path.join(DATA_DIR, 
    "dv/%d_sini%.2f_hlr%.2f_intsig%.3f_fiberconf%d.pkl"%(flux_bin, sini, hlr, args.sigma_int, fiberconf))
postfix = "sim_%d_sini%.2f_hlr%.2f_intsig%.3f_fiberconf%d"%(flux_bin, sini, hlr, args.sigma_int, fiberconf)

Nparams = len(sampled_pars)

### Read chains
with open(sampler_fn, 'rb') as fp:
    sampler = pickle.load(fp)
dv = FiberDataVector(file=datafile)
### get sampled parameters
chains = sampler.get_chain(flat=False)
chains_flat = sampler.get_chain(flat=True)
# get blobs (priors, like)
blobs = sampler.get_blobs(flat=False)
blobs_flat = sampler.get_blobs(flat=True)

### build getdist.MCSamples object from the chains
Nsteps_start = int(chains.shape[0]*0.5)
good_walkers = np.where(blobs[-1,:,1]>-100)[0]
print(f'Failed walkers {blobs.shape[1]-len(good_walkers)}/{blobs.shape[1]}')
samples = MCSamples(samples=[chains[Nsteps_start:,gw,:] for gw in good_walkers], 
                loglikes=[-1*np.sum(blobs[Nsteps_start:,gw,:], axis=1) for gw in good_walkers],
                names = sampled_pars, labels = sampled_pars_label)

### 1. plot trace
### =============
fig, axes = plt.subplots(Nparams+1,1,figsize=(8,12), sharex=True)
for i in range(Nparams):
    for j in range(2*Nparams):
        axes[i].plot(chains[:,j,i])
    axes[i].set(ylabel=r'$%s$'%sampled_pars_label[i])
    axes[i].axhline(sampled_pars_value_dict[sampled_pars[i]], ls='--', color='k')
for j in range(2*Nparams):
    axes[Nparams].semilogy(-blobs[:,j,1])
axes[Nparams].set(ylim=[0.5,1e8])
plt.savefig(os.path.join(FIG_DIR,"trace/"+postfix+".png"))
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
g.export(os.path.join(FIG_DIR,"posterior/"+postfix+".png"))

### 3. shape noise
### ============== 
marge_stat = samples.getMargeStats()
g1, eg1 = marge_stat.parWithName('g1').mean, marge_stat.parWithName('g1').err
g2, eg2 = marge_stat.parWithName('g2').mean, marge_stat.parWithName('g2').err
sigma_e_rms = np.sqrt(eg1**2+eg2**2)
print(f'r.m.s. shape noise = {sigma_e_rms}')

### 4. best-fitting v.s. data
### =========================
sampled_pars_bestfit = chains_flat[np.argmax(np.sum(blobs_flat, axis=1)), :]
sampled_pars_bestfit_dict = {k:v for k,v in zip(sampled_pars, sampled_pars_bestfit)}
meta_pars = deepcopy(default_meta)

obsSED = emission.ObsFrameSED(meta_pars['sed'])
rmag = obsSED.calculateMagnitude("../data/Bandpass/CTIO/DECam.r.dat")
pars = Pars(sampled_pars, meta_pars)
log_posterior = LogPosterior(pars, dv, likelihood='fiber')
loglike = log_posterior.log_likelihood

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
        snr = get_emline_snr(dv.get_data(_obs_id_)+0*dv.get_noise(_obs_id_), 
                             wave*10, dv.get_noise(_obs_id_), emline, 
                             pars.meta['sed']['z'], subtract_cont=True)
        ax.plot(wave*10, dv.get_data(_obs_id_)+dv.get_noise(_obs_id_), color="grey", drawstyle="steps")
        ax.text(0.05,0.05, "SNR=%.3f"%snr, transform=ax.transAxes, color='red', weight='bold')
        ax.text(0.05,0.9, "(%.1f, %.1f)"%(dx, dy), transform=ax.transAxes, color='red', weight='bold')
        ax.plot(wave*10, images_bestfit[_obs_id_], ls='-', color="k")
        # ax.axvline(emline_cen-20, c='grey', ls='-.')
        # ax.axvline(emline_cen+20, c='grey', ls='-.')
        # ax.axvline(emline_cen-10, c='grey', ls=':')
        # ax.axvline(emline_cen+10, c='grey', ls=':')
        if j==0:
            ax.set(ylabel='Flux [ADU]')
        if (np.abs(dx)<1e-3) & (np.abs(dy)<1e-3):
            SNR_best.append(snr)
        _obs_id_+=1
    axes[len(offsets)-1, j].set(xlabel="Wavelength [A]")
    axes[0, j].set(title=f'{emline}')
plt.xlabel('wavelength [A]')
plt.ylabel('ADU')
# Ha_center = LINE_LAMBDAS['Ha'].to('Angstrom').value * (1+pars.meta['sed']['z'])
# fig, axes = plt.subplots(Nspec,  figsize=(2*Nspec,2), sharey=True)
# for i in range(Nspec):
#     ax = axes[i]
#     #print(dv.get_data(i).shape)
#     snr = get_emline_snr(dv.get_data(i), wave, dv.get_noise(i), 'Ha', 
#                pars.meta['sed']['z'], subtract_cont=True)
#     if i==Nspec-1:
#         SNR_best=snr
#     fiberpos = (dv.get_config(i)['FIBERDX'],dv.get_config(i)['FIBERDY'])
#     ax.plot(wave, dv.get_data(i)+dv.get_noise(i), color='grey', drawstyle='steps')
#     ax.plot(wave, dv.get_data(i), color='k', label='data')
#     ax.plot(wave, images_bestfit[i], label='bestfit', ls=':', color='r')
#     ax.text(0.05, 0.02, "S/N=%.2f"%snr, transform=ax.transAxes)
#     ax.set(xlim=[Ha_center-20, Ha_center+20], xlabel='Wavelength [A]')
#     ax.text(0.05, 0.9, '(%.1f", %.1f")'%(fiberpos[0], fiberpos[1]), transform=ax.transAxes)
# axes[0].legend(frameon=False)
# axes[0].set(ylabel='ADU')
plt.savefig(os.path.join(FIG_DIR, "spectra/"+postfix+".png"))
plt.close(fig)

### 6. broad-band image
### ===================
fig, axes = plt.subplots(1,3,figsize=(9,3), sharey=True)
noisy_data = dv.get_data(_obs_id_)+dv.get_noise(_obs_id_)
dchi2 = (((dv.get_data(_obs_id_)-images_bestfit[_obs_id_])/np.std(dv.get_noise(_obs_id_)))**2).sum()

Ny, Nx = noisy_data.shape
extent = np.array([-Nx/2, Nx/2, -Ny/2, Ny/2])*dv.get_config(_obs_id_)['PIXSCALE']

cb = axes[0].imshow(noisy_data, origin='lower', extent=extent)
vmin, vmax = cb.get_clim()
axes[1].imshow(images_bestfit[_obs_id_], origin='lower', vmin=vmin, vmax=vmax, extent=extent)
axes[2].imshow(noisy_data-images_bestfit[_obs_id_], origin='lower', vmin=vmin, vmax=vmax, extent=extent)


plt.colorbar(cb, ax=axes.ravel().tolist(), location='right', fraction=0.0135,
             label='ADU', pad=0.005)
axes[0].text(0.05, 0.9, 'Data (noise-free)', color='white', transform=axes[0].transAxes)
axes[1].text(0.05, 0.9, 'Bestfit', color='white', transform=axes[1].transAxes)
axes[2].text(0.05, 0.9, 'Redisuals', color='white', transform=axes[2].transAxes)
axes[2].text(0.75, 0.9, r'$\Delta\chi^2=$%.1e'%(dchi2), color='white', ha='center',
             transform=axes[2].transAxes)

axes[0].set(xlabel="X [arcsec]", ylabel="Y [arcsec]")
axes[1].set(xlabel="X [arcsec]")
axes[2].set(xlabel="X [arcsec]")

for (dx, dy) in offsets:
    rad = fiber_rad
    #conf = dv.get_config(i)
    #dx, dy, rad = conf['FIBERDX'],conf['FIBERDY'],conf['FIBERRAD']
    circ = Circle((dx, dy), rad, fill=False, ls='-.', color='red')
    axes[0].add_patch(circ)
    axes[0].text(dx, dy, "+", ha='center', va='center', color='red')

plt.savefig(os.path.join(FIG_DIR,"image/"+postfix+".png"))
plt.close(fig)

### 7. save summary stats
### =====================
with open(os.path.join(SUM_DIR,postfix+".dat"), "w") as fp:
    res1 = "%d %f %.2f %.2f %d %le %le"%(flux_bin, rmag, sini, hlr, fiberconf, sigma_e_rms, np.max(SNR_best))
    pars_bias = [sampled_pars_bestfit_dict[key]-sampled_pars_value_dict[key] for key in sampled_pars]
    pars_errs = [marge_stat.parWithName(key).err for key in sampled_pars]
    res2 = ' '.join("%le"%bias for bias in pars_bias)
    res3 = ' '.join("%le"%err for err in pars_errs)
    fp.write(' '.join([res1, res2, res3])+'\n')
if (args.Iflux==0) and (args.sini==0) and (args.hlr==0) and (args.fiberconf==0):
    with open(os.path.join(SUM_DIR,"colnames.dat"), "w") as fp:
        hdr1 = "# flux_bin rmag sini hlr fiberconf sn_rms snr_best"
        hdr2 = ' '.join("%s_bias"%key for key in sampled_pars)
        hdr3 = ' '.join("%s_std"%key for key in sampled_pars)
        fp.write(' '.join([hdr1, hdr2, hdr3])+'\n')

print("Done")

