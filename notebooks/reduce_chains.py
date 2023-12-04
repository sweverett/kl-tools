HOME_DIR = "/home/u17/jiachuanxu/kl-tools"
import numpy as np
import galsim as gs
from galsim.angle import Angle, radians, degrees
import sys
import os
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
sys.path.append(HOME_DIR+"/kl_tools")
sys.path.append(HOME_DIR+"/kl_tools/grism_modules")

import kltools_grism_module_2 as m
from parameters import Pars, SampledPars, MetaPars, MCMCPars
import priors
import utils
from likelihood import LogPosterior, FiberLikelihood, get_Cube
from datavector import DataVector, FiberDataVector
from emission import LINE_LAMBDAS
def return_lse_func(xs, ys, ivars):
    xc = np.sum(xs*ivars)/np.sum(ivars)
    yc = np.sum(ys*ivars)/np.sum(ivars)
    alpha = np.sum(ys*(xs-xc)*ivars)/np.sum((xs-xc)*(xs-xc)*ivars)
    return lambda x: alpha*(x-xc) + yc

def fit_cont(flux, wave, noise, emline_name, redshift):
    emline_obs_wave = LINE_LAMBDAS[emline_name].to('Angstrom').value * (1+redshift)
    #print(f'{emline_name} at z={redshift} is {emline_obs_wave} A')
    cont_wave = ((emline_obs_wave-20 < wave) & (wave < emline_obs_wave-10)) | \
                    ((emline_obs_wave+10 < wave) & (wave < emline_obs_wave+20))
    xs, ys, ivars = wave[cont_wave], flux[cont_wave], 1/noise[cont_wave]**2
    return return_lse_func(xs, ys, ivars)

def get_emline_snr(flux, wave, noise, emline_name, redshift, subtract_cont=False):
    emline_obs_wave = LINE_LAMBDAS[emline_name].to('Angstrom').value * (1+redshift)
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
default_meta = {
    ### shear and alignment
    'g1': 'sampled',
    'g1': 'sampled',
    'theta_int': 'sampled',
    'sini': 'sampled',
    ### oriors
    'priors': {
        'g1': priors.GaussPrior(0., 0.1, clip_sigmas=2),
        'g2': priors.GaussPrior(0., 0.1, clip_sigmas=2),
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
        'obs_cont_norm': [850, 2.6e-15],
        # a dict of line names and obs-frame flux values (erg/s/cm2)
        'lines':{
            'Ha': 1.25e-15,
            'O2': [1.0e-14, 1.2e-14],
            'O3_1': 1.0e-14,
            'O3_2': 1.2e-14,
        },
        # intrinsic linewidth in nm
        'line_sigma_int':{
            'Ha': 0.05,
            'O2': [0.2, 0.2],
            'O3_1': 0.2,
            'O3_2': 0.2,
        },
    },
}

### Parameter Truth
### ===============
### state which parameters are being sampled and their fiducial values
### remember to set those parameters to "sampled" in meta_pars
sampled_pars = ['g1', 'g2', 'theta_int', 'sini', 'v0', 'vcirc', 'rscale', 'hlr']
sampled_pars_value = [0.0, 0.0, 0, 0.5, 0.0, 300.0, 1.5, 1.5]
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
DATA_DIR = "/xdisk/timeifler/jiachuanxu/kl_fiber/bgs_like_array/"
sampler_fn = DATA_DIR+"sampler_%d_sini%.2f_hlr%.1f_fiberconf%d.pkl"
### Data vector used in the code
datafile = DATA_DIR+"dv_%d_sini%.2f_hlr%.1f_fiberconf%d.pkl"

Nparams = len(sampled_pars)

info_table = np.zeros(960, 
                      dtype=[('flux_bin', '>f4'), 
                             ('sini', '>f4'), 
                             ('hlr', '>f4'), 
                             ('fiberconf', '>i4'), 
                             ('SNR_best', '>f4'),
                             ('SNrms', '>f4')]
                     )
print("Wof")
ct = 0
for a in range(2):
    if a==0:
        Nspec=5 # both major and minor
    elif (a==1) or (a==2) or (a==3):
        Nspec=3 # major; minor; semi-major/minor
    elif a==4:
        Nspec=4 # 3+1
    else:
        print(f'Fiber conf = {a} is not supported!')
        exit(-1)
    for b in range(12):
        flux_scaling = 1.58489**b
        for c in range(4):
            hlr = 0.5 + c*0.5
            for d in range(10):
                sini = 0.05 + d*0.1
                ct += 1
                print(">>>>>>>>>> Sim %d"%ct)
                #if(a==0):
                #    continue
                info_table['flux_bin'][ct-1] = b
                info_table['sini'][ct-1] = sini
                info_table['hlr'][ct-1] = hlr
                info_table['fiberconf'][ct-1] = a
                
                with open(sampler_fn%(b, sini, hlr, a), 'rb') as fp:
                    sampler = pickle.load(fp)
                dv = FiberDataVector(file=datafile%(b, sini, hlr, a))
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

                ### plot trace
                fig, axes = plt.subplots(Nparams+1,1,figsize=(8,12), sharex=True)
                for i in range(Nparams):
                    for j in range(2*Nparams):
                        axes[i].plot(chains[:,j,i])
                    axes[i].set(ylabel=r'$%s$'%sampled_pars_label[i])
                    axes[i].axhline(sampled_pars_value_dict[sampled_pars[i]], ls='--', color='k')
                for j in range(2*Nparams):
                    axes[Nparams].semilogy(-blobs[:,j,1])
                axes[Nparams].set(ylim=[0.5,1e8])
                plt.savefig("../figs/trace/sim%d_trace.png"%ct)
                plt.close(fig)
                ### triangle plot
                g = plots.get_subplot_plotter()
                g.settings.title_limit_fontsize = 14
                g.triangle_plot([samples,], filled=True, 
                                markers=sampled_pars_value_dict,
                                marker_args = {'lw':2, 'ls':'--', 'color':'k'},
                                #param_limits={k:v for k,v in zip(param_names, param_limit)}
                                title_limit = 1,
                               )
                g.export("../figs/posterior/sim%d_posterior.png"%ct)
                ### shape noise 
                marge_stat = samples.getMargeStats()
                g1, eg1 = marge_stat.parWithName('g1').mean, marge_stat.parWithName('g1').err
                g2, eg2 = marge_stat.parWithName('g2').mean, marge_stat.parWithName('g2').err
                sigma_e_rms = np.sqrt(eg1**2+eg2**2)
                print(f'r.m.s. shape noise = {sigma_e_rms}')
                #SNrms_list.append(sigma_e_rms)
                info_table['SNrms'][ct-1] = sigma_e_rms
                ### best-fitting v.s. data
                sampled_pars_bestfit = chains_flat[np.argmax(np.sum(blobs_flat, axis=1)), :]
                meta_pars = deepcopy(default_meta)
                meta_pars['sed']['obs_cont_norm'][1] *= flux_scaling
                meta_pars['sed']['lines']['Ha'] *= flux_scaling
                meta_pars['sed']['lines']['O2'][0] *= flux_scaling
                meta_pars['sed']['lines']['O2'][1] *= flux_scaling
                meta_pars['sed']['lines']['O3_1'] *= flux_scaling
                meta_pars['sed']['lines']['O3_2'] *= flux_scaling

                pars = Pars(sampled_pars, meta_pars)
                log_posterior = LogPosterior(pars, dv, likelihood='fiber')
                loglike = log_posterior.log_likelihood
                wave = get_Cube(0).lambdas.mean(axis=1)*10 # Angstrom
                images_bestfit = loglike.get_images(sampled_pars_bestfit)

                ### fiber spectra
                Ha_center = LINE_LAMBDAS['Ha'].to('Angstrom').value * (1+pars.meta['sed']['z'])
                fig, axes = plt.subplots(1,Nspec, figsize=(2*Nspec,2), sharey=True)
                for i in range(Nspec):
                    ax = axes[i]
                    snr = get_emline_snr(dv.get_data(i), wave, dv.get_noise(i), 'Ha', 
                               pars.meta['sed']['z'], subtract_cont=True)
                    if i==Nspec-1:
                        #SNR_list.append(snr)
                        info_table['SNR_best'][ct-1]=snr
                    fiberpos = (dv.get_config(i)['FIBERDX'],dv.get_config(i)['FIBERDY'])
                    ax.plot(wave, dv.get_data(i)+dv.get_noise(i), color='grey', drawstyle='steps')
                    ax.plot(wave, dv.get_data(i), color='k', label='data')
                    ax.plot(wave, images_bestfit[i], label='bestfit', ls=':', color='r')
                    ax.text(0.05, 0.02, "S/N=%.2f"%snr, transform=ax.transAxes)
                    ax.set(xlim=[Ha_center-20, Ha_center+20], xlabel='Wavelength [A]')
                    ax.text(0.05, 0.9, '(%.1f", %.1f")'%(fiberpos[0], fiberpos[1]), transform=ax.transAxes)
                axes[0].legend(frameon=False)
                axes[0].set(ylabel='ADU')
                plt.savefig("../figs/spectra/sim%d_spectra.png"%ct)
                plt.close(fig)
                ### broad-band image
                fig, axes = plt.subplots(1,3,figsize=(9,3), sharey=True)
                noisy_data = dv.get_data(Nspec)+dv.get_noise(Nspec)
                dchi2 = (((dv.get_data(Nspec)-images_bestfit[Nspec])/np.std(dv.get_noise(Nspec)))**2).sum()

                Ny, Nx = noisy_data.shape
                extent = np.array([-Nx/2, Nx/2, -Ny/2, Ny/2])*dv.get_config(Nspec)['PIXSCALE']

                cb = axes[0].imshow(noisy_data, origin='lower', extent=extent)
                vmin, vmax = cb.get_clim()
                axes[1].imshow(images_bestfit[Nspec], origin='lower', vmin=vmin, vmax=vmax, extent=extent)
                axes[2].imshow(noisy_data-images_bestfit[Nspec], origin='lower', vmin=vmin, vmax=vmax, extent=extent)


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

                for i in range(Nspec):
                    conf = dv.get_config(i)
                    dx, dy, rad = conf['FIBERDX'],conf['FIBERDY'],conf['FIBERRAD']
                    circ = Circle((dx, dy), rad, fill=False, ls='-.', color='red')
                    axes[0].add_patch(circ)
                    axes[0].text(dx, dy, "+", ha='center', va='center', color='red')

                plt.savefig("../figs/image/sim%d_image.png"%ct)
                plt.close(fig)
                del sampler, dv, images_bestfit, chains, chains_flat, blobs, blobs_flat
                del marge_stat, pars, log_posterior, loglike, wave, noisy_data
np.savetxt("kl_chains_summary_table.dat", info_table, header=' '.join(info_table.dtype.names))
print("Done")

