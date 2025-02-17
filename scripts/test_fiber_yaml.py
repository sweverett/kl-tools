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

####################### Main function ##########################################
def main(args, pool):
    global rank, size
    ########################### Initialization #################################
    # First, parse the YAML file
    sampled_pars, fidvals, latex_labels, derived, meta_dict, mcmc_dict, ball_mean, ball_std, ball_proposal = utils.parse_yaml(args.yaml)
    if rank==0:
        print(f'Sampled parameters: {sampled_pars}')
    # overwrite some parameters stated in YAML file with arguments
    nsteps = mcmc_dict["nsteps"] if args.nsteps==-1 else args.nsteps 
    ncores = args.ncores
    mpi = args.mpi
    Isini = mcmc_dict["Isini"] if args.Isini==-1 else args.Isini
    Ipa = mcmc_dict["Ipa"] if args.Ipa==-1 else args.Ipa
    Ihlr = mcmc_dict["Ihlr"] if args.Ihlr==-1 else args.Ihlr
    Iflux = mcmc_dict["Iflux"] if args.Iflux==-1 else args.Iflux
    flux_scaling = mcmc_dict["flux_scaling_base"] ** Iflux
    sini = 0.05 + 0.1*Isini;assert (0<sini<1)
    PA = 10/180*np.pi*Ipa
    while PA>np.pi/2.:
        PA -= np.pi
    hlr = 0.5 + 0.5*Ihlr
    eint = np.tan(np.arcsin(sini)/2.)**2
    eint1, eint2 = eint*np.cos(2*PA), eint*np.sin(2*PA)
    # update the parameters
    update_keys = ["eint1", "eint2", "sini", "theta_int", "hlr"]
    update_vals = [eint1, eint2, sini, PA, hlr]
    for k, v in zip(update_keys, update_vals):
        if k in sampled_pars:
            if rank==0:
                print(f'Overwrite fiducial {k} value: {fidvals[k]} -> {v}')
            fidvals[k] = v
            ball_mean[k] = v
        elif (k in meta_dict.keys()) and (meta_dict[k]!="sampled"):
            if rank==0:
                print(f'Overwrite fiducial {k} value: {meta_dict[k]} -> {v}')
            meta_dict[k] = v
        elif (k in meta_dict["intensity"].keys()) and \
                (meta_dict["intensity"][k]!="sampled"):
            if rank==0:
                print(f'Overwrite fiducial {k} value: {meta_dict["intensity"][k]} -> {v}')
            meta_dict["intensity"][k] = v
    # re-scale the flux
    rescale_keys = ["em_Ha_flux", "em_Hb_flux", "em_O2_flux", "em_O3_1_flux", "em_O3_2_flux", "obs_cont_norm_flam"]
    for k in rescale_keys:
        if k in sampled_pars:
            if rank==0:
                print(f'Re-scale {k}: {fidvals[k]} x {flux_scaling}')
            fidvals[k] *= flux_scaling
            ball_mean[k] *= flux_scaling
        elif (k in meta_dict["sed"].keys()) and (meta_dict["sed"][k]!="sampled"):
            if rank==0:
                print(f'Re-scale {k}: {meta_dict["sed"][k]} x {flux_scaling:.4f}')
            meta_dict["sed"][k] *= flux_scaling
    ### Outputs
    outdir = os.path.join(utils.TEST_DIR, 'test_data', args.run_name)
    #outdir = os.path.join("/xdisk/timeifler/jiachuanxu/kl_fiber", args.run_name)

    fig_dir = os.path.join(outdir, "figs")
    sum_dir = os.path.join(outdir, "summary_stats")

    filename_fmt = "%s_sini%.2f_hlr%.2f_fiberconf%d"%\
        (Iflux, sini, hlr, mcmc_dict["fiberconf"])
    outfile_sampler = os.path.join(outdir, "sampler", filename_fmt+".pkl")
    outfile_dv = os.path.join(outdir,"dv", filename_fmt)

    #-----------------------------------------------------------------
    # Setup sampled posterior
    pars = Pars(sampled_pars, meta_dict)
    pars_order = pars.sampled.pars_order
    # log_posterior arguments: theta, data, pars
    log_posterior = LogPosterior(pars, None, likelihood='fiber', sampled_theta_fid=[fidvals[k] for k in sampled_pars])

    #-----------------------------------------------------------------
    # Setup sampler

    ndims = log_posterior.ndims
    nwalkers = 2*ndims
    if rank==0:
        print(f'{ndims} dimension param space; {nwalkers} walkers.')

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
            os.system(f'cp {args.yaml} {os.path.join(outdir, "config.yaml")}')  
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
    p0 = emcee.utils.sample_ball([ball_mean[k] for k in sampled_pars], 
        [ball_std[k]*ball_proposal[k] for k in sampled_pars],size=nwalkers)
    # if we want to monitor the auto-correlation length real-time
    if mcmc_dict.get("monitor_autocorr", False):
        backend_fn = os.path.join(outdir, "chains_backend")
        if rank==0:
            print("Monitoring real-time auto-correlation length")
            if os.path.exists(backend_fn):
                print(f'Removing existing backend file {backend_fn}')
                os.remove(backend_fn)
        backend = emcee.backends.HDFBackend(backend_fn);
        MCMCsampler = emcee.EnsembleSampler(nwalkers, ndims, log_posterior,
            backend=backend, args=[None, pars], pool=pool)
        autocorr, old_tau = [], -np.inf
        for sample in MCMCsampler.sample(p0, iterations=nsteps, progress=True):
            # Only check convergence every 100 steps
            if MCMCsampler.iteration % 100:
                continue
            tau = MCMCsampler.get_autocorr_time(tol=0)
            autocorr.append(np.mean(tau))
            print(f'\nAuto-correlation length at {MCMCsampler.iteration} iterations: {np.mean(tau):.2f} ({np.min(tau):.2f}-{np.max(tau):.2f})')
            # Check convergence
            converged = np.all(tau * 100 < MCMCsampler.iteration)
            converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
            if converged:
                print("Converged!")
                break
            old_tau = tau
    # do not monitor, just run
    else:
        MCMCsampler = emcee.EnsembleSampler(nwalkers, ndims, log_posterior,
            args=[None, pars], pool=pool)
        MCMCsampler.run_mcmc(p0, nsteps, progress=True)

    ########################## Save MCMC outputs ###############################
    if rank==0:
        print(f'Pickling sampler to {outfile_sampler}')
        print(f'Saving chain to {outfile_sampler[:-3]+"chain"}')
        print(f'Saving data vector to {outfile_dv}')
        with open(outfile_sampler, 'wb') as f:
            pickle.dump(MCMCsampler, f)
        dv = get_GlobalDataVector(0)
        dv.to_fits(outfile_dv, overwrite=True)
        chains_flat = MCMCsampler.get_chain(flat=True)
        np.save(outfile_sampler[:-3]+"chain", chains_flat)
    ######################### Analysis the chains ##############################
    ### Read chains
    # with open(outfile_sampler, 'rb') as f:
    # MCMCsampler = pickle.load(f)
    chains = MCMCsampler.get_chain(flat=False)
    #chains_flat = MCMCsampler.get_chain(flat=True)
    # get blobs (priors, like)
    blobs = MCMCsampler.get_blobs(flat=False)
    blobs_flat = MCMCsampler.get_blobs(flat=True)

    ### build getdist.MCSamples object from the chains
    goodwalkers = np.where(blobs[-1,:,1]>-100)[0]
    if rank==0:
        print(f'Failed walkers {blobs.shape[1]-len(goodwalkers)}/{blobs.shape[1]}')
    samples = MCSamples(samples=[chains[nsteps//2:,gw,:] for gw in goodwalkers],
        loglikes=[-1*blobs[nsteps//2:,gw,:].sum(axis=1) for gw in goodwalkers],
        names = sampled_pars, 
        labels = [sampled_pars_label[k] for k in sampled_pars])

    ### 1. plot trace
    ### =============
    fig, axes = plt.subplots(ndims+1,1,figsize=(8,12), sharex=True)
    for i in range(ndims):
        for j in range(nwalkers):
            axes[i].plot(chains[:,j,i])
        axes[i].set(ylabel=r'$%s$'%sampled_pars_label[sampled_pars[i]])
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
    if rank==0:
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

    ### 6. fiber spectra
    ### ================
    _obs_id_, SNR_best = 0, [-np.inf,]
    if (len(emlines)>0):
        fig, axes = plt.subplots(len(offsets),len(emlines), figsize=(2*len(emlines),2*len(offsets)))
        for j, (emline, bid) in enumerate(zip(emlines, blockids)):
            wave = likelihood.get_GlobalLambdas(bid).mean(axis=1)
            emline_cen = np.mean(LINE_LAMBDAS[emline].to('Angstrom').value) * (1+pars.meta['sed']['z'])
            for i, (dx, dy) in enumerate(offsets):
                if len(emlines)==1:
                    ax = axes[i]
                else:
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
            if len(emlines)>1:
                axes[len(offsets)-1, j].set(xlabel="Wavelength [A]")
                axes[0, j].set(title=f'{emline}')
            else:
                axes[len(offsets)-1].set(xlabel="Wavelength [A]")
                axes[0].set(title=f'{emline}')
        plt.xlabel('wavelength [A]')
        plt.ylabel('ADU')
        plt.savefig(os.path.join(fig_dir, "spectra", filename_fmt+".png"))
        plt.close(fig)

    ### 7. broad-band image
    ### ===================
    if Nphot_used>0:
        fig, axes = plt.subplots(Nphot_used,3,figsize=(9,3*Nphot_used), sharey=True)
        for i in range(Nphot_used):
            row_axes = axes[:] if Nphot_used==1 else axes[i,:]
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
            ax1.text(0.05, 0.9, '%s Data (noise-free)'%(photometry_band[i]),
            color='white', transform=ax1.transAxes)
            ax2.text(0.05, 0.9, '%s Bestfit'%(photometry_band[i]),
            color='white', transform=ax2.transAxes)
            ax3.text(0.05, 0.9, '%s Redisuals'%(photometry_band[i]),
            color='white', transform=ax3.transAxes)
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
            if i==Nphot_used-1:
                for ax in row_axes:
                    ax.set(xlabel="X [arcsec]")
            _obs_id_ += 1

        plt.savefig(os.path.join(fig_dir,"image", filename_fmt+".png"))
        plt.close(fig)

    ### 5. save summary stats
    ### =====================
    with open(os.path.join(sum_dir, filename_fmt+".dat"), "w") as fp:
        res1 = "%d %.4f %.2f %.2f %le %d %d %le %le"%(args.Iflux, rmag, sini, hlr, PA, args.EXP_PHOTO, fiber_conf, sigma_e_rms, np.max(SNR_best))
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
    ####################### Parsing arguments ##################################
    parser = ArgumentParser()
    parser.add_argument('yaml', type=str, help="Input YAML file")
    parser.add_argument('-nsteps', type=int, default=-1,
                        help='Number of mcmc iterations per walker')
    parser.add_argument('-run_name', type=str, default='',help='MCMC run name')
    parser.add_argument('-Isini', type=int, default=-1, help='Index of sini')
    parser.add_argument('-Ipa', type=int, default=-1, help='Index of PA')
    parser.add_argument('-Ihlr', type=int, default=-1, help='Index of hlr')
    parser.add_argument('-Iflux', type=int, default=-1, help='Index of flux')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--mpi', dest='mpi', default=False, action='store_true',
                       help='Run with MPI.')
    group.add_argument('-ncores', default=1, type=int,
                        help='Number of processes (uses `multiprocessing` sequencial pool).')
    args = parser.parse_args()

    rc = main(args, None)

    if rc == 0:
        print('All tests ran successfully')
    else:
        print(f'Tests failed with return code of {rc}')
