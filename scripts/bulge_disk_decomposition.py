import sys
import numpy as np
import galsim as gs
import pocomc as pc
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import os
import glob
import re
from scipy.stats import uniform, norm, truncnorm, loguniform
from time import time
from argparse import ArgumentParser
import astropy.io.fits as fits
from kl_tools.parameters import ImagePars
from kl_tools.intensity import build_intensity_map
from kl_tools.interloper_mask import create_interloper_mask
from getdist import plots, MCSamples
from astropy.stats import SigmaClip
from photutils.background import StdBackgroundRMS
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

''' This script fits disk and bulge components to galaxy images using a 3D inclined exponential disk and 
a 3D inclined Sérsic bulge model. The disk and bulge components have independent inclination, position angle,
scale radius, flux, Sersic index (the disk component has a fixed Sersic index of 1), center offset, and aspect ratio.
This is designed to deal with bulge-disk misalignment (seperate inclination and position angle), 
and edge-on isophot (use 3D models instead of 2D models, with varying aspect ratio). 
NOTE: the profiles have zero shear.
'''
parser = ArgumentParser()
parser.add_argument('-ID', type=int, default=-1, help='ID of the object to fit')
parser.add_argument('--overwrite', action='store_true', default=False,
                    help='Overwrite existing outputs')
parser.add_argument('-nsteps', type=int, default=-1,
                    help='Number of mcmc iterations per walker')
parser.add_argument('-nparticles', type=int, default=512,
                    help='[pocoMC] Number of effective particles')
parser.add_argument('-ntotal', type=int, default=4096,
                    help='[pocoMC] Total number of effectively independent samples')

class BulgeDiskDecompositionModel(object):
    '''
    Class to build a bulge + inclined disk galaxy model using Galsim.

    Parameters
    ----------
    model_param_names : list
        List of model parameter names. Must include 'flux', 'scale_radius', 'sini', 
        'theta_int', 'n', 'q', 'x0', 'y0'.
    image_pars : dict
        Dictionary of image parameters. Must include 'shape', 'pixel_scale', etc.
    psf : GSObject or None, optional
        Optional PSF for convolution with the galaxy model.
    gsparams : galsim.GSParams or None, optional
        Optional GSParams for controlling the precision of the Galsim 
        calculations. If None, default GSParams are used.
    '''

    def __init__(
        self,
        model_param_names,
        model_param_labels,
        model_param_priors,
        image_pars,
        output_dir,
        OBJID,
        nparticles=512,
        ntotal=4096,
        psf=None,
        gsparams=None,
    ):
        self.model_param_names = model_param_names
        self.model_param_labels = model_param_labels
        self.priors = model_param_priors
        self.image_pars = image_pars
        self.output_dir = output_dir
        #assert type(psf) in [gs.GSObject, type(None)], "psf must be a GSObject or None."
        self.psf = psf
        self.OBJID = OBJID
        self.nparticles = nparticles
        self.ntotal = ntotal
        self.gsparams = gsparams
        self.bestfit = None
        return 

    def parse_model_param_list(self, params_list):
        '''
        Parse a list of model parameters into a dictionary.

        Parameters
        ----------
        params_list : list
            List of model parameters in the order specified by `model_param_names`.

        Returns
        -------
        model_params : dict
            Dictionary of model parameters.
        '''
        assert len(params_list) == len(self.model_param_names), \
            "Length of params_list must match length of model_param_names."
        model_params = {k:v for k,v in zip(self.model_param_names, params_list)}
        return model_params
    
    def build_model(self, model_params, which='all'):
        '''
        Build a bulge + inclined disk galaxy model using Galsim.

        Parameters
        ----------
        model_params : dict
            Dictionary of model parameters. Must include 'flux', 'half_light_radius', 'sini', 
            'theta_int', 'n', 'q', 'x0', 'y0'.
        image_pars : dict
            Dictionary of image parameters. Must include 'shape', 'pixel_scale', etc.
        psf : GSObject or None, optional
            Optional PSF for convolution with the galaxy model.
        gsparams : galsim.GSParams or None, optional
            Optional GSParams for controlling the precision of the Galsim 
            calculations. If None, default GSParams are used.
        '''
        model_params_dict = self.parse_model_param_list(model_params)
        # build intensity model
        if which=='disk' or which=='all':
            disk_inc = np.arcsin(model_params_dict['disk_sini'])
            disk_pa = model_params_dict['disk_theta_int']
            disk = gs.InclinedExponential(
                inclination=disk_inc * gs.radians,
                half_light_radius=model_params_dict['disk_hlr'],
                scale_h_over_r=model_params_dict['disk_q'],
                flux=model_params_dict['disk_flux'],
                gsparams=self.gsparams
            ).rotate(disk_pa * gs.radians).shift(
                model_params_dict['disk_x0'], model_params_dict['disk_y0']
            )
        else:
            disk = None
        if which=='bulge' or which=='all':
            disk_inc = np.arcsin(model_params_dict['disk_sini'])
            disk_pa = model_params_dict['disk_theta_int']
            # Inclined Sérsic profile has a bug?
            bulge_inc = disk_inc + model_params_dict['bulge_inc_offset']
            bulge_pa = disk_pa + model_params_dict['bulge_theta_int_offset']
            bulge_hlr = model_params_dict['bulge_hlr']
            #g1 = (1 - bulge_bovera) / (1 + bulge_bovera)
            bulge = gs.InclinedSersic(
                n=4,
                inclination=bulge_inc * gs.radians,
                half_light_radius=bulge_hlr,
                scale_h_over_r=model_params_dict['bulge_q'],
                flux=model_params_dict['bulge_flux'],
                trunc=2.0,
                gsparams=self.gsparams
            ).rotate(bulge_pa * gs.radians).shift(
                model_params_dict['bulge_x0'], model_params_dict['bulge_y0']
            )
        else:
            bulge = None
        # return rendered intensity profile
        if which == 'disk':
            gal = disk
        elif which == 'bulge':
            gal = bulge
        elif which == 'all':
            gal = disk + bulge
        else:
            raise ValueError("which must be 'disk', 'bulge', or 'all'.")
        if self.psf is not None:
            gal = gs.Convolve([gal, self.psf])

        try:
            image = gal.drawImage(
                nx=self.image_pars.Nx,
                ny=self.image_pars.Ny,
                scale=self.image_pars.pixel_scale,
                method='no_pixel',
            ).array
            return image
        except Exception as e:
            print("Bad galaxy model params:")
            for k, v in model_params_dict.items():
                print(f"  {k}: {v}")
            return np.zeros((self.image_pars.Ny, self.image_pars.Nx))
    
    # Define the likelihood function at module level for MPI compatibility
    def loglike(self, model_params, data, noise, mask, which='all'):
        '''
        Compute the log-likelihood of the data given the model.

        Parameters
        ----------
        model_params : np.ndarray
            Array of model parameters.
        data : ndarray
            The observed data.
        noise : ndarray
            The noise in the data.
        model : BulgeDiskDecompositionModel
            The model object.
        which : str
            Which component to fit ('all', 'disk', or 'bulge').
        Returns
        -------
        log_likelihood : float
            The log-likelihood of the data given the model.
        '''
        if which == 'all':
            model_params_dict = self.parse_model_param_list(model_params)
            if model_params_dict['disk_hlr'] <= model_params_dict['bulge_hlr']:
                return -np.inf
        try:
            model_image = self.build_model(model_params, which=which)
            residual = np.ma.array(data - model_image, mask=mask)
            chi2 = np.sum(residual**2 / np.ma.array(noise, mask=mask)**2)
            log_likelihood = -0.5 * chi2
            return log_likelihood
        except Exception as e:
            # Return very low likelihood if model evaluation fails
            return -np.inf

    def get_resume_state_path(self):
        checkpoint_root = os.path.join(self.output_dir, "pmc")
        statefiles = glob.glob(checkpoint_root + "_*.state")
        max_checkpoint = 0
        finished = False
        for statefile in statefiles:
            match = re.match(checkpoint_root + r"_(\S*).state", statefile)
            pattern = match.groups(0)[0]
            if pattern=='final':
                max_checkpoint = 'final'
                finished = True
                chain = np.genfromtxt(os.path.join(self.output_dir, "poco_chain.txt"), 
                                      skip_header=2, names=True)
                chain_array = chain[list(chain.dtype.names)].view(np.float64).reshape(len(chain), -1)
                self.bestfit = chain_array[np.argmax(chain['logpost'])][:-3]
                self.bestfit_dict = {k:v for k,v in zip(self.model_param_names, self.bestfit)}
                break
            else:
                max_checkpoint = max(max_checkpoint, int(pattern))
        if max_checkpoint==0:
            resume_state_path = None
        else:
            resume_state_path = os.path.join(self.output_dir, 
                            f'pmc_{max_checkpoint}.state')
        return resume_state_path, finished

    def write_sampler_to_file(self, sampler):
        samples, weights, logl, logp = sampler.posterior(resample=False)
        logZ, logZerr = sampler.evidence()
        data_block = [samples, weights[:,np.newaxis], logl[:,np.newaxis], logp[:,np.newaxis]]
        header = f'# logZ {logZ}\n# logZerr {logZerr}'
        header += "\n# " + " ".join(self.model_param_names) + " weight loglike logpost"
        data_block = np.hstack(data_block)
        # save chain
        np.savetxt(os.path.join(self.output_dir, "poco_chain.txt"), data_block, header=header, comments="")
        self.bestfit = samples[np.argmax(logp)]
        self.bestfit_dict = {k:v for k,v in zip(self.model_param_names, self.bestfit)}
        return

    def plot_posterior(self, sampler):
        samples, weights, logl, logp = sampler.posterior(resample=False)
        chain = MCSamples(samples=samples, weights=weights, names=self.model_param_names,
                        labels=self.model_param_labels, label=f'Bulge-Disk Decomposition')
        g = plots.get_subplot_plotter()
        g.triangle_plot([chain], filled=True)
        g.export(os.path.join(self.output_dir, f"ID_{self.OBJID}_post.png"))
        return
    
    def fit(self, data, noise, mask, pool, which='all', overwrite=False):
        if not overwrite:
            resume_state_path, finished = self.get_resume_state_path()
        else:
            print("Overwriting existing outputs.")
            resume_state_path = None
            finished = False
        if finished:
            print("Sampler already reached final state, loading best-fitting parameters from file.")
            return
        if not pool.is_master():
            pool.wait()
            sys.exit(0)
        else:
            sampler = pc.Sampler(
                    n_dim = len(self.model_param_names),
                    n_effective = self.nparticles,
                    n_active = self.nparticles // 2,
                    prior=self.priors,
                    likelihood=self.loglike,
                    likelihood_args = (data, noise, mask),
                    likelihood_kwargs={'which': which},
                    vectorize=False,
                    output_dir=self.output_dir,
                    pool=pool,
            )
            sampler.run(n_total=self.ntotal, save_every = 10,
                resume_state_path=resume_state_path)
        
            ### save output
            self.write_sampler_to_file(sampler)
            self.plot_posterior(sampler)

def main(args):
    ### read the 2D image cutout and PSF
    ### =======================================================
    data_root = "/xdisk/timeifler/jiachuanxu/jwst/fresco/extract_PaBr_2d_v2"
    hdul = fits.open(f'{data_root}/data_compile_short_GDS_ID{args.ID}_emlonly.fits')
    image = hdul["F444WIMG"].data 
    image_flux_guess_iter1 = np.sum(image)
    pixscale = hdul["F444WIMG"].header["PIXSCALE"]
    image_par = ImagePars(shape=image.shape, pixel_scale=pixscale, wcs=None, indexing='ij')
    noise = hdul["F444WERR"].data
    # build PSF
    psf_scale = hdul["F444WPSF"].header["PIXELSCL"]
    psf_img = gs.Image(hdul["F444WPSF"].data / np.sum(hdul["F444WPSF"].data), scale=psf_scale)
    psf = gs.InterpolatedImage(psf_img, flux=1.0)
    # build interloper mask
    interloper_OBJID_list = [201483, 202485, 205371, 211572, 216861] 
    if args.ID in interloper_OBJID_list:
        # build interloper mask
        sigma_clip = SigmaClip(sigma=3.0)
        bkgrms = StdBackgroundRMS(sigma_clip)
        bkgrms_value = bkgrms.calc_background_rms(image)
        mask = create_interloper_mask(image, bkgrms_value, threshold=3)
        masked_pixels = np.sum(mask)
        print(f"Interloper mask applied. Number of masked pixels: {masked_pixels}")
    else:
        mask = np.zeros_like(image, dtype=bool)
    chain_root_iter1 = "/xdisk/timeifler/jiachuanxu/jwst/fresco/bulge_disk_separation_v2/ID%d_iter1" % args.ID
    # Use a single pool for both iterations
    pool = schwimmbad.choose_pool(mpi=True, processes=1)
    
    ### build disk + bulge model (1st iteration)
    ### =======================================================  
    model_param_names_iter1 = [
        'disk_flux', 'disk_hlr', 'disk_sini', 'disk_theta_int', 'disk_q', 'disk_x0', 'disk_y0',
        'bulge_flux', 'bulge_hlr', 'bulge_inc_offset', 'bulge_theta_int_offset', 'bulge_q',
        'bulge_x0', 'bulge_y0'
    ]
    model_param_labels_iter1 = [
        r'F_\mathrm{d}', r'r_h^\mathrm{d}', r'\mathrm{sin}(i_\mathrm{d})$', r'\theta_\mathrm{int}^\mathrm{d}',
        r'q_\mathrm{d}', r'x_0^\mathrm{d}', r'y_0^\mathrm{d}',
        r'F_\mathrm{b}', r'r_h^\mathrm{b}',  r'i_\mathrm{b}-i_\mathrm{d}', 
        r'\theta_\mathrm{int}^\mathrm{b}-\theta_\mathrm{int}^\mathrm{d}', r'q_\mathrm{b}', 
        r'x_0^\mathrm{b}', r'y_0^\mathrm{b}'
    ]
    priors_iter1 = pc.Prior([
        loguniform(a=image_flux_guess_iter1*1e-3, b=image_flux_guess_iter1*2),  # disk_flux
        uniform(loc=0.001, scale=5.0),  # disk_hlr
        uniform(loc=0.01, scale=0.98),  # disk_sini
        uniform(loc=-np.pi/2., scale=np.pi),  # disk_theta_int
        uniform(loc=0.001, scale=0.299),  # disk_q
        truncnorm(-4.0, 4.0, loc=0.0, scale=0.05),  # disk_x0
        truncnorm(-4.0, 4.0, loc=0.0, scale=0.05),  # disk_y0
        loguniform(a=image_flux_guess_iter1*1e-8, b=image_flux_guess_iter1*2),  # bulge_flux
        uniform(loc=0.0001, scale=0.3),  # bulge_hlr
        truncnorm(-3, 3, loc=0.0, scale=10/180*np.pi),  # bulge_inc_offset, rad
        #norm(loc=0.01, scale=0.99),  # bulge_bovera
        truncnorm(-3, 3, loc=0.0, scale=10/180*np.pi),  # bulge_theta_int_offset, rad
        #uniform(loc=2, scale=4),  # bulge_n
        uniform(loc=0.1-0.0001, scale=0.0002),  # bulge_q
        truncnorm(-4, 4, loc=0.0, scale=0.025),  # bulge_x0
        truncnorm(-4, 4, loc=0.0, scale=0.025)   # bulge_y0
    ])

    model_iter1 = BulgeDiskDecompositionModel(
        model_param_names=model_param_names_iter1,
        model_param_labels=model_param_labels_iter1,
        model_param_priors=priors_iter1,
        image_pars=image_par,
        output_dir=chain_root_iter1,
        OBJID=args.ID,
        psf=psf,
        nparticles=args.nparticles,
        ntotal=args.ntotal,
        gsparams=None
    )

    # example evaluation of likelihood
    print(f'Example Evaluation of likelihood')
    start_time = time()
    test_model_params = np.array([image_flux_guess_iter1*0.5, 0.5, 0.5, 0.0, 0.1, 0.0, 0.0,
                         image_flux_guess_iter1*0.5, 0.05, 0.0, 0.0, 0.1, 0.0, 0.0])
    lglike_test = model_iter1.loglike(test_model_params, image, noise, mask)
    end_time = time()
    print(f'Time taken for likelihood evaluation: {end_time - start_time:.4f} seconds')
    print(f'Log-likelihood: {lglike_test}')

    ### First iteration
    model_iter1.fit(image, noise, mask, pool, overwrite=args.overwrite)
    print("Best-fitting parameters (iter1):")
    for name, val in zip(model_iter1.model_param_names, model_iter1.bestfit):
        print(f"  {name}: {val}")
    bestfit_image_iter1 = model_iter1.build_model(model_iter1.bestfit, which='all')
    bestfit_bulge_iter1 = model_iter1.build_model(model_iter1.bestfit, which='bulge')
    bestfit_disk_iter1 = model_iter1.build_model(model_iter1.bestfit, which='disk')
    residual_image_iter1 = image - bestfit_image_iter1

    ### Fit the disk component again after removing bulge + basis functions
    ### =======================================================
    # try fitting the residual with basis functions (to capture bars, spiral arms, etc.)
    exp_shapelet_pars = {
        'basis_type': 'exp_shapelets',
        'basis_plane': 'disk',
        'basis_kwargs': {
            'nmax': 25,  # number of shapelet coefficients
            # best-fitting scale radius from disk component
            'beta': model_iter1.bestfit_dict['disk_hlr'] * 0.005,
            'psf': psf,
        }
    }
    theta_pars = {
        'sini': model_iter1.bestfit_dict['disk_sini'],
        'theta_int': model_iter1.bestfit_dict['disk_theta_int'],
        'x0': model_iter1.bestfit_dict['disk_x0'],
        'y0': model_iter1.bestfit_dict['disk_y0'],
        'g1': 0.0,  # no shear
        'g2': 0.0,  # no shear
    }
    exp_shapelet_imap = build_intensity_map('basis', exp_shapelet_pars)
    exp_shapelet_image = exp_shapelet_imap.render(
        image_par,
        theta_pars,
        {},
        image=residual_image_iter1,
    )
    # build the bulge and basis function subtracted image 
    disk_estimate_image = image - bestfit_bulge_iter1 - exp_shapelet_image
    np.save(os.path.join(chain_root_iter1, "bestfit_disk_iter1.npy"), bestfit_disk_iter1)
    np.save(os.path.join(chain_root_iter1, "bestfit_bulge_iter1.npy"), bestfit_bulge_iter1)
    np.save(os.path.join(chain_root_iter1, "bestfit_image_iter1.npy"), bestfit_image_iter1)
    np.save(os.path.join(chain_root_iter1, "basis_function_fit_iter1.npy"), exp_shapelet_image)
    np.save(os.path.join(chain_root_iter1, "disk_estimate_image.npy"), disk_estimate_image)

    fig, axes = plt.subplots(2, 4, figsize=(8,4))
    extent = [-image_par.Nx/2*image_par.pixel_scale, image_par.Nx/2*image_par.pixel_scale,
              -image_par.Ny/2*image_par.pixel_scale, image_par.Ny/2*image_par.pixel_scale]
    p1 = axes[0,0].imshow(image, origin='lower', cmap='viridis', extent=extent)
    vmin, vmax = p1.get_clim()
    axes[0,0].text(0.05, 0.95, 'Original Image', color='white', fontsize=8, 
                   ha='left', va='top', transform=axes[0,0].transAxes)
    axes[0,0].text(0.05, 0.05, f'ID{args.ID}', color='white', fontsize=8, 
                   ha='left', va='bottom', transform=axes[0,0].transAxes)
    axes[0,1].imshow(bestfit_bulge_iter1, origin='lower', cmap='viridis', extent=extent, vmin=vmin, vmax=vmax)
    axes[0,1].text(0.05, 0.95, 'Bestfit Bulge', color='white', fontsize=8, 
                   ha='left', va='top', transform=axes[0,1].transAxes)
    axes[0,2].imshow(bestfit_disk_iter1, origin='lower', cmap='viridis', extent=extent, vmin=vmin, vmax=vmax)
    axes[0,2].text(0.05, 0.95, 'Bestfit Disk', color='white', fontsize=8, 
                   ha='left', va='top', transform=axes[0,2].transAxes)
    axes[0,3].imshow(bestfit_image_iter1, origin='lower', cmap='viridis', extent=extent, vmin=vmin, vmax=vmax)
    axes[0,3].text(0.05, 0.95, 'Bestfit B+D', color='white', fontsize=8, 
                   ha='left', va='top', transform=axes[0,3].transAxes)
    p2 = axes[1,0].imshow(residual_image_iter1, origin='lower', cmap='viridis', extent=extent)
    axes[1,0].text(0.05, 0.95, 'Residual', color='white', fontsize=8, 
                   ha='left', va='top', transform=axes[1,0].transAxes)
    rvmin, rvmax = p2.get_clim()
    axes[1,1].imshow(exp_shapelet_image, origin='lower', cmap='viridis', extent=extent, vmin=rvmin, vmax=rvmax)
    axes[1,1].text(0.05, 0.95, 'Basis Function Fit', color='white', fontsize=8, 
                   ha='left', va='top', transform=axes[1,1].transAxes)
    axes[1,2].imshow(disk_estimate_image, origin='lower', cmap='viridis', extent=extent, vmin=vmin, vmax=vmax)
    axes[1,2].text(0.05, 0.95, 'Disk Estimate', color='white', fontsize=8, 
                   ha='left', va='top', transform=axes[1,2].transAxes)

    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
        ax.tick_params(axis='both', which='both', direction='in', color='white')
        ax.xaxis.set_major_locator(MultipleLocator(0.5))
        ax.yaxis.set_major_locator(MultipleLocator(0.5))
        ax.xaxis.set_minor_locator(MultipleLocator(0.1))
        ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    for i in range(2):
        for j in range(4):
            ax = axes[i,j]
            if i==0:
                ax.set_xticklabels([])
            if j>0:
                ax.set_yticklabels([])
    plt.subplots_adjust(wspace=0.0, hspace=0.0)
    axes[1,3].set_visible(False)
    plt.savefig(os.path.join(chain_root_iter1, f"ID_{args.ID}_decomposition_results.png"),
                 dpi=300)
    plt.close()

    return 0


if __name__ == '__main__':
    args = parser.parse_args()
    rc = main(args)
    exit(rc)