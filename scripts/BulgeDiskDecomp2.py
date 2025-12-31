# second iteration of bulge-disk decomposition fit 
import sys
import numpy as np
import galsim as gs
import pocomc as pc
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import os
import glob
import re
from scipy.stats import uniform, norm, lognorm, loguniform
from time import time
from argparse import ArgumentParser
import astropy.io.fits as fits
from kl_tools.parameters import ImagePars
from kl_tools.intensity import build_intensity_map
from bulge_disk_decomposition import BulgeDiskDecompositionModel
from getdist import plots, MCSamples
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

parser = ArgumentParser()
parser.add_argument('-ID', type=int, default=-1, help='ID of the object to fit')
parser.add_argument('--not_overwrite', action='store_true', default=False,
                    help='Not Overwrite existing outputs')
parser.add_argument('-nsteps', type=int, default=-1,
                    help='Number of mcmc iterations per walker')
parser.add_argument('-nparticles', type=int, default=512,
                    help='[pocoMC] Number of effective particles')
parser.add_argument('-n_total', type=int, default=4096,
                    help='[pocoMC] Total number of effectively independent samples')

def main(args):
    ### read the 2D image cutout and PSF
    ### =======================================================
    data_root = "/xdisk/timeifler/jiachuanxu/jwst/fresco/extract_PaBr_2d_v2"
    chain_root_iter1 = "/xdisk/timeifler/jiachuanxu/jwst/fresco/bulge_disk_separation/ID%d_iter1" % args.ID
    chain_root_iter2 = "/xdisk/timeifler/jiachuanxu/jwst/fresco/bulge_disk_separation/ID%d_iter2" % args.ID
    hdul = fits.open(f'{data_root}/data_compile_short_GDS_ID{args.ID}_emlonly.fits')
    image = np.load(os.path.join(chain_root_iter1, "disk_estimate_image.npy"))
    pixscale = hdul["F444WIMG"].header["PIXSCALE"]
    image_par = ImagePars(shape=image.shape, pixel_scale=pixscale, wcs=None, indexing='ij')
    noise = hdul["F444WERR"].data
    # build PSF
    psf_scale = hdul["F444WPSF"].header["PIXELSCL"]
    psf_img = gs.Image(hdul["F444WPSF"].data / np.sum(hdul["F444WPSF"].data), scale=psf_scale)
    psf = gs.InterpolatedImage(psf_img, flux=1.0)
    
    # Use a single pool for both iterations
    pool = schwimmbad.choose_pool(mpi=True, processes=1)
    
    ### disk model (2nd iteration)
    ### ======================================================= 
    model_param_names_iter2 = [
        'disk_flux', 'disk_scale_radius', 'disk_sini', 'disk_theta_int', 'disk_q', 'disk_x0', 'disk_y0',
    ]
    model_param_labels_iter2 = [
        r'F_\mathrm{d}', r'r_e^\mathrm{d}', r'\mathrm{sin}(i_\mathrm{d})', r'\theta_\mathrm{int}^\mathrm{d}',
        r'q_\mathrm{d}', r'x_0^\mathrm{d}', r'y_0^\mathrm{d}',
    ]
    # define priors
    image_flux_guess_iter2 = np.sum(image)
    priors_iter2 = pc.Prior([
        loguniform(a=image_flux_guess_iter2*1e-3, b=image_flux_guess_iter2*2),  # disk_flux
        uniform(loc=0.01, scale=5.0),  # disk_scale_radius
        uniform(loc=0.01, scale=0.98),  # disk_sini
        uniform(loc=-np.pi/2., scale=np.pi),  # disk_theta_int
        uniform(loc=0.01, scale=0.29),  # disk_q
        norm(loc=0.0, scale=0.1),  # disk_x0
        norm(loc=0.0, scale=0.1),  # disk_y0
    ])
    model_iter2 = BulgeDiskDecompositionModel(
        model_param_names=model_param_names_iter2,
        model_param_labels=model_param_labels_iter2,
        model_param_priors=priors_iter2,
        image_pars=image_par,
        output_dir=chain_root_iter2,
        psf=psf,
        gsparams=None
    )

    ### Second iteration sampling
    model_iter2.fit(image, noise, pool, which='disk')
    print("Best-fitting parameters (iter2):")
    for name, val in zip(model_iter2.model_param_names, model_iter2.bestfit):
        print(f"  {name}: {val}")
    bestfit_disk_iter2 = model_iter2.build_model(model_iter2.bestfit, which='disk')
    np.save(os.path.join(chain_root_iter2, "bestfit_disk.npy"), bestfit_disk_iter2)

    fig, axes = plt.subplots(1, 3, figsize=(6,2))
    extent = [-image_par.Nx/2*image_par.pixel_scale, image_par.Nx/2*image_par.pixel_scale,
              -image_par.Ny/2*image_par.pixel_scale, image_par.Ny/2*image_par.pixel_scale]
    p1 = axes[0].imshow(image, origin='lower', cmap='viridis', extent=extent)
    vmin, vmax = p1.get_clim()
    axes[0].text(0.05, 0.95, 'Original Image', color='white', fontsize=8, 
                   ha='left', va='top', transform=axes[0].transAxes)
    axes[1].imshow(bestfit_disk_iter2, origin='lower', cmap='viridis', extent=extent,
                   vmin=vmin, vmax=vmax)
    axes[1].text(0.05, 0.95, 'Bestfit Disk', color='white', fontsize=8,
                   ha='left', va='top', transform=axes[1].transAxes)
    axes[2].imshow(image - bestfit_disk_iter2, origin='lower', cmap='viridis', extent=extent)
    axes[2].text(0.05, 0.95, 'Residual', color='white', fontsize=8,
                   ha='left', va='top', transform=axes[2].transAxes)
    
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.tick_params(axis='both', which='both', direction='in', color='white')
        ax.xaxis.set_major_locator(MultipleLocator(0.5))
        ax.yaxis.set_major_locator(MultipleLocator(0.5))
        ax.xaxis.set_minor_locator(MultipleLocator(0.1))
        ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    plt.subplots_adjust(wspace=0.0, hspace=0.0)
    plt.savefig(os.path.join(chain_root_iter2, 'decomposition_results_iter2.png'),
                 dpi=300)
    plt.close()

    return 0


if __name__ == '__main__':
    args = parser.parse_args()
    rc = main(args)
    exit(rc)