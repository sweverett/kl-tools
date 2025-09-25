import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import galsim as gs
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.units import Unit as u
from copy import copy, deepcopy
from mpl_toolkits.axes_grid1 import make_axes_locatable

import sep

from kl_tools.interloper_mask import create_interloper_mask
from kl_tools.kross.data import get_kross_obj_data
from kl_tools.kross.cube import KROSSDataCube
from kl_tools.kross.tfr import estimate_vtf
from kl_tools.velocity import VelocityMap
from kl_tools.intensity import IntensityMapFitter, build_intensity_map
from kl_tools.galaxy_fitter import estimate_gal_properties
from kl_tools.coordinates import OrientedAngle
from kl_tools.parameters import ImagePars
from kl_tools.basis import build_basis
from kl_tools.utils import add_colorbar, build_map_grid, MidpointNormalize

import ipdb

#kid = 171
#kid = 125
#kid = 20
#kid = 137
#kid = 11
#kid = 116
#kid = 63
#kid = 38
#kid = 122 # our favorite face-on.
#kid = 52
#kid = 112
#kid = 135 # fails due to galsim FFT size too large
#kid = 71
#kid = 49
#kid = 89
#kid = 35 nearly face-on (sini = 0.7)
#kid = 86
#kid = 44
#kid = 165
#kid = 15
#kid = 132
#kid = 143 # fails due to galsim FFT size too large
#kid = 76
#kid = 117
#kid = 152
#kid = 25
#kid = 47 # fails due to galsim FFT size too large
#kid = 28 # fails due to galsim FFT size too large
#kid = 155
#kid = 104 # fails due to galsim FFT size too large
#kid = 83
#kid = 45
#kid = 170
#kid = 23 # bad fit, hits parameter prior boundary.
#kid = 66
#kid = 94 
#kid = 108 
#kid = 101 # looks face on, sini = 0.1
#kid = 124
#kid = 139
#kid = 74 # fails due to galsim FFT size too large
#kid = 136
#kid = 72
#kid = 134
#kid = 69
#kid = 32 # Looks _extremely_ face on, but sini=0.62
#kid = 118 # fails due to galsim FFT size too large
#kid = 92
#kid = 33
kid = 53


cube = KROSSDataCube(kid)

# truncate to the Halpha line
cube.set_line('Ha')

# might be helpful later on
obj_data = get_kross_obj_data(kid)

image = obj_data['hst']
image_wcs = WCS(obj_data['hst_hdr'])
vmap = obj_data['velocity']
vmap_wcs = WCS(obj_data['velocity_hdr'])

name = obj_data['catalog']['NAME'][0]
z = obj_data['catalog']['Z'][0]
print(f'Name: {name}')
print(f'z: {z}')


# quick & dirty estimate of the bkg variance
bkg_mask = np.zeros(image.shape, dtype=bool)
bkg_mask[-50:,-50:] = True
bkg_var = np.nanvar(image[bkg_mask])
bkg_std = np.nanstd(image[bkg_mask])
bkg_image = image[-40:, -40:]
bkg_std = np.nanstd(bkg_image)
bkg_var = np.nanvar(bkg_image)

print("Generating adaptive mask with SEP...")
im_mask = create_interloper_mask(image, bkg_std, threshold=1.0,gaussian_sigma=5.0)
print(f"Masked {np.sum(im_mask)} pixels as interlopers.")

masked_image = image.copy()
masked_image[im_mask] = np.nan


Nrow, Ncol = image.shape
Nx, Ny = Ncol, Nrow
image_pars = ImagePars((Nx, Ny), indexing='xy', wcs=image_wcs)

image_flux = np.sum(image[~im_mask])
hst_pixel_scale = image_pars.pixel_scale # arcsec
print('HST pixel scale:', hst_pixel_scale, 'arcsec')

table = Table(obj_data['catalog'])
theta_int = OrientedAngle(
    table['VEL_PA'][0] * u('deg'),
    orientation='east-of-north'
).cartesian.radian
print('Velocity PA:', theta_int, 'deg')
inclination = table['THETA_IM'][0] # deg
sini = np.sin(np.radians(inclination))
print('Inclination:', inclination, 'deg')
print('Sini:', sini)
sersic_n = table['VDW12_N'][0]
print('Sersic n:', sersic_n)

guess = {
    'flux': image_flux,
    'scale_radius': hst_pixel_scale * 15,
    'theta_int': theta_int,
    'sini': sini,
    # 'n': sersic_n, # can be buggy, such as 0
    'n': 1.0,
}

bounds = {
    'flux': (0, 2*image_flux),
    'scale_radius': (hst_pixel_scale * 5, hst_pixel_scale * 100),
    'theta_int': tuple(theta_int + np.radians([-15, 15])),
    'n': (0.5, 4.5)
}

psf = None

result = estimate_gal_properties(
    image,
    image_pars,
    guess,
    bounds=bounds,
    psf=psf,
    mask=im_mask,
)

# unpack
model_image = result['model_image']
model_image[im_mask] = np.nan
fit_pars = result['params']


residual_image = model_image - masked_image

fitted_image = residual_image.copy()
# fitted_image = image.copy()

norm = MidpointNormalize(
    vmin=0.8*np.nanmin(fitted_image), vmax=0.8*np.nanmax(fitted_image)
    )

plt.imshow(fitted_image, origin='lower', norm=norm, cmap='RdBu_r')
plt.colorbar()
plt.title('Residual Image')
plt.gcf().set_size_inches(4, 3)

Nmax = 25
basis_plane = 'disk'
#basis_plane = 'obs'
#basis_plane = 'cen'
print(f'Using a beta of {fit_pars["scale_radius"]:.3f} arcsec')
print(f'Using a basis plane of {basis_plane}')
print(f'Using a Nmax of {Nmax}')

exp_shapelet_pars = {
    'basis_type': 'exp_shapelets',
    'basis_plane': basis_plane,
    'skip_ground_state': False,  # skip the ground state basis function
    'basis_kwargs': {
        'nmax': Nmax,  # Max order of the polar basis
        # 'beta': fit_pars['scale_radius'],
        'beta': fit_pars['scale_radius']/100.,
        'psf': psf,
    }
}

exp_shapelet_imap = build_intensity_map('basis', exp_shapelet_pars)

# transformation parameters
theta_pars = {
    'sini': fit_pars['sini'],
    'theta_int': fit_pars['theta_int'],
    'x0': fit_pars['x0'],
    'y0': fit_pars['y0'],
    'g1': 0.0,  # no shear
    'g2': 0.0,  # no shear
}
print("---------------------------")
print("transformation parameters:")
for k, v in theta_pars.items():
    print(f'{k}: {v}')
print("---------------------------")    
print("model fit parameters:")
for k, v in fit_pars.items():
    print(f'{k}: {v}')
print("---------------------------")

basis_image = exp_shapelet_imap.render(
    image_pars,
    theta_pars,
    None,
    image=image,
    mask=im_mask,
)
basis_image[im_mask] = np.nan
ipdb.set_trace()
## Now for plotting

data_vmin = 0.8*min(np.nanmin(basis_image), np.nanmin(fitted_image))
data_vmax = 0.8*max(np.nanmax(basis_image), np.nanmax(fitted_image))
data_norm = MidpointNormalize(vmin=data_vmin, vmax=data_vmax)
cmap = 'RdBu_r'

fig, axes = plt.subplots(2, 3, figsize=(10, 5))

# plt.subplot(331)
ax1 = axes[0,0]

im1 = ax1.imshow(image, origin='lower', cmap=cmap, norm=data_norm)
ax1.contour(im_mask, colors='black', linewidths=1.2, levels=[0.5])
divider = make_axes_locatable(ax1)
cax = divider.append_axes("right", size="5%", pad=0.05)
ax1.set_ylabel('Sersic Fit')

ax1.set_title(f'Data with Mask Outline (KID {kid})')
plt.colorbar(im1, cax=cax)

# plt.subplot(332)
ax2 = axes[0,1]
im2 = ax2.imshow(model_image, origin='lower', cmap=cmap, norm=data_norm)
divider = make_axes_locatable(ax2)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im2, cax=cax)
ax2.set_title('Inclined Sersic Model')
# plt.subplot(333)
ax3 = axes[0,2]
residual = model_image - masked_image
residual[im_mask] = np.nan
model_chi2 = np.nansum((residual / bkg_std) ** 2) - np.sum(im_mask) - len(fit_pars)
residuals_norm = MidpointNormalize(
    vmin=0.8*np.nanmin(residual), vmax=0.8*np
.nanmax(residual)
    )
sersic_chi2 = np.nansum((residual / bkg_std) ** 2) / (Nrow * Ncol - np.sum(im_mask) - len(fit_pars))
im3 = ax3.imshow(residual, origin='lower', norm=residuals_norm, cmap='RdBu_r')
divider = make_axes_locatable(ax3)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im3, cax=cax)
ax3.set_title(f'Sersic Residuals')# (Red Chi2: {sersic_chi2:.2f})')
# plt.subplot(334)
ax4 = axes[1,0]
im4 = ax4.imshow(
    masked_image, origin='lower', norm=data_norm, cmap='RdBu_r'
    )
divider = make_axes_locatable(ax4)
cax = divider.append_axes("right", size="5%", pad=0.05)
ax4.set_ylabel('Basis Fit')
plt.colorbar(im4, cax=cax)
ax4.set_title(f'Data (KID {kid})')
# plt.subplot(335)
ax5 = axes[1,1]
im5 = ax5.imshow(basis_image, origin='lower', cmap=cmap, norm=data_norm)
divider = make_axes_locatable(ax5)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im5, cax=cax)
ax5.set_title('Basis Model')
# plt.subplot(336)
ax6 = axes[1,2]
basis_diff = basis_image - masked_image
basis_chi2 = np.nansum((basis_diff / bkg_std) ** 2) - np.sum(im_mask) - Nmax**2
data_resid_norm = MidpointNormalize(
    vmin=0.8*np.nanmin(basis_diff), vmax=0.8*np.nanmax(basis_diff)
    )
im6 = ax6.imshow(
    basis_diff, origin='lower', cmap=cmap, norm=data_resid_norm
    )
divider = make_axes_locatable(ax6)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im6, cax=cax)
ax6.set_title(f'Basis Residuals')# (Red Chi2: {basis_chi2:.2f})')

plt.tight_layout()
plt.savefig(f"demo_plots/model_fit-{kid}")
plt.show()