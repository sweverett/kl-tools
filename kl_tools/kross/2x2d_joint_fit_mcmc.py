import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from argparse import ArgumentParser
from collections import OrderedDict
import matplotlib.pyplot as plt
import galsim as gs
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.units import Unit as u
from copy import copy, deepcopy
from nautilus import Prior, Sampler
from corner import corner

from kl_tools.kross.data import get_kross_obj_data
from kl_tools.kross.cube import KROSSDataCube
from kl_tools.kross.tfr import estimate_vtf
from kl_tools.velocity import VelocityMap, dist_to_major_axis
from kl_tools.psf import psf_convolved_vmap
from kl_tools.parameters import ImagePars
from kl_tools.coordinates import OrientedAngle
from kl_tools.interloper_mask import create_interloper_mask
from kl_tools.basis import build_basis
from kl_tools.utils import add_colorbar, build_map_grid
import kl_tools.intensity as intensity

def theta2pars(theta, r_unit=u('arcsec'), v_unit=u('km/s'), pars_type='both'):
    '''
    Map a fixed array of parameters to a dict of vmap parameters.

    We have more general tools to handle this, but this is a simple
    way to handle the fixed parameters in the KROSS model.
    '''

    vmap_pars = {
        'v0': theta[0],
        'vcirc': theta[1],
        'rscale': theta[2],
        'sini': theta[3],
        'theta_int': theta[4],
        'g1': theta[5],
        'g2': theta[6],
        'vmap_x0': theta[7],
        'vmap_y0': theta[8],
        'r_unit': r_unit,
        'v_unit': v_unit,
    }

    image_pars = {
        'flux': theta[9], # total flux of the disk
        'hlr': theta[10], # half-light radius of the disk,
        'image_x0': theta[11],
        'image_y0': theta[12],
    }

    if pars_type == 'vmap':
        pars = vmap_pars
    elif pars_type == 'image':
        pars = image_pars
    else:
        pars = {**vmap_pars, **image_pars}

    return pars

def pars2theta(pars):
    '''
    Map a dict of vmap parameters to a fixed array of parameters for the fitter

    We have more general tools to handle this, but this is a simple
    way to handle the fixed parameters in the KROSS model.
    '''

    theta_vmap = [
        pars['v0'],
        pars['vcirc'],
        pars['rscale'],
        pars['sini'],
        pars['theta_int'],
        pars['g1'],
        pars['g2'],
        pars['vmap_x0'],
        pars['vmap_y0'],
    ]

    theta_image = [
        pars['flux'], # total flux of the disk
        pars['hlr'], # half-light radius of the disk,
        pars['image_x0'],
        pars['image_y0'],
    ]

    return np.array(theta_vmap + theta_image)

def make_model_imap(theta, imap_data):
    imap = imap_data[0]
    imap_weights = imap_data[1]
    imap_mask = imap_data[2]
    imap_image_pars = imap_data[3]
    imap_wcs = imap_image_pars.wcs
    shape = imap.shape
    Nrow, Ncol = shape
    Nx, Ny = Ncol, Nrow

    # get the model pars
    pars_vmap = theta2pars(theta, pars_type='vmap')
    pars_imap = theta2pars(theta, pars_type='image')

    #----------------------------------------------------------------
    # make the disk component

    # NOTE: we need to do some hacky business here to use the typical vmap 
    # transformation parameters for the basis functions, but with a *different*
    # set of offset parameters
    transformation_pars = pars_vmap.copy()
    transformation_pars['x0'] = pars_imap['image_x0']
    transformation_pars['y0'] = pars_imap['image_y0']
    del transformation_pars['vmap_x0']
    del transformation_pars['vmap_y0']

    pars = {
        'psf': HST_PSF
    }

    imap_disk = intensity.InclinedExponential(
        pars_imap['flux'], pars_imap['hlr'],
    )

    disk_image = imap_disk.render(
        imap_image_pars,
        transformation_pars,
        pars=pars,
        image=None,
    )

    #----------------------------------------------------------------
    # make the basis component on the residuals

    residual_image = imap - disk_image

    # setup the basis
    Nbasis = NBASIS_COEFF
    Nmax = BASIS_NMAX
    if Nmax % 1 != 0:
        raise ValueError(
            'Number of basis functions must be a perfect square, for a polar ' 
            f'basis; {Nbasis} were passed'
            )
    Nmax = int(Nmax)
    if basis_type == 'exp_shapelets':
        basis_kwargs = {
            'nmax': Nmax,
            'beta': BASIS_BETA,
        }
    elif basis_type == 'sersiclets':
        basis_kwargs = {
            'nmax': Nmax,
            'beta': BASIS_BETA,
            'index': 1
            }

    if USE_BASIS_PSF is True:
        basis_kwargs['psf'] = HST_PSF

    imap_basis = intensity.BasisIntensityMap(
        basis_type,
        basis_kwargs,
        PLANE,
    )

    basis_image = imap_basis.render(
        imap_image_pars,
        transformation_pars,
        pars,
        weights=imap_weights,
        mask=imap_mask,
        image=residual_image,
        )

    if USE_BASIS is False:
        # for complciated reasons, we still have to run the actual basis fitting step
        # above, even if we now throw it out
        basis_image = np.zeros_like(disk_image)
        imap_basis.fitter.mle_coefficients = np.zeros(Nbasis, dtype=complex)

    # TODO: for testing only!
    # basis_image = np.zeros_like(basis_image)

    # TODO: for testing only!
    # disk_image = np.zeros_like(disk_image)

    model_image = disk_image + basis_image

    # now make the composite imap model for later PSF convolution
    model_imap = intensity.CompositeIntensityMap(
        imap_disk, imap_basis,
    )

    if (DEBUG_IMAP is True) and (np.random.random() > DEBUG_THRESH):
        plt.subplot(131)
        im = plt.imshow(model_image, origin='lower')
        add_colorbar(im)
        plt.title('Model')
        plt.subplot(132)
        im = plt.imshow(imap, origin='lower')
        add_colorbar(im)
        plt.title('HST Image')
        plt.subplot(133)
        im = plt.imshow(imap - model_image, origin='lower')
        add_colorbar(im)
        plt.title('Data - Model')
        plt.gcf().set_size_inches(16, 4)
        plt.show()
        for k,v in pars_imap.items():
            print(f'{k}: {v}')

    return model_image, model_imap

def make_model_vmap(theta, vmap_data, model_imap):

    vmap = vmap_data[0]
    vmap_image_pars = vmap_data[3]
    vmap_wcs = vmap_image_pars.wcs

    # get the pixel scale of the velocity map
    pixel_scale = vmap_image_pars.pixel_scale

    # NOTE: have to do some hacky business to allow for different offsets
    # between the vmap and the iamge
    vmap_pars = theta2pars(theta, pars_type='vmap')
    vmap_pars['x0'] = vmap_pars['vmap_x0']
    vmap_pars['y0'] = vmap_pars['vmap_y0']
    del vmap_pars['vmap_x0']
    del vmap_pars['vmap_y0']
    model_vmap = VelocityMap('offset', vmap_pars)

    # now apply the PSF convolution, if desired
    if KROSS_PSF is not None:
        model_vmap_image = psf_convolved_vmap(
            model_imap,
            model_vmap,
            vmap_image_pars,
            KROSS_PSF,
            vmap_pars,
        )
    else:
        Nrow, Ncol = vmap.shape
        Nx, Ny = Ncol, Nrow
        X, Y= build_map_grid(Nx, Ny, indexing='xy')

        Xarcsec = X * pixel_scale
        Yarcsec = Y * pixel_scale
        model_vmap_image = model_vmap(PLANE, Xarcsec, Yarcsec)

    if (DEBUG_VMAP is True) and (np.random.random() > DEBUG_THRESH):
        msk = np.where(vmap == 0)
        mdl = model_vmap_image.copy()
        mdl[msk] = 0
        plt.subplot(131)
        im = plt.imshow(mdl, origin='lower')
        add_colorbar(im)
        plt.title('Model')
        plt.subplot(132)
        im = plt.imshow(vmap, origin='lower')
        add_colorbar(im)
        plt.title('Data')
        plt.subplot(133)
        diff = vmap - mdl
        chi = np.sum(diff**2) / (mdl.shape[0]*mdl.shape[1])
        im = plt.imshow(diff, origin='lower')
        add_colorbar(im)
        plt.title(f'Data - Model; Chi2: {chi:.4f}')
        plt.gcf().set_size_inches(16, 4)
        plt.show()
        for k,v in vmap_pars.items():
            print(f'{k}: {v}')

    return model_vmap_image, model_vmap

def compute_model(theta, datavector):
    imap_data = datavector[0]
    vmap_data = datavector[1]

    model_imap_image, model_imap = make_model_imap(theta, imap_data)
    model_vmap_image, model_vmap = make_model_vmap(theta, vmap_data, model_imap)

    model = [
        (model_imap_image, model_imap),
        (model_vmap_image, model_vmap),
    ]

    return model

def compute_residuals(datavector, model):
    image_tuple = datavector[0]
    vmap_tuple = datavector[1]

    image = image_tuple[0]
    image_weights = image_tuple[1]
    image_mask = image_tuple[2]
    vmap = vmap_tuple[0]
    vmap_weights = vmap_tuple[1]
    vmap_mask = vmap_tuple[2]

    # reshape data
    image = image.ravel()
    image_weights = image_weights.ravel()
    image_mask = image_mask.ravel()
    vmap = vmap.ravel()
    vmap_weights = vmap_weights.ravel()
    vmap_mask = vmap_mask.ravel()

    # these are the model images themselves, not the model objects
    model_image = model[0][0].ravel()
    model_vmap = model[1][0].ravel()

    image_residuals = (image - model_image) * image_weights * (~image_mask)
    vmap_residuals = (vmap - model_vmap) * vmap_weights * (~vmap_mask)

    return (image_residuals, vmap_residuals)

def log_objective(theta, datavector=None, l1_lambda=None, l2_lambda=None):
    '''
    NOTE: these "kwargs" are not optional, its a hack to make nautilus work
    '''

    # different packages use different conventions
    if isinstance(theta, dict):
        theta = pars2theta(theta)

    model = compute_model(theta, datavector)
    residuals = compute_residuals(datavector, model)

    # [0] are the image residuals, [1] are the vmap residuals
    imap_chi2 = np.sum((residuals[0]/IMAP_BKG_STD)**2)
    vmap_chi2 = np.sum((residuals[1]/VMAP_BKG_STD)**2)
    if CHI_TYPE == 'both':
        chi2 = imap_chi2 + vmap_chi2
    elif CHI_TYPE == 'image':
        chi2 = imap_chi2
    elif CHI_TYPE == 'vmap':
        chi2 = vmap_chi2
    else:
        raise ValueError(f'Unknown CHI_TYPE: {CHI_TYPE}')

    basis_coeff = model[0][1].basis_imap.fitter.mle_coefficients.real

    if CHI_TYPE in ['both', 'image']:
        l1_norm = l1_lambda * np.sum(np.abs(basis_coeff))
        l2_norm = l2_lambda * np.sum(basis_coeff**2)
    else:
        l1_norm = 0.0
        l2_norm = 0.0

    # try to normalize the residuals
    imap_mask = datavector[0][2]
    vmap_mask = datavector[1][2]

    Nimap = np.sum(~imap_mask)
    Nvmap = np.sum(~vmap_mask)
    Npars = Nimap + Nvmap - len(theta)

    return (0.5 * chi2 + l1_norm + l2_norm) / Npars

#-------------------------------------------------------------------------------
# Test the objective function

# START HERE

plot = True
DEBUG_IMAP = False # extra plots for debugging
DEBUG_VMAP = False # extra plots for debugging
DEBUG_THRESH = 0.975 # number between 0 and 1; higher = fewer plots
vb = False
PLANE = 'cen'
CHI_TYPE = 'both' # 'image', 'vmap', or 'both'
basis_type = 'exp_shapelets'
BASIS_NMAX = 1
BASIS_BETA = 0.03 / 10
NBASIS_COEFF = BASIS_NMAX**2 # for a polar basis
USE_BASIS = False # whether to use basis functions at all
# basis_type = 'sersiclets'
HST_PSF = gs.Gaussian(fwhm=0.12) # approximate HST PSF
# KROSS_PSF = gs.Gaussian(fwhm=0.7) # approximate KROSS PSF
KROSS_PSF = None # approximate KROSS PSF
USE_BASIS_PSF = False
USE_SLIT_MASK = True
SLIT_MASK_WIDTH = 6.0 # vmap pixels
VMAP_BKG_STD = 3.0
lambda1_val = 0.0
lambda2_val = 1e4
# PLANE = 'obs' # plane of the basis functions. Typicaly 'obs' or 'disk'
# OPTIMIZER = 'scipy'
OPTIMIZER = 'nautilus'
OPTIMIZER_NITER = 100

KID = 171
# KID = 20
# KID = 11 # compact and offset
# KID = 116 # spiral arms with shape; offset

obj_data = get_kross_obj_data(KID)
row = obj_data['catalog']

imap = obj_data['hst']
vmap = obj_data['velocity']

imap_wcs = WCS(obj_data['hst_hdr'])
vmap_wcs = WCS(obj_data['velocity_hdr'])

imap_image_pars = ImagePars(
    imap.shape, wcs=imap_wcs, indexing='ij'
)
vmap_image_pars = ImagePars(
    vmap.shape, wcs=vmap_wcs, indexing='ij'
)

imap_pixel_scale = imap_image_pars.pixel_scale # arcsec / pixel
vmap_pixel_scale = vmap_image_pars.pixel_scale # arcsec / pixel

datavector = [imap, vmap]

Nimap = 4 # extra parameters for the imap model
# Nimap = Nbasis_coeff + Nimap # add a disk flux & hlr + basis scale factor & offset to scan over
Nvmap = 9 # fixed for an offset vmap model
Ntheta = Nvmap + Nimap
theta_vmap = np.random.rand(Nvmap)
theta_image = np.random.random(Nimap)
theta = np.concatenate((theta_vmap, theta_image))

# get an estimate for the bkg and interloper mask
# quick & dirty estimate of the bkg variance
bkg_image = imap[-40:, -40:]
bkg_std = np.nanstd(bkg_image)
bkg_var = bkg_std**2
imap_mask = create_interloper_mask(imap, bkg_std, threshold=3)
IMAP_BKG_STD = bkg_std

imap_weights = np.ones_like(imap)
vmap_weights = np.ones_like(vmap)

vmap_mask = np.zeros_like(vmap, dtype=bool)
vmap_mask[vmap == 0] = 1

if USE_SLIT_MASK is True:
    # add an extra mask to only look near the kinematic major axis
    slit_width = SLIT_MASK_WIDTH * vmap_pixel_scale # arcsec
    X, Y = build_map_grid(vmap_image_pars.Nx, vmap_image_pars.Ny, indexing='xy')
    Xarcsec = X * vmap_pixel_scale
    Yarcsec = Y * vmap_pixel_scale

    vel_pa = row['VEL_PA'][0]
    vel_pa = OrientedAngle(vel_pa, unit='deg', orientation='east-of-north')

    # NOTE: We assume the vmap is centered on the image for the slit, which KROSS does
    dist = dist_to_major_axis(Xarcsec, Yarcsec, 0, 0, position_angle=vel_pa)
    slit_mask = np.abs(dist) > (slit_width / 2)

    vmap_mask = vmap_mask | slit_mask

vmap_weights[vmap_mask] = 0
datavector = [
    (imap, imap_weights, imap_mask, imap_image_pars),
    (vmap, vmap_weights, vmap_mask, vmap_image_pars),
    ]

# make a plot of the datavector

if vb is True:
    plt.subplot(121)
    plt_image = copy(imap)
    plt_image[imap_mask] = np.nan
    im = plt.imshow(plt_image, origin='lower')
    add_colorbar(im)
    plt.title('HST Image')

    plt.subplot(122)
    plt_vmap = copy(vmap)
    plt_vmap[vmap_mask] = np.nan
    im = plt.imshow(plt_vmap, origin='lower')
    add_colorbar(im)
    plt.title('Velocity Map')
    plt.gcf().set_size_inches(6, 3)
    plt.show()

#-------------------------------------------------------------------------------
# get sensible initial values for the vmap parameters

vel_pa = row['VEL_PA'][0] # position angle of the measured vmap
theta_im = row['THETA_IM'][0] # inclination angle of the galaxy
# v22_obs = row['V22_OBS'][0] # observed velocity at 2.2 x Rd
mstar = row['MASS'][0]

# derived from KROSS
log_mstar = np.log10(mstar)
sini_kross = np.sin(np.deg2rad(theta_im))

# we'll use the KROSS PA to set the initial guess for theta_int
vel_pa = OrientedAngle(vel_pa, unit='deg', orientation='east-of-north')

vtf, v_bounds = estimate_vtf(log_mstar, return_error_bounds=True)
vlow, vhigh = vtf - v_bounds[0], vtf + v_bounds[1]

sig_vtf = 1
vtf_scatter_dex = sig_vtf * 0.05
vtf_fator = 10**vtf_scatter_dex
vcirc_base = vtf * sini_kross
vcirc_low = vcirc_base / vtf_fator
vcirc_high = vcirc_base * vtf_fator

# NOTE: FOR TESTING ONLY!!
vcirc_low = 0.2*vcirc_base
vcirc_high = 2*vcirc_base

sini_low = 0.25 * sini_kross
sini_high = 2 * sini_kross
sini_low = max(0, sini_low)
sini_high = min(1, sini_high)

if vb is True:
    print('---------')
    print(f'vtf: {vtf:.2f}')
    print(f'vlow: {vlow:.2f}')
    print(f'vhigh: {vhigh:.2f}')
    print('---------')
    print(f'vcirc: {vcirc_base:.2f}')
    print(f'vcirc_low: {vcirc_low:.2f}')
    print(f'vcirc_high: {vcirc_high:.2f}')
    print('---------')
    print(f'sini: {sini_kross:.2f}')
    print(f'sini_low: {sini_low:.2f}')
    print(f'sini_high: {sini_high:.2f}')
    print('---------')

base_bounds = OrderedDict([
    ('v0', (-20, 20)),
    ('vcirc', (vcirc_low, vcirc_high)),
    ('rscale', (vmap_pixel_scale, 20*vmap_pixel_scale)), # KROSS pixels
    ('sini', (sini_low, sini_high)),
    ('theta_int', (0, 2*np.pi)),
    ('g1', (-.0000001, .0000001)),
    ('g2', (-.0000001, .0000001)),
    ('vmap_x0', (-8*vmap_pixel_scale, 8*vmap_pixel_scale)), # vmap x0
    ('vmap_y0', (-8*vmap_pixel_scale, 8*vmap_pixel_scale)), # vmap y0
    ('flux', (0.25*np.nansum(imap), 2*np.nansum(imap))), # disk flux
    ('hlr', (imap_pixel_scale, 25*imap_pixel_scale)),
    ('image_x0', (-10*imap_pixel_scale, 10*imap_pixel_scale)), # HST pixel offset
    ('image_y0', (-10*imap_pixel_scale, 10*imap_pixel_scale)), # HST pixel offset
    ])

# initial guess for the optimizer
base_initial_theta = [
    0.0, # v0
    vcirc_base, # vcirc
    3*vmap_pixel_scale, # rscale; KROSS pixels
    sini_kross, # sini
    vel_pa.cartesian.rad, # theta_int
    0.0, # g1
    0.0, # g2
    0.0, # vmap x0
    0.0, # vmap y0
    np.nansum(imap), # disk flux
    10*imap_pixel_scale, # hlr; arcsec; N x HST pixels
    0.0, # HST pixel offset
    0.0, # HST pixel offset
    ]

# combine with unconstrained basis coefficients
bounds = list(
    base_bounds[k] for k in base_bounds.keys()
)

# Ensure initial theta values are within the bounds
initial_theta = np.array([
    base_initial_theta[i] if (i < Nvmap+5) else np.random.rand() for i in range(Ntheta)
])

print('Initial theta:', initial_theta)

if OPTIMIZER == 'scipy':
    result = minimize(
        log_objective,
        initial_theta,
        args=(datavector, lambda1_val, lambda2_val),
        method='L-BFGS-B',
        # method='Powell',
        bounds=bounds,
        tol=1e-8,
    )

    optimized_theta = result.x

    print("Optimization success:", result.success)

elif OPTIMIZER == 'pso_bacco':
    import pso_bacco

    # reformat to what pso_bacco expects
    params = {'c1': 0.7, 'c2': 0.2, 'w': 0.9}
    array_bounds = np.array([
        [b[0], b[1]] for b in bounds
    ])
    pso = pso_bacco.pso_bacco.global_PSO(
        bounds=array_bounds,
        params=params,
        npoints=30
        )

    pso.run(
        log_objective,
        func_argv=(datavector, lambda1_val, lambda2_val),
        niter=OPTIMIZER_NITER,
        # backup_name='backup.h5'
        )

    optimized_theta = pso.swarm['pos_bglobal']

    print('Optimization success: True')

elif OPTIMIZER == 'nautilus':
    prior = Prior()
    for key, bound in base_bounds.items():
        prior.add_parameter(key, dist=bound)

    sampler = Sampler(
        prior,
        log_objective,
        likelihood_kwargs=
        {'datavector': datavector, 'l1_lambda': lambda1_val, 'l2_lambda': lambda2_val
        },
        pool=1,
        filepath='./nautilus_kross_test/checkpoint.h5',
        resume=True,
        # nlive=1000,
    )

    sampler.run(verbose=True)

    print('Optimization success: True')

    points, log_w, log_l = sampler.posterior()
    corner(
        points, weights=np.exp(log_w), bins=20, labels=prior.keys, color='purple',
        plot_datapoints=False, range=np.repeat(0.999, len(prior.keys))
        )
    plt.show()

else:
    raise ValueError(f'Unknown OPTIMIZER: {OPTIMIZER}')

optimized_log_obj = log_objective(
    optimized_theta, datavector, lambda1_val
    )

print("Optimized parameters:")
optimized_vmap_pars = theta2pars(optimized_theta, pars_type='vmap')
optimized_imap_pars = theta2pars(optimized_theta, pars_type='image')
print('Vmap:')
for k, val in optimized_vmap_pars.items():
    print(f"  {k}: {val}")
print('Image:')
for k, val in optimized_imap_pars.items():
    print(f"  {k}: {val}")

opt_imap, opt_model_imap = make_model_imap(optimized_theta, datavector[0])
opt_vmap, opt_model_vmap = make_model_vmap(
    optimized_theta, datavector[1], opt_model_imap
    )

# apply the same masks
image_mask = copy(datavector[0][2])
vmap_mask = copy(datavector[1][2])

opt_imap[image_mask] = np.nan
opt_vmap[vmap_mask] = np.nan

image_vmin = np.nanmin([imap, opt_imap])
image_vmax = np.nanmax([imap, opt_imap])
vmap_vmin = np.nanmin([vmap, opt_vmap])
vmap_vmax = np.nanmax([vmap, opt_vmap])

plt.subplot(231)
im = plt.imshow(plt_image, origin='lower', vmin=image_vmin, vmax=image_vmax)
add_colorbar(im)
plt.title('Data')
plt.subplot(232)
im = plt.imshow(opt_imap, origin='lower', vmin=image_vmin, vmax=image_vmax)
add_colorbar(im)
plt.title('Model')
plt.subplot(233)
im = plt.imshow(plt_image - opt_imap, origin='lower')
add_colorbar(im)
plt.title('Data - Model')
plt.subplot(234)
im = plt.imshow(plt_vmap, origin='lower', vmin=vmap_vmin, vmax=vmap_vmax)
plt.title('Data')
add_colorbar(im)
plt.subplot(235)
im = plt.imshow(opt_vmap, origin='lower', vmin=vmap_vmin, vmax=vmap_vmax)
add_colorbar(im)
plt.title('Model')
plt.subplot(236)
im = plt.imshow(vmap - opt_vmap, origin='lower')
add_colorbar(im)
plt.title('Data - Model')

plt.suptitle(
    f'KID {KID} Optimization\nOptimal Log Objective: {optimized_log_obj:.2f}'
    )

plt.gcf().set_size_inches(12, 6)
plt.tight_layout()
plt.show()