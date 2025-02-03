import numpy as np
from argparse import ArgumentParser
from astropy.io import fits
from astropy.table import Table
from astropy.units import Unit as u
from astropy.wcs import WCS
import fitsio
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

from kl_tools.utils import get_base_dir, build_map_grid, MidpointNormalize, make_dir, plot
from kl_tools.velocity import VelocityMap
from kl_tools.coordinates import OrientedAngle
from kl_tools.kross.tfr import estimate_vtf
from kl_tools.kross.kross_utils import theta2pars, pars2theta

def parse_args():
    parser = ArgumentParser()

    parser.add_argument('--show', action='store_true', default=False)
    parser.add_argument('--vb', action='store_true', default=False)

    return parser.parse_args()

#---------------------------------------------------------------------------
# the following methods define the velocity map fitter 

def get_model(pars, X, Y):
    vmap = VelocityMap('offset', pars)
    return vmap('obs', X, Y)

def residuals(theta, data, weights, mask, X, Y):
    pars = theta2pars(theta)
    model = get_model(pars, X, Y)
    res = ((data - model) * np.sqrt(weights) * mask).flatten()

    return res

def fit_model(data, weights, mask, initial_guess, bounds=(-np.inf, np.inf),
              method='lm', loss='linear'):
    Nrow, Ncol = data.shape
    Nx, Ny = Ncol, Nrow
    X, Y = build_map_grid(Nx, Ny, indexing='xy')
    return least_squares(
        residuals, initial_guess, args=(data, weights, mask, X, Y), method=method, bounds=bounds, loss=loss
        )

#---------------------------------------------------------------------------

def main():

    #---------------------------------------------------------------------------
    # initial setup

    args = parse_args()
    vb = args.vb
    show = args.show
    save = True

    out_dir = get_base_dir() / 'notebooks/kross/vmap_fitter_debug'
    make_dir(out_dir)

    # NOTE: If you are missing any relevant data products, they can be
    # downloaded at the following link:
    # https://astro.dur.ac.uk/KROSS/data.html
    kross_dir = get_base_dir() / 'data/kross'
    vmap_dir = kross_dir / 'vmaps'
    kross = Table.read(kross_dir / 'kross_release_v2.fits')

    #---------------------------------------------------------------------------
    # Set up the KROSS object we will debug

    # obj we will debug
    # name is 'C-zcos_z1_925', but it's eaiser to access by KID
    kid = 171
    obj = kross[kross['KID'] == kid]

    name = obj['NAME'][0].strip() # C-zcos_z1_925
    vel_pa = obj['VEL_PA'][0] # position angle of the measured vmap
    theta_im = obj['THETA_IM'][0] # inclination angle of the galaxy
    v22_obs = obj['V22_OBS'][0] # observed velocity at 2.2 x Rd
    mstar = obj['MASS'][0]
    log_mstar = np.log10(mstar)

    # we'll use the KROSS PA to set the initial guess for theta_int
    vel_pa = OrientedAngle(vel_pa, unit='deg', orientation='east-of-north')

    #---------------------------------------------------------------------------
    # load vmap and setup weights/mask

    vmap_file = vmap_dir / f'{name}.fits'
    vmap = fitsio.read(vmap_file)

    norm = MidpointNormalize(
        vmin=np.percentile(vmap, 1), vmax=np.percentile(vmap, 99)
    )

    mask = np.ones(vmap.shape, dtype=bool)
    mask[vmap == 0.0] = 0

    # setup pixel weights
    weights = np.ones(vmap.shape)
    weights[~mask] = 0.

    #---------------------------------------------------------------------------
    # to set vcirc bounds, we need to use TF relation
    vtf, v_bounds = estimate_vtf(log_mstar, return_error_bounds=True)
    vlow, vhigh = vtf - v_bounds[0], vtf + v_bounds[1]
    if vb is True:
        print(f'vtf: {vtf:.2f}')
        print(f'vlow: {vlow:.2f}')
        print(f'vhigh: {vhigh:.2f}')
        print('---------')

    #---------------------------------------------------------------------------
    # setup fitter bounds and initial guess

    bounds_pair = [
        (-20, 20), # v0
        (vtf-25, vtf+25), # vcirc
        (0, 20), # rscale
        (0, 1), # sini
        (0, 2*np.pi), # theta_int
        (-0.1, 0.1), # g1
        (-0.1, 0.1), # g2
        (-5, 5), # x0
        (-5, 5) # y0
    ]

    # initial guess for the optimizer
    initial_guess = pars2theta({
        'v0': 0.0,
        'vcirc': vtf,
        'rscale': 2.0,
        'sini': 0.5,
        'theta_int': vel_pa.cartesian.rad,
        'g1': 0.0,
        'g2': 0.0,
        'x0': 0.0,
        'y0': 0.0,
        'r_unit': u('arcsec'),
        'v_unit': u('km/s'),
    })

    # least_squares wants it formatted differently
    bounds = [[], []]
    for i in range(len(bounds_pair)):
        bounds[0].append(bounds_pair[i][0])
        bounds[1].append(bounds_pair[i][1])

    #---------------------------------------------------------------------------
    # setup and run the fitter

    method = 'trf' # lm doesn't support bounds
    loss = 'soft_l1' # more robust to outliers
    # loss = 'linear'
    result = fit_model(
        vmap, weights, mask, initial_guess, bounds=bounds,
        method=method, loss=loss
        )
    fit_theta = result.x
    fit_pars = theta2pars(fit_theta)
    chi2 = result.cost / (np.sum(mask) - len(fit_theta))
    if vb is True:
        for key, val in fit_pars.items():
            if isinstance(val, float):
                print(f'{key}: {val:.2f}')
            else:
                print(f'{key}: {val}')

    #---------------------------------------------------------------------------
    # make model fit vmap

    Nrow, Ncol = vmap.shape
    Nx, Ny = Ncol, Nrow
    X, Y = build_map_grid(Nx, Ny, indexing='xy')
    fit = VelocityMap('offset', fit_pars)('obs', X, Y)
    fit[~mask] = 0

    #---------------------------------------------------------------------------
    # make comparison plots

    cmap = 'RdBu_r'
    plt.subplot(131)
    norm = MidpointNormalize(
        vmin=np.percentile(fit, 1), vmax=np.percentile(fit, 99)
    )
    im = plt.imshow(fit, origin='lower', norm=norm, cmap=cmap)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title(f'Fit (Chi2: {chi2:.2f})')
    ipar = 0
    plt.text(
        0.025, 0.95, 'Fit Parameters:', color='k', transform=plt.gca().transAxes
        )
    for key, val in fit_pars.items():
        if isinstance(val, float):
            plt.text(
                0.025, 0.9-ipar*0.05,
                f'{key}: {val:.2f}',
                color='k',
                transform=plt.gca().transAxes
                )
        ipar += 1

    plt.subplot(132)
    norm = MidpointNormalize(
        vmin=np.percentile(vmap, 1), vmax=np.percentile(vmap, 99)
    )
    im = plt.imshow(vmap, origin='lower', norm=norm, cmap=cmap)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    meas_pa = OrientedAngle(
        fit_pars['theta_int'], unit='rad', orientation='cartesian'
        )
    plt.text(
        0.025, 0.95, 'KROSS Parameters', color='k', transform=plt.gca().transAxes
    )
    plt.text(
        0.025, 0.9, f'V22_OBS: {v22_obs:.2f} km/s', color='k', transform=plt.gca().transAxes
    )
    plt.text(
        0.025, 0.85, 
        f'sini: {np.sin(np.deg2rad(theta_im)):.2f}',
        color='k', transform=plt.gca().transAxes
    )
    plt.text(
        0.025, 0.8, 
        f'x0/y0: {0}',
        color='k', transform=plt.gca().transAxes
    )
    plt.text(
        0.05,
        0.10,
        f'KROSS PA: {vel_pa.east_of_north.deg:.2f} deg (East-of-North)',
        color='k',
        transform=plt.gca().transAxes
        )
    plt.text(
        0.05,
        0.05,
        f'Meas PA:    {meas_pa.east_of_north.deg:.2f} deg (East-of-North)',
        color='k',
        transform=plt.gca().transAxes
        )
    plt.title('Data')

    plt.subplot(133)
    norm = MidpointNormalize(
        vmin=np.percentile(fit-vmap, 1), vmax=np.percentile(fit-vmap, 99)
    )
    resid = (fit-vmap) * mask
    resid[mask] / vmap[mask]
    im = plt.imshow(resid, origin='lower', norm=norm, cmap=cmap)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title('% Residuals (Fit - vmap)')

    plt.suptitle(f'{name}; KID={kid}')
    plt.gcf().set_size_inches(16, 5)

    plot(show, save, out_dir / f'compare_vmap_fit_{name}.png')

    return

if __name__ == '__main__':
    main()