import numpy as np
import fitsio
import pickle
from scipy.optimize import least_squares
from astropy.table import Table
from astropy.units import Unit as u
from argparse import ArgumentParser
from glob import glob
from pathlib import Path
import matplotlib.pyplot as plt

from kl_tools.velocity import VelocityMap
from kl_tools.coordinates import OrientedAngle
from kl_tools.utils import build_map_grid, get_base_dir, make_dir, MidpointNormalize
from kl_tools.kross.tfr import estimate_vtf
from kl_tools.kross.kross_utils import theta2pars, pars2theta

def parse_args():
    parser = ArgumentParser()

    parser.add_argument('-s', '--show', action='store_true', default=False,
                        help='Show plots')
    parser.add_argument('-o', '--overwrite', action='store_true', default=False,
                        help='Overwrite existing files')
    parser.add_argument('-v', '--verbose', action='store_true', default=False)
    parser.add_argument(
        '-m', '--max_nfev', type=int, default=None,
        help='Maximum number of function evaluations for the optimizer'
        )


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
              method='lm', loss='linear', max_nfev=1000):
    Nrow, Ncol = data.shape
    Nx, Ny = Ncol, Nrow
    X, Y = build_map_grid(Nx, Ny, indexing='xy')

    return least_squares(
        residuals,
        initial_guess,
        args=(data, weights, mask, X, Y),
        method=method,
        bounds=bounds,
        loss=loss,
        max_nfev=max_nfev
        )

def get_sigma_map(vmap_file, sig_files):

    vmap_name = Path(vmap_file).name.split('.')[0]

    for sig_file in sig_files:
        file_name = Path(sig_file).name
        if vmap_name == file_name.split('.')[0]:
            return sig_file
    
    return None

def plot(show, save, out_file=None):
    '''
    helper function to streamline plotting options
    '''

    if save is True:
        if out_file is None:
            raise ValueError('Must provide out_file if save is True')
        plt.savefig(out_file)
    if show is True:
        plt.show()
    else:
        plt.close()

    return

#---------------------------------------------------------------------------

def main():

    #---------------------------------------------------------------------------
    # initial setup

    args = parse_args()

    show = args.show
    overwrite = args.overwrite
    max_fnev = args.max_nfev
    vb = args.verbose
    show = args.show
    save = True

    data_dir = get_base_dir() / 'data/kross'

    out_dir = get_base_dir() / 'kl_tools/kross/vmap_fits'
    pkl_dir = out_dir / 'pkl'
    plot_dir = out_dir / 'plots'
    make_dir(out_dir)
    make_dir(pkl_dir)
    make_dir(plot_dir)

    out_file = out_dir / 'vmap_fits.fits'
    if out_file.exists() and (overwrite is False):
        raise FileExistsError(
            f'{out_file} already exists and overwrite is False'
            )

    # first, vmap data
    vmap_dir = data_dir / 'vmaps'
    vmap_files = glob(str(vmap_dir / '*.fits'))

    # next, dispersion data for weights
    sig_dir = data_dir / 'disp'
    sig_files = glob(str(sig_dir / '*.fits'))

    # kross table for cross-checks
    kross_dir = data_dir
    kross_file = kross_dir / 'kross_release_v2.fits'
    kross = Table.read(kross_file)

    # kross names need a bit of parsing
    kross_names = kross['NAME']
    for i, name in enumerate(kross_names):
        kross_names[i] = name.strip()

    # will eventually become a numpy record array
    fits = []

    #---------------------------------------------------------------------------
    # Loop over KROSS vmaps and fit the velocity model

    sci_ext = 0
    sig_ext = 0
    for i, vmap_file in enumerate(vmap_files):
        if vb is True:
            print(i)

        #-----------------------------------------------------------------------
        # grab the corresponding kross obj & useful quantities
        
        name = Path(vmap_file).name.split('.')[0]

        if name not in kross_names:
            print(f'{name} not in KROSS table')
            print('Skipping')
            continue

        obj_i = kross_names == name
        obj = kross[obj_i]

        kid = obj['KID'][0] # unique identifier
        vel_pa = obj['VEL_PA'][0] # position angle of the measured vmap
        theta_im = obj['THETA_IM'][0] # inclination angle of the galaxy
        v22_obs = obj['V22_OBS'][0] # observed velocity at 2.2 x Rd
        mstar = obj['MASS'][0]
        log_mstar = np.log10(mstar)

        # derived from KROSS
        sini_kross = np.sin(np.deg2rad(theta_im))

        # we'll use the KROSS PA to set the initial guess for theta_int
        vel_pa = OrientedAngle(vel_pa, unit='deg', orientation='east-of-north')

        # if kid != 171:
            # import ipdb; ipdb.set_trace()
            # continue

        #-----------------------------------------------------------------------
        # load vmap and setup weights/mask

        vmap = fitsio.read(vmap_file, ext=sci_ext)
        mask = np.ones(vmap.shape, dtype=bool)
        mask[vmap == 0.0] = 0

        # setup pixel weights
        weights = np.ones(vmap.shape)
        weights[~mask] = 0.

        # NOTE: This is not really correct, as the velocity dispersion is not
        # the same as the velocity meas error. Keeping for posterity in case we
        # are able to access spatial velocity error maps
        # sig_file = get_sigma_map(vmap_file, sig_files)
        # if sig_file is not None:
        #     sig = fitsio.read(sig_file, ext=sig_ext)
        #     sig_mask = mask & (sig > 0)
        #     weights[sig_mask] = 1. / sig[sig_mask]**2

        #     # handle the cases where sig is zero or negative
        #     # for now, we'll just assign a low value
        #     # NOTE: Tried median, seemed less stable
        #     weight_guess = np.min(weights[sig_mask])
        #     weights[(sig <= 0) & mask] = weight_guess

        #-----------------------------------------------------------------------
        # to set vcirc bounds, we need to use TF relation

        vtf, v_bounds = estimate_vtf(log_mstar, return_error_bounds=True)
        vlow, vhigh = vtf - v_bounds[0], vtf + v_bounds[1]
        if vb is True:
            print(f'vtf: {vtf:.2f}')
            print(f'vlow: {vlow:.2f}')
            print(f'vhigh: {vhigh:.2f}')
            print('---------')

        #-----------------------------------------------------------------------
        # setup fitter bounds and initial guess

        sig_vtf = 2
        vtf_scatter_dex = sig_vtf * 0.05
        vtf_fator = 10**vtf_scatter_dex
        vcirc_base = vtf * sini_kross
        vcirc_low = vcirc_base / vtf_fator
        vcirc_high = vcirc_base * vtf_fator
        bounds_pair = [
            (-50, 50), # v0
            (vcirc_low, vcirc_high), # vcirc
            (0, 20), # rscale
            (0, 1), # sini
            (0, 2*np.pi), # theta_int
            (-0.000001, 0.000001), # g1
            (-0.000001, 0.000001), # g2
            (-25, 25), # x0
            (-25, 25) # y0
        ]

        # initial guess for the optimizer
        initial_guess = pars2theta({
            'v0': 0.0,
            'vcirc': vtf * sini_kross,
            'rscale': 5.0,
            'sini': sini_kross,
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

        #-----------------------------------------------------------------------
        # setup and run the fitter

        method = 'trf' # lm doesn't support bounds
        loss = 'soft_l1' # more robust to outliers
        # loss = 'linear'
        try:
            result = fit_model(
                vmap, weights, mask, initial_guess, bounds=bounds,
                method=method, loss=loss, max_nfev=max_fnev
                )
        except Exception as e:
            print(f'Fitting failed for {name}; KID {kid}')
            print(f'Error: {e}')
            import ipdb; ipdb.set_trace()
            print('Skipping')
            continue

        # import ipdb; ipdb.set_trace()

        fit_theta = result.x
        fit_pars = theta2pars(fit_theta)
        chi2 = result.cost / (np.sum(mask) - len(fit_theta))
        if vb is True:
            for key, val in fit_pars.items():
                if isinstance(val, float):
                    print(f'{key}: {val:.2f}')
                else:
                    print(f'{key}: {val}')

        #-----------------------------------------------------------------------
        # make model fit vmap

        Nrow, Ncol = vmap.shape
        Nx, Ny = Ncol, Nrow
        X, Y = build_map_grid(Nx, Ny, indexing='xy')
        fit = VelocityMap('offset', fit_pars)('obs', X, Y)
        fit[~mask] = 0

        #-----------------------------------------------------------------------
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

        plt.text(
            0.025,
            0.16,
            f'vTF: {vtf:.2f} km/s',
            color='k',
            transform=plt.gca().transAxes
        )
        vcirc = fit_pars['vcirc']
        plt.text(
            0.025,
            0.11,
            f'vcirc: {vcirc:.2f} km/s',
            color='k',
            transform=plt.gca().transAxes
        )
        vcirc = fit_pars['vcirc']
        plt.text(
            0.025,
            0.06,
            f'vcirc/vTF: {vcirc/vtf:.2f}',
            color='k',
            transform=plt.gca().transAxes
        )
        plt.text(
            0.025,
            0.01,
            f'kross sini: {sini_kross:.2f}',
            color='k',
            transform=plt.gca().transAxes
        )

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

        plot(show, save, plot_dir / f'compare_vmap_fit_{name}.png')

        if show is True:
            plt.show()
        else:
            plt.close()

        # get some extra cols
        success = result.success
        status = result.status
        message = result.message
        fits.append(tuple(
            [kid] + list(fit_theta) + [chi2, name, success, status, message]
            ))

        # pickle the results for later
        with open(str(pkl_dir / f'{name}.pkl'), 'wb') as f:
            pickle.dump(result, f)

    # setup & write the output table
    out_cols = [
        'kid',
        'v0',
        'vcirc',
        'rscale',
        'sini',
        'theta_int',
        'g1',
        'g2',
        'x0',
        'y0',
        'chi2',
        'name',
        'success',
        'status',
        'message'
        ]
    out_dtypes = ['f8' for col in out_cols]
    out_dtypes[0] = 'i4' # kid
    out_dtypes[11] = 'S20' # name
    out_dtypes[12] = 'bool' # success
    out_dtypes[13] = 'i4' # status
    out_dtypes[14] = 'S100' # message

    out_shape = len(fits)
    dtype = np.dtype({'names': out_cols, 'formats': out_dtypes})
    vmap_fits = np.array(fits, dtype=dtype)
    t = Table(vmap_fits)
    t.write(out_file, overwrite=overwrite)

    # fitsio.write(out_file, vmap_fits, clobber=overwrite)

    return

if __name__ == '__main__':
    main()