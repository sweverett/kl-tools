import numpy as np
import fitsio
from scipy.optimize import least_squares
from astropy.table import Table
from astropy.units import Unit as u
from argparse import ArgumentParser
from glob import glob
from pathlib import Path
import matplotlib.pyplot as plt

from kl_tools.velocity import VelocityMap
from kl_tools.coordinates import OrientedAngle
from kl_tools.utils import build_map_grid, get_base_dir, make_dir, MidpointNormalize, plot
from kl_tools.kross.kross_utils import plot_line_on_image, theta2pars

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-s', '--show', action='store_true', default=False,
                        help='Show plots')
    parser.add_argument('-o', '--overwrite', action='store_true', default=False,
                        help='Overwrite existing files')
    parser.add_argument('-v', '--verbose', action='store_true', default=False)

    return parser.parse_args()

def dist_to_major_axis(X, Y, x0, y0, position_angle):
    '''
    Calculate the distance from each pixel to the major axis

    Parameters
    ----------
    X : np.ndarray
        2D array of x-coordinates
    Y : np.ndarray
        2D array of y-coordinates
    x/y0 : float
        x/y-coordinate of the galaxy center
    position_angle : OrientedAngle
        Position angle of the galaxy major axis
    '''

    # get distance from each pixel to the galaxy center
    Xcen, Ycen = X - x0, Y - y0
    R = np.sqrt(Xcen**2 + Ycen**2)

    # angle from each pixel to the galaxy center
    Theta = np.arctan2(Ycen, Xcen) # returns (-pi, pi)

    # match the [0, 360] wrapping of the position angle
    Theta[Theta < 0] += 2 * np.pi 

    # difference in angle between the pixel and the major axis
    dTheta = position_angle.cartesian.rad - Theta

    # calculate distance from each pixel to the major axis
    dist = abs(R * np.sin(dTheta))

    # import matplotlib.colors as mcolors
    # norm = mcolors.TwoSlopeNorm(
    #     vmin=-np.max(np.abs(dist)), vcenter=0, vmax=np.max(np.abs(dist))
    #    )
    # plt.imshow(dist.T, origin='lower', cmap='RdBu', norm=norm)
    # plt.text(0.1, 0.1, f'PA: {np.rad2deg(position_angle):.2f} deg')
    # plt.colorbar()
    # plt.show()

    return dist

def main():

    #---------------------------------------------------------------------------
    # initial setup

    args = parse_args()

    show = args.show
    overwrite = args.overwrite 
    vb = args.verbose
    save = True

    out_dir = get_base_dir() / 'kl_tools/kross/rotation_curves'
    plot_dir = out_dir / 'plots'
    make_dir(out_dir)
    make_dir(plot_dir)

    out_table = out_dir / 'rotation_curves.fits'
    if out_table.exists() and (overwrite is False):
        raise FileExistsError(
            f'{out_table} already exists; set overwrite=True to overwrite'
            )

    # TODO: eventually make this a physical unit
    threshold_dist = 1.5 # pixels from the major axis

    #---------------------------------------------------------------------------
    # grab the kross table, vmap data, and velocity fits

    kross_dir = get_base_dir() / 'data/kross'
    kross_file = kross_dir / 'kross_release_v2.fits'
    kross = Table.read(kross_file)
    kross_names = kross['NAME']
    for i, name in enumerate(kross_names):
        kross_names[i] = name.strip()

    vmap_dir = get_base_dir() / 'data/kross/vmaps'
    vmap_files = glob(str(vmap_dir / '*.fits'))

    models_file = get_base_dir() / 'kl_tools/kross/vmap_fits/vmap_fits.fits'

    try:
        models = Table.read(models_file)
    except FileNotFoundError:
        print('No models file found; have you run velocity_map_fitter.py?')

    #---------------------------------------------------------------------------
    # loop over the vmap files and plot the rotation curves

    names_list = []
    bin_centers_list = []
    obs_rotation_curve_list = []
    obs_rotation_curve_err_list = []
    model_rotation_curve_list = []
    for i, vmap_file in enumerate(vmap_files):

        # find the corresponding KROSS source
        name = Path(vmap_file).name.split('.')[0]

        if vb is True:
            print(f'Processing {name} ({i+1}/{len(vmap_files)})')

        obj = kross[kross_names == name]

        if len(obj) == 0:
            print(f'No KROSS data found for {name}; skipping')
            continue

        kross_pa = obj['VEL_PA'][0] # position angle of the measured vmap
        kross_pa = OrientedAngle(
            kross_pa, unit='deg', orientation='east-of-north'
            )

        # grab the KROSS velocity map and fitted model
        vmap, vmap_hdr = fitsio.read(vmap_file, header=True)
        mask = np.ones(vmap.shape, dtype=bool)
        mask[vmap == 0] = False
        try:
            model = models[models['name'] == name][0]
        except (ValueError, IndexError):
            # a few don't have fitted models
            print(f'No model found for {name}; skipping')
            continue

        # grab only the components in the vmap model def
        model_pars = theta2pars(model[0:-2])

        # build the model of the observed velocity map
        model_vmap_buidler = VelocityMap('offset', model_pars)
        Nrow, Ncol = vmap.shape
        Nx, Ny = Ncol, Nrow
        X, Y = build_map_grid(Nx, Ny, indexing='xy')
        model_vmap = model_vmap_buidler('obs', X, Y)

        # determine the distance from the major axis for all pixels
        x0, y0 = model['x0'], model['y0']
        model_pa = OrientedAngle(
            model['theta_int'], unit='rad', orientation='cartesian'
        )
        dist_major = dist_to_major_axis(X, Y, x0, y0, model_pa)

        # bin the velocity map by radius from the galaxy center
        Nrbins = 10
        R = np.sqrt((X-x0)**2 + (Y-y0)**2)
        # we add a sign to the distance to the major axis to account for which 
        # "side" of the center we are on
        R = R * np.sign(X - x0)
        R_in_major = R[(dist_major <= threshold_dist) & mask]
        if len(R_in_major) == 0:
            print(
                f'No pixels within {threshold_dist} of the major axis; '
                'skipping'
                )
            continue
        Rmin = R_in_major.min()
        Rmax = R_in_major.max()
        rbins = np.linspace(Rmin, Rmax, Nrbins+1)

        bin_centers = (rbins[:-1] + rbins[1:]) / 2
        obs_rotation_curve = []
        obs_rotation_curve_err = []
        model_rotation_curve = []

        # Calculate average velocity for each bin
        for n in range(Nrbins):
            radial_mask = (R >= rbins[n]) & (R < rbins[n+1])
            dist_major_mask = dist_major <= threshold_dist
            combined_mask = radial_mask & dist_major_mask

            if np.any(combined_mask):
                sini = model_pars['sini']
                avg_obs_vel = np.mean(vmap[combined_mask] / sini)
                std_obs_vel = np.std(vmap[combined_mask] / sini)
                avg_model_vel = np.mean(model_vmap[combined_mask] / sini)
                obs_rotation_curve.append(avg_obs_vel)
                obs_rotation_curve_err.append(std_obs_vel)
                model_rotation_curve.append(avg_model_vel)
            else:
                # handle empty bins
                obs_rotation_curve.append(np.nan)
                obs_rotation_curve_err.append(np.nan)
                model_rotation_curve.append(np.nan)  

        plt.subplot(131)
        vmin, vmax = np.percentile(vmap, 1), np.percentile(vmap, 99)
        plt.imshow(
            vmap, origin='lower', cmap='RdBu',
            norm=MidpointNormalize(vmin, vmax)
            )
        plot_line_on_image(
            vmap, (x0, y0), model_pa, plt.gca(), c='k', label='model major axis'
        )
        # plot the threshold region
        plot_line_on_image(
            vmap, (x0, y0+threshold_dist/2.), model_pa, plt.gca(), c='k', 
            ls=':', label='model major axis'
        )
        plot_line_on_image(
            vmap, (x0, y0-threshold_dist/2.), model_pa, plt.gca(), c='k', 
            ls=':', label='model major axis'
        )
        plt.xlim(0, Nx)
        plt.ylim(0, Ny)
        plt.title(f'{name} Observed Velocity Map')

        plt.subplot(132)
        masked_model_vmap = model_vmap.copy()
        masked_model_vmap[~mask] = 0
        vmin = np.percentile(masked_model_vmap, 1)
        vmax = np.percentile(masked_model_vmap, 99)
        plt.imshow(
            masked_model_vmap, origin='lower', cmap='RdBu',
            norm=MidpointNormalize(vmin, vmax)
            )
        plot_line_on_image(
            masked_model_vmap,
            (x0, y0),
            model_pa,
            plt.gca(),
            c='k',
            label='model major axis'
        )
        # plot the threshold region
        plot_line_on_image(
            vmap, (x0, y0+threshold_dist/2.), model_pa, plt.gca(), c='k', 
            ls=':', label='model major axis'
        )
        plot_line_on_image(
            vmap, (x0, y0-threshold_dist/2.), model_pa, plt.gca(), c='k', 
            ls=':', label='model major axis'
        )
        plt.text(
            0.1,
            0.15,
            f'Model PA: {model_pa.east_of_north.deg:.1f} deg',
            transform=plt.gca().transAxes
            )
        plt.text(
            0.1,
            0.1,
            f'KROSS PA: {kross_pa.east_of_north.deg:.1f} deg',
            transform=plt.gca().transAxes
            )
        plt.xlim(0, Nx)
        plt.ylim(0, Ny)
        plt.title(f'{name} Model Velocity Map')

        plt.subplot(133)
        plt.errorbar(
            bin_centers, obs_rotation_curve, obs_rotation_curve_err, marker='o', label='observed'
                    )
        plt.plot(
            bin_centers, model_rotation_curve, ls='--', c='k', label='model'
                    )
        plt.xlabel('Radial Distance (pixels)')
        plt.ylabel('3D Rotational Velocity (km/s)')
        plt.legend()
        plt.title(f'{name} Galaxy Rotation Curve')
        plt.grid(True)

        plt.gcf().set_size_inches(18, 5)
        plt.tight_layout()

        out_file = plot_dir / f'rotation_curve_{name}.png'
        plot(show, save, out_file=out_file)

        # save the data
        names_list.append(name)
        bin_centers_list.append(bin_centers)
        obs_rotation_curve_list.append(obs_rotation_curve)
        obs_rotation_curve_err_list.append(obs_rotation_curve_err)
        model_rotation_curve_list.append(model_rotation_curve)

    rotation_curves = Table(
        data=[
            names_list,
            bin_centers_list,
            obs_rotation_curve_list,
            obs_rotation_curve_err_list,
            model_rotation_curve_list,
            ],
        names=[
            'name',
            'distance',
            'obs_rotation_curve',
            'obs_rotation_curve_err',
            'model_rotation_curve',
            ]
        )

    rotation_curves.write(out_table, overwrite=overwrite)

    return

if __name__ == '__main__':
    main()