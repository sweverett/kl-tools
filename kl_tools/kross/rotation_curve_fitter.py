import numpy as np
import fitsio
from scipy.optimize import least_squares
from astropy.table import Table
from astropy.units import Unit as u
from argparse import ArgumentParser
from glob import glob
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from kl_tools.velocity import VelocityMap, dist_to_major_axis
from kl_tools.coordinates import OrientedAngle
from kl_tools.utils import build_map_grid, get_base_dir, make_dir, MidpointNormalize, plot
from kl_tools.kross.kross_utils import theta2pars
from kl_tools.plotting import plot_line_on_image

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-s', '--show', action='store_true', default=False,
                        help='Show plots')
    parser.add_argument('-o', '--overwrite', action='store_true', default=False,
                        help='Overwrite existing files')
    parser.add_argument('-v', '--verbose', action='store_true', default=False)

    return parser.parse_args()

def rotate_coordinates(X, Y, x0, y0, theta):
    '''
    Use to transform the coordinates of a 2D array to a new coordinate system
    defined by the center (x0, y0) and the rotation angle theta
    '''
    # Translate coordinates to the center
    X_trans = X - x0
    Y_trans = Y - y0

    # Apply rotation
    X_rot = X_trans * np.cos(theta) - Y_trans * np.sin(theta)
    Y_rot = X_trans * np.sin(theta) + Y_trans * np.cos(theta)
    
    return X_rot, Y_rot

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
    kid = kross['KID']
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

        # TODO: cleanup
        # if not (name == 'U-HiZ_z1_103'):
        #     continue

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

        # when binning across radial distance from the galaxy center, we'll
        # want to do it in a rotated coordinate system such that the kinematic
        # axis is aligned with the x-axis to avoid issues with steep gradients
        # relative to the pixelization grid
        # X_rot, Y_rot = rotate_coordinates(X, Y, x0, y0, model_pa.cartesian.rad)

        # bin the velocity map by radius from the galaxy center
        Nrbins = 10
        R = np.sqrt((X-x0)**2 + (Y-y0)**2)
        # R = np.sqrt((X_rot)**2 + (Y_rot)**2)
        # we add a sign to the distance to the major axis to account for which 
        # "side" of the center we are on
        # R = R * np.sign(X_rot)
        dx = np.cos(model_pa)
        dy = np.sin(model_pa)
        R_signed = (X - x0) * dx + (Y - y0) * dy

        # now find the pixels within the threshold distance of the major axis
        # dist_major = np.abs(Y_rot)
        R_in_major = R_signed[(dist_major <= threshold_dist) & mask]

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

        # TODO: cleanup
        # plt.subplot(141)
        # plt.imshow(vmap, origin='lower', cmap='RdBu', vmin=-150, vmax=150)
        # plt.colorbar()
        # plt.title('Observed Velocity Map')
        # plt.subplot(142)
        # mm = model_vmap.copy()
        # mm[~mask] = 0
        # plt.imshow(mm, origin='lower', cmap='RdBu', vmin=-150, vmax=150)
        # plt.colorbar()
        # plt.title('Model Velocity Map')
        # plt.subplot(143)
        # plt.imshow(dist_major, origin='lower')
        # plt.colorbar()
        # plt.title('Distance to Major Axis')
        # plt.subplot(144)
        # plt.imshow(R_signed, origin='lower')
        # plt.colorbar()
        # plt.title('Radial Signed Distance')
        # plt.gcf().set_size_inches(18, 5)
        # plt.show()

        # Calculate average velocity for each bin
        for n in range(Nrbins):
            radial_mask = (R_signed >= rbins[n]) & (R_signed < rbins[n+1])
            # radial_mask_rot = (X_rot >= rbins[n]) & (X_rot < rbins[n+1])

            # now transform the radial mask back to the original coords
            # rot_sel = np.where(radial_mask_rot)
            # X_rot_sel, Y_rot_sel = X_rot[rot_sel], Y_rot[rot_sel]

            # X_sel, Y_sel = rotate_coordinates(
            #     X_rot_sel, Y_rot_sel, -x0, -y0, -model_pa.cartesian.rad
            #     )

            # radial_mask = np.zeros_like(vmap, dtype=bool)
            # import ipdb; ipdb.set_trace()
            # for x, y in zip(X_sel, Y_sel):
            #     ix = int(np.round(x - X.min()))
            #     iy = int(np.round(y - Y.min()))

            #     if 0 <= ix < X.shape[1] and 0 <= iy < X.shape[0]:
            #         radial_mask[ix, iy] = True

            # now apply the other mask
            dist_major_mask = dist_major <= threshold_dist

            # combine the masks
            combined_mask = radial_mask & dist_major_mask
            # import ipdb; ipdb.set_trace()

            # TODO: cleanup
            # if 0 < n < 9:
            #     count += 1
            #     plt.subplot(1,8,count)
            #     mm = model_vmap.copy()
            #     mm[~combined_mask] = 0
            #     im = plt.imshow(
            #         mm, origin='lower', cmap='RdBu', vmin=-150, vmax=150
            #         )
            #     plt.title(f'Bin {count+1}')
            #     divider = make_axes_locatable(plt.gca())
            #     cax = divider.append_axes("right", size="5%", pad=0.05)
            #     plt.colorbar(im, cax=cax)

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

        # TODO: cleanup
        # plt.gcf().set_size_inches(20, 4)
        # plt.suptitle(f'Vmap radial bins (+/-{threshold_dist} pix threshold)')
        # plt.show()

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