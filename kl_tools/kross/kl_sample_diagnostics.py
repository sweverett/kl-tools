import numpy as np
import fitsio
from astropy.table import Table
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.units import deg
from pathlib import Path
from glob import glob
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.backends.backend_pdf import PdfPages

from kl_tools.velocity import VelocityMap
from kl_tools.coordinates import OrientedAngle
from kl_tools.utils import make_dir, get_base_dir, MidpointNormalize, build_map_grid
from kl_tools.kross.tfr  import estimate_vtf
from kl_tools.kross.kross_utils import plot_line_on_image

def main():

    #---------------------------------------------------------------------------
    # Script options

    show = False
    save = True
    vb = True

    # fig size
    Sx, Sy = 45, 5

    # whether to subselect the KROSS sample
    subselect = False
    subselect_frac = 0.5

    # whether to use only the COSMOS objects
    use_cosmos_only = False

    # whether to include sample inclusion & outlier information
    show_sample_info = False

    # KL shape noise
    kl_g_err = 0.03

    #---------------------------------------------------------------------------
    # General IO

    data_dir = get_base_dir() / 'data'
    out_dir = get_base_dir() / 'kl_tools/kross/plots/diagnostics'
    if subselect is True:
        out_dir = out_dir / f'subselect_{subselect_frac:.2f}'
    make_dir(out_dir)

    #---------------------------------------------------------------------------
    # KROSS data loading & parsing

    kross_dir = data_dir / 'kross'
    kross_file = kross_dir / 'kross_release_v2.fits'
    kross = Table.read(kross_file)
    names = kross['NAME']

    kross_ra = kross['RA']
    kross_dec = kross['DEC']
    kross_coords = SkyCoord(ra=kross_ra*deg, dec=kross_dec*deg)

    # KROSS datacubes
    cube_files = glob(str(kross_dir / 'cubes/*.fits'))

    for i, name in enumerate(names):
        new = name.strip()
        names[i] = new
    
    # KROSS datacubes
    cube_files = glob(str(kross_dir / 'cubes/*.fits'))

    # KROSS Halpha intensity maps
    imap_files = glob(str(kross_dir / 'halpha/*.fits'))
    kross_imaps = {}
    for imap in imap_files:
        fname = Path(imap).name
        name = fname.split('.fits')[0]
        kross_imaps[name] = imap

    # KROSS velocity maps
    vmap_files = glob(str(kross_dir / 'vmaps/*.fits'))
    kross_vmaps = {}
    for vmap in vmap_files:
        fname = Path(vmap).name
        name = fname.split('.fits')[0]
        kross_vmaps[name] = vmap

    # Model fit velocity maps
    model_vmap_dir = get_base_dir() / 'kl_tools/kross/vmap_fits/'
    model_vmaps = Table.read(str(model_vmap_dir / 'vmap_fits.fits'))

    # Model fit rotation curves
    model_rc_dir = get_base_dir() / 'kl_tools/kross/rotation_curves/'
    model_rcs = Table.read(str(model_rc_dir / 'rotation_curves.fits'))

    #---------------------------------------------------------------------------
    # COSMOS data loading & parsing

    cosmo_dir = data_dir / 'cosmos'
    cosmo_files = glob(str(cosmo_dir / 'cutouts/*.fits'))

    cosmo_cutouts = {}
    for f in cosmo_files:
        fname = Path(f).name

        # match to the KROSS sky coordinates to get the obj name
        ra  = float(fname.split('_')[1])
        dec = float(fname.split('_')[2])
        coord = SkyCoord(ra=ra*deg, dec=dec*deg)

        indx, sep, _ = coord.match_to_catalog_sky(kross_coords)
        cosmo_cutouts[names[indx]] = f

    #---------------------------------------------------------------------------
    # Shear estimates

    sample_dir = get_base_dir() / 'kl_tools/kross/sample'
    estimates = Table.read(sample_dir / 'shear_estimates.fits')
    estimates_names = estimates['name']

    for i, name in enumerate(estimates_names):
        new = name.strip()
        estimates_names[i] = new

    #---------------------------------------------------------------------------

    halpha_true = 6562.8 # A

    #---------------------------------------------------------------------------

    Ncubes = len(cube_files)
    print(f'Processing {Ncubes} datacube files')
    for cube_file in cube_files:
        if subselect is True:
            if np.random.rand() > subselect_frac:
                continue

        name = Path(cube_file).name.split('.fits')[0]

        if vb is True:
            print(f'Processing {name}')

        # not all KROSS objects are in the COSMOS field
        if (use_cosmos_only) and (not name in cosmo_cutouts):
            print(f'{name} not in COSMOS field')
            continue
        
        # useful quantities

        try:
            obj = kross[names == name][0]
        except IndexError:
            print('fNo KROSS object found for {name}; skipping')
            continue

        kid = obj['KID']
        z = obj['Z']
        pa_im = OrientedAngle(
            obj['PA_IM'], unit='deg', orientation='east-of-north'
            )
        pa_k = OrientedAngle(
            obj['VEL_PA'], unit='deg', orientation='east-of-north'
            )
        theta_im = obj['THETA_IM']
        sini = np.sin(np.deg2rad(theta_im)) 
        ktype = obj['KIN_TYPE'].strip()
        log_mstar = np.log10(obj['MASS'])
        vtf = estimate_vtf(log_mstar)
        halpha_obs = halpha_true * (1 + z)

        # let's remap the angles to be defined from the x-axis instead of east of north
        # pa_im = 90 + pa_im
        # pa_k = 90 + pa_k
        pa_im = pa_im.to_orientation('cartesian')
        pa_k = pa_k.to_orientation('cartesian')

        cube = fitsio.read(cube_file)
        wcs = WCS(cube_file).dropaxis(2)
        stack = np.sum(cube, axis=0)

        if name in estimates_names:
            shear = estimates[estimates['name'] == name]
            g1_kross = shear['gplus']
            g1_cosmos = shear['cosmos_gplus']
            g1_cosmos_err = shear['cosmos_gplus_err']
            in_sample = True

            # if name == 'C-zcos_z1_925':
            #     x = estimates['gplus']
            #     y = estimates['cosmos_gplus']
            #     yerr = estimates['cosmos_gplus_err']
            #     plt.errorbar(x, y, yerr, ls='none')
            #     plt.plot([-0.1, 0.1], [-0.1, 0.1], color='k', ls='--')
            #     plt.fill_between(
            #         [-0.1, 0.1],
            #         [-0.1-0.03, 0.1-0.03],
            #         [-0.1+0.03, 0.1+0.03],
            #         color='gray',
            #     )
            #     plt.xlim(-0.1, 0.1)
            #     plt.ylim(-0.1, 0.1)
            #     plt.xlabel('KROSS g+')
            #     plt.ylabel('COSMOS g+')
            #     plt.scatter(g1_kross, g1_cosmos, color='red')
            #     plt.show()

            # g1_resid = (g1_kross - g1_cosmos) / g1_cosmos_err
            g1_resid = (g1_kross - g1_cosmos) / kl_g_err
            if np.abs(g1_resid) > 3:
                outlier = True
            else:
                outlier = False
        else:
            in_sample = False

        fig = plt.figure(figsize=(Sx, Sy))
        Nsubplots = 6

        #-----------------------------------------------------------------------
        # plot the Hubble image

        # we don't currently have hubble cutouts for all sources
        try:
            ax = fig.add_subplot(1, Nsubplots, 1, projection=wcs)
            cutout_file = cosmo_cutouts[name]
            cutout = fitsio.read(cutout_file)
            wcs = WCS(cutout_file)
            ax.imshow(cutout, origin='lower', cmap='gray_r')
            i, j = np.unravel_index(np.argmax(cutout), cutout.shape)
            cen = [i - cutout.shape[0]//2, j - cutout.shape[1]//2]
            # cen = [
            #     np.argmax(cutout)[0]-cutout.shape[0],
            #     np.argmax(cutout)[1]-cutout.shape[1]
            #     ]
            plot_line_on_image(cutout, cen, pa_im, ax, 'orange', label='photometric axis')
            ax.set_title('Hubble Image')
            ax.text(0.025, 0.9, name, color='k', transform=ax.transAxes, fontsize=12)
            ax.text(0.025, 0.85, f'kid: {kid}', color='k', transform=ax.transAxes, fontsize=12)
            ax.text(0.025, 0.8, f'z={z:.3f}', color='k', transform=ax.transAxes, fontsize=12)
            ax.text(0.025, 0.75, f'sini={sini:.2f}', color='k', transform=ax.transAxes, fontsize=12)
            ax.text(0.025, 0.7, f'pa_diff={pa_im-pa_k:.1f} deg', color='k', transform=ax.transAxes, fontsize=12)
            ax.text(0.025, 0.65, f'KIN_TYPE={ktype}', color='k', transform=ax.transAxes, fontsize=12)
            ax.text(0.025, 0.6, f'mstar=10^{log_mstar:.2f}', color='k', transform=ax.transAxes, fontsize=12)
            ax.text(0.025, 0.55, f'vTF={vtf:.1f}', color='k', transform=ax.transAxes, fontsize=12)
            # ax.text(0.025, 0.7, f'pa_im={pa_im:.1f} deg', color='k', transform=ax.transAxes, fontsize=12)
            # ax.text(0.025, 0.65, f'pa_k={pa_k:.1f} deg', color='k', transform=ax.transAxes, fontsize=12)
            if show_sample_info is True:
                ax.text(0.025, 0.10, f'In sample: {in_sample}', color='k', transform=ax.transAxes, fontsize=12)
            if (in_sample is True) and (show_sample_info is True):
                ax.text(0.025, 0.05, f'3-sig outlier: {outlier}', color='k', transform=ax.transAxes, fontsize=12)
            ax.coords[0].set_axislabel('Right Ascension (J2000)', fontsize=10)
            ax.coords[1].set_axislabel('Declination (J2000)', fontsize=10)
            plt.legend(loc='upper right')

            #-------------------------------------------------------------------
            # plot the Hubble image, rebinned for higher s2n

            # average the cutout image in 4x4 bins
            new_cutout = np.zeros((cutout.shape[0]//4, cutout.shape[1]//4))
            for i in range(new_cutout.shape[0]):
                for j in range(new_cutout.shape[1]):
                    new_cutout[i, j] = np.mean(cutout[4*i:4*i+4, 4*j:4*j+4])

            ax = fig.add_subplot(1, Nsubplots, 2, projection=wcs)
            wcs = WCS(cutout_file)
            ax.imshow(new_cutout, origin='lower', cmap='gray_r')
            i, j = np.unravel_index(np.argmax(new_cutout), new_cutout.shape)
            cen = [i - new_cutout.shape[0]//2, j - new_cutout.shape[1]//2]
            plot_line_on_image(new_cutout, cen, pa_im, ax, 'orange', label='photometric axis')
            ax.set_title('Hubble Image (Rebinned 4x4)')
            ax.coords[0].set_axislabel('Right Ascension (J2000)', fontsize=10)
            ax.coords[1].set_axislabel('Declination (J2000)', fontsize=10)
            plt.legend(loc='upper right')

        except:
            print(f'No Hubble cutout for {name}; skipping')

        #-----------------------------------------------------------------------
        # plot the H-alpha image

        ax = fig.add_subplot(1, Nsubplots, 3, projection=wcs)

        # corrupted files...
        try:
            imap_file = kross_imaps[name]
            imap = fitsio.read(imap_file)
            wcs = WCS(imap_file)
            vmin = np.percentile(imap, 1)
            vmax = np.percentile(imap, 99)
            im = ax.imshow(imap, origin='lower', vmin=vmin, vmax=vmax, cmap='gray_r')
            ax.set_title('H-alpha Intensity Map')
            ax.coords[0].set_axislabel('Right Ascension (J2000)', fontsize=10)
            ax.coords[1].set_axislabel('Declination (J2000)', fontsize=10)
            plt.colorbar(im, fraction=0.046, pad=0.04, label='erg/s/cm^2')
            # ax.coords[0].set_ticklabel_visible(False)
            # ax.coords[1].set_ticklabel_visible(False)
        except OSError:
            pass

        #-----------------------------------------------------------------------
        # plot the vmap image

        ax = fig.add_subplot(1, Nsubplots, 4, projection=wcs)
        vmap_file = kross_vmaps[name]
        vmap = fitsio.read(vmap_file)
        wcs = WCS(vmap_file)
        norm = MidpointNormalize(
            midpoint=0,
            vmin=np.percentile(vmap, 1),
            vmax=np.percentile(vmap, 99)
            )
        im = ax.imshow(vmap, origin='lower', norm=norm, cmap='RdBu')
        ax.set_title('Velocity Map')
        cen = [0,0]
        plot_line_on_image(
            vmap, cen, pa_im, ax, 'orange', label='Hubble photometric axis'
            )
        plot_line_on_image(
            vmap, cen, pa_k, ax, c='k', label='KROSS kinematic axis'
            )
        ax.text(0.05, 0.85, f'KIN_TYPE={ktype}', color='k', transform=ax.transAxes, fontsize=12)
        ax.coords[0].set_axislabel('Right Ascension (J2000)', fontsize=10)
        ax.coords[1].set_axislabel('Declination (J2000)', fontsize=10)
        # divider = make_axes_locatable(ax)
        # cax = divider.append_axes("right", size="5%", pad=0.05)
        # plt.colorbar(im, cax=cax)
        plt.colorbar(im, fraction=0.046, pad=0.04, label='km/s')
        plt.legend(loc='upper right')

        #-----------------------------------------------------------------------
        # plot the fitted velocity map

        try:
            ax = fig.add_subplot(1, Nsubplots, 5, projection=wcs)
            mask = vmap == 0
            Nrow, Ncol = vmap.shape
            Nx, Ny = Ncol, Nrow
            X, Y = build_map_grid(Nx, Ny, indexing='xy')
            vmap_pars = model_vmaps[model_vmaps['name'] == name]
            for col in ['chi2', 'name']: vmap_pars.remove_column(col)
            vmap_pars = dict(vmap_pars[0])
            vmap_pars['r_unit'] = 'arcsec'
            vmap_pars['v_unit'] = 'km/s'
            model_vmap = VelocityMap('offset', vmap_pars)('obs', X, Y)
            model_vmap[mask] = 0
            norm = MidpointNormalize(
                midpoint=0,
                vmin=np.percentile(model_vmap, 1),
                vmax=np.percentile(model_vmap, 99)
                )
            im = ax.imshow(model_vmap, origin='lower', norm=norm, cmap='RdBu')
            ax.set_title('Model Velocity Map')
            cen = [vmap_pars['x0'], vmap_pars['y0']]
            plot_line_on_image(
                model_vmap, [0,0], pa_im, ax, 'orange', label='Hubble photometric axis'
                )
            plot_line_on_image(
                model_vmap, [0,0], pa_k, ax, 'k', label='KROSS kinematic axis'
                )
            pa_k_fitted = OrientedAngle(
                vmap_pars['theta_int'], unit='rad', orientation='cartesian'
                )
            plot_line_on_image(
                model_vmap, cen, pa_k_fitted, ax, 'red', label='Fitted kinematic axis (offset)'
                )
            ax.text(0.05, 0.85, f'KIN_TYPE={ktype}', color='k', transform=ax.transAxes, fontsize=12)
            ax.coords[0].set_axislabel('Right Ascension (J2000)', fontsize=10)
            ax.coords[1].set_axislabel('Declination (J2000)', fontsize=10)
            # divider = make_axes_locatable(ax)
            # cax = divider.append_axes("right", size="5%", pad=0.05)
            # plt.colorbar(im, cax=cax)
            plt.colorbar(im, fraction=0.046, pad=0.04, label='km/s')
            plt.legend(loc='upper right')
        except:
            pass

        #-----------------------------------------------------------------------
        # plot the measured & fitted rotation curves

        try:
            ax = fig.add_subplot(1, Nsubplots, 6, projection=wcs)
            rc = model_rcs[model_rcs['name'] == name]
            bin_centers = rc['distance'][0]
            obs_rotation_curve = rc['obs_rotation_curve'][0]
            obs_rotation_curve_err = rc['obs_rotation_curve_err'][0]
            model_rotation_curve = rc['model_rotation_curve'][0]
            ax.errorbar(
                bin_centers, obs_rotation_curve, obs_rotation_curve_err, marker='o', label='observed'
                        )
            ax.plot(
                bin_centers, model_rotation_curve, ls='--', c='k', label='model'
                        )
            ax.set_xlabel('Radial Distance (pixels)')
            ax.set_ylabel('3D Rotational Velocity (km/s)')
            plt.legend()
            ax.set_title(f'{name} Galaxy Rotation Curve')
            ax.set_aspect('auto')
        except:
            pass

        #-----------------------------------------------------------------------
        # save and possibly show the plot

        # plt.tight_layout()

        if save is True:
            out_file = out_dir / f'{name}-diag.png'
            plt.savefig(out_file, dpi=300)

        if show is True:
            plt.show()
        else:
            plt.close()

    #---------------------------------------------------------------------------
    # Now collate into one PDF

    pdf_file = out_dir / 'kross-cosmos-diagnostics.pdf'
    png_files = glob(str(out_dir / '*.png'))

    with PdfPages(pdf_file) as pdf:
        for png_file in png_files:
            fig = plt.figure(figsize=(Sx, Sy))
            ax = fig.add_subplot(111)
            ax.imshow(plt.imread(png_file))
            ax.axis('off')
            pdf.savefig(fig)
            plt.close()

    return

if __name__ == '__main__':
    main()
