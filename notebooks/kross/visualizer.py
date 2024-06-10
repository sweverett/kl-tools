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

from kl_tools.utils import make_dir, get_base_dir, MidpointNormalize

def estimate_vtf(log_mstar, alpha=4.51, log_M100=9.49):
    '''
    Values are for K-band. Logs are base 10
    '''
    log_vtf = (1. / alpha) * (log_mstar - log_M100)
#     print(10**log_vtf)
    return 100*10**log_vtf

def plot_line_on_image(im, angle, ax, c='k', label=None):
    """
    Plots a line on an image at a given angle using WCS projection.
    
    Parameters:
    im : Image data
    angle : Angle in degrees
    ax : Matplotlib Axes object
    c: color
    """

    # angle += 90

    # Get the dimensions of the image
    height, width = im.shape
    
    # Calculate the angle in radians
    angle_rad = np.deg2rad(angle)

    x0, x1 = 0, width - 1

    # y0 = (0 - np.cos(angle_rad) * x0) / np.sin(angle_rad)
    # y0 = np.tan(angle_rad) * (-width / 2.) - height / 2.
    # y1 = np.tan(angle_rad) * (width / 2.)
    # y1 = -y0
    dy = np.tan(angle_rad) * (width / 2.)
    y0 = height / 2. - dy
    y1 = height / 2. + dy

    # Create the line coordinates in pixel space
    line_x = np.array([x0, x1])
    line_y = np.array([y0, y1])
    
    # Transform the coordinates from pixel to world coordinates
    # line_coords = wcs.pixel_to_world(line_x, line_y)
    
    # Draw the line using the world coordinates
    # ax.plot(line_coords.ra.deg, line_coords.dec.deg, transform=ax.get_transform('world'), color='red', linewidth=2)
    ax.plot(
        # line_coords.ra.deg,
        # line_coords.dec.deg,
        line_x,
        line_y,
        color=c,
        linewidth=2,
        ls='--',
        label=label
        # transform=ax.get_transform('world')
        )
    plt.xlim(0, width)
    plt.ylim(0, height)

    return

def main():

    #---------------------------------------------------------------------------
    # Script options

    show = False
    save = True

    # fig size
    Sx, Sy = 26, 5

    # whether to subselect the KROSS sample
    subselect = True
    subselect_frac = 0.5

    # KL shape noise
    kl_g_err = 0.03

    #---------------------------------------------------------------------------
    # General IO

    data_dir = get_base_dir() / 'data'
    out_dir = get_base_dir() / 'notebooks/kross/diag'
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

    estimates = Table.read('./shear_estimates.fits')
    estimates_names = estimates['name']

    for i, name in enumerate(estimates_names):
        new = name.strip()
        estimates_names[i] = new

    #---------------------------------------------------------------------------

    halpha_true = 6562.8 # A

    #---------------------------------------------------------------------------

    for cube_file in cube_files:
        if subselect is True:
            if np.random.rand() > subselect_frac:
                continue

        name = Path(cube_file).name.split('.fits')[0]

        # not all KROSS objects are in the COSMOS field
        if not name in cosmo_cutouts:
            continue
        
        # useful quantities

        obj = kross[names == name][0]
        kid = obj['KID']
        z = obj['Z']
        pa_im = obj['PA_IM']
        pa_k = obj['VEL_PA']
        theta_im = obj['THETA_IM']
        sini = np.sin(np.deg2rad(theta_im)) 
        ktype = obj['KIN_TYPE'].strip()
        log_mstar = np.log10(obj['MASS'])
        vtf = estimate_vtf(log_mstar)
        halpha_obs = halpha_true * (1 + z)

        # let's remap the angles to be defined from the x-axis instead of east of north
        pa_im = 90 + pa_im
        pa_k = 90 + pa_k

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

        #-----------------------------------------------------------------------
        # plot the Hubble image

        ax = fig.add_subplot(1, 4, 1, projection=wcs)
        cutout_file = cosmo_cutouts[name]
        cutout = fitsio.read(cutout_file)
        wcs = WCS(cutout_file)
        ax.imshow(cutout, origin='lower', cmap='gray_r')
        plot_line_on_image(cutout, pa_im, ax, 'orange', label='photometric axis')
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
        ax.text(0.025, 0.10, f'In sample: {in_sample}', color='k', transform=ax.transAxes, fontsize=12)
        if in_sample is True:
            ax.text(0.025, 0.05, f'3-sig outlier: {outlier}', color='k', transform=ax.transAxes, fontsize=12)
        ax.coords[0].set_axislabel('Right Ascension (J2000)', fontsize=10)
        ax.coords[1].set_axislabel('Declination (J2000)', fontsize=10)
        plt.legend(loc='upper right')

        #-----------------------------------------------------------------------
        # plot the Hubble image, rebinned for higher s2n

        # average the cutout image in 4x4 bins
        new_cutout = np.zeros((cutout.shape[0]//4, cutout.shape[1]//4))
        for i in range(new_cutout.shape[0]):
            for j in range(new_cutout.shape[1]):
                new_cutout[i, j] = np.mean(cutout[4*i:4*i+4, 4*j:4*j+4])

        ax = fig.add_subplot(1, 4, 2, projection=wcs)
        wcs = WCS(cutout_file)
        ax.imshow(new_cutout, origin='lower', cmap='gray_r')
        plot_line_on_image(new_cutout, pa_im, ax, 'orange', label='photometric axis')
        ax.set_title('Hubble Image (Rebinned 4x4)')
        ax.coords[0].set_axislabel('Right Ascension (J2000)', fontsize=10)
        ax.coords[1].set_axislabel('Declination (J2000)', fontsize=10)
        plt.legend(loc='upper right')

        #-----------------------------------------------------------------------
        # plot the H-alpha image

        ax = fig.add_subplot(1, 4, 3, projection=wcs)

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

        ax = fig.add_subplot(1, 4, 4, projection=wcs)
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
        plot_line_on_image(vmap, pa_im, ax, 'orange', label='photometric axis')
        plot_line_on_image(vmap, pa_k, ax, c='k', label='kinematic axis')
        ax.text(0.05, 0.85, f'KIN_TYPE={ktype}', color='k', transform=ax.transAxes, fontsize=12)
        ax.coords[0].set_axislabel('Right Ascension (J2000)', fontsize=10)
        ax.coords[1].set_axislabel('Declination (J2000)', fontsize=10)
        # divider = make_axes_locatable(ax)
        # cax = divider.append_axes("right", size="5%", pad=0.05)
        # plt.colorbar(im, cax=cax)
        plt.colorbar(im, fraction=0.046, pad=0.04, label='km/s')
        plt.legend(loc='upper right')

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
