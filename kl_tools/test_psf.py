import numpy as np
import galsim as gs
import os
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from argparse import ArgumentParser
from astropy.units import Unit
from copy import deepcopy

import utils
import intensity
import mocks

import ipdb

def parse_args():
    parser = ArgumentParser()

    parser.add_argument('--show', action='store_true', default=False,
                        help='Set to show test plots')
    return parser.parse_args()

def plot_components(psf_im, gal_im, obs_im, s=(18,4),
                    outfile=None, show=False, title=None):

    psf_mom = psf_im.FindAdaptiveMom()
    psf_cen = psf_mom.moments_centroid

    gal_mom = gal_im.FindAdaptiveMom()
    gal_cen = gal_mom.moments_centroid

    obs_mom = obs_im.FindAdaptiveMom()
    obs_cen = obs_mom.moments_centroid

    im_ratio = s[0] / s[1]
    plt.subplot(131)
    plt.imshow(psf_im.array, origin='lower')
    plt.colorbar(fraction=0.047*im_ratio)
    plt.title(f'PSF image; centroid=({psf_cen.x:.3f}, {psf_cen.y:.3f})')

    plt.subplot(132)
    plt.imshow(gal_im.array, origin='lower')
    plt.colorbar(fraction=0.047*im_ratio)
    plt.title(f'gal image; centroid=({gal_cen.x:.3f}, {gal_cen.y:.3f})')

    plt.subplot(133)
    plt.imshow(obs_im.array, origin='lower')
    plt.colorbar(fraction=0.047*im_ratio)
    plt.title(f'obs image; centroid=({obs_cen.x:.3f}, {obs_cen.y:.3f})')

    plt.gcf().set_size_inches(s)

    if title is not None:
        plt.suptitle(title)

    if outfile is not None:
        plt.savefig(outfile, bbox_inches='tight', dpi=300)

    if show is True:
        plt.show()
    else:
        plt.close()

    return obs_im.array

def plot_interpolated_im_compare(render, pix_scale, shape,
                                 outfile=None, show=False, s=(16,8)):
    '''
    render: np.ndarray
        Return of imap.render()
    '''

    Nx, Ny = shape[0], shape[1]

    # Image will inherit render shape
    gal_im = gs.Image(render, scale=pix_scale)
    gal_int_im = gs.InterpolatedImage(gal_im)

    im_ratio = s[0] / s[1]
    plt.subplot(231)
    plt.imshow(render, origin='lower')
    plt.colorbar(fraction=0.047*im_ratio)
    plt.title('KL imap.render()')

    plt.subplot(232)
    plt.imshow(gal_im.array, origin='lower')
    plt.colorbar(fraction=0.047*im_ratio)
    plt.title('GalSim Image obj of imap.render()')

    plt.subplot(233)
    interp_im = gal_int_im.drawImage(
        scale=pix_scale, nx=Nx, ny=Ny, method='no_pixel'
        ).array
    plt.imshow(interp_im, origin='lower')
    plt.colorbar(fraction=0.047*im_ratio)
    plt.title('GalSim InterpolatedImage.draw() of imap.render()')

    plt.subplot(235)
    plt.imshow(gal_im.array-render, origin='lower')
    plt.colorbar(fraction=0.047*im_ratio)
    plt.title('GalSim Image - imap.render()')

    plt.subplot(236)
    plt.imshow(interp_im-render, origin='lower')
    plt.colorbar(fraction=0.047*im_ratio)
    plt.title('GalSim InterpolatedImage.draw() - imap.render()')

    plt.gcf().set_size_inches(s)

    if outfile is not None:
        plt.savefig(outfile, bbox_inches='tight', dpi=300)

    if show is True:
        plt.show()
    else:
        plt.close()

    return

def main(args):

    show = args.show

    outdir = os.path.join(utils.TEST_DIR, 'test-psf')
    utils.make_dir(outdir)

    pix_scale = 0.25
    # Nx, Ny = 100, 100
    Nx, Ny = 101, 101

    #------------------------------------------------------------------
    print('Starting full galsim rendering test')
    fwhm = 3 # arcsec
    psf = gs.Gaussian(fwhm=fwhm, flux=1.)
    psf_im = psf.drawImage(scale=pix_scale, nx=Nx, ny=Ny)

    flux = 3e3
    hlr = 3 # arcsec
    sini = 0.85
    inc = gs.Angle(np.arcsin(sini), gs.radians)
    theta = gs.Angle(45, gs.degrees)
    gal = gs.InclinedExponential(
        inc, flux=flux, half_light_radius=hlr
        )
    gal = gal.rotate(theta)
    gal_im = gal.drawImage(scale=pix_scale, nx=Nx, ny=Ny)

    obs = gs.Convolve([psf, gal])
    obs_im = obs.drawImage(scale=pix_scale, nx=Nx, ny=Ny)

    print('Plotting galsim PSF & gal image')
    outfile = os.path.join(outdir, 'component-images-gs.png')
    out_im = plot_components(
        psf_im, gal_im, obs_im, outfile=outfile, show=show,
        title='GSObjects only'
        )

    #------------------------------------------------------------------
    print('Starting kl render + galsim psf rendering test')
    true_flux = flux
    true_hlr = hlr
    true_pars = {
        # 'g1': 0.025,
        # 'g2': -0.0125,
        'g1': 0.,
        'g2': 0.,
        'theta_int': theta.rad,
        'sini': sini,
        'v0': 5,
        'vcirc': 200,
        'rscale': 3,
    }
    datacube_pars = {
        # image meta pars
        'Nx': Nx, # pixels
        'Ny': Nx, # pixels
        'pix_scale': pix_scale, # arcsec / pixel
        # intensity meta pars
        'true_flux': true_flux, # counts
        'true_hlr': true_hlr, # pixels
        # velocty meta pars
        'v_model': 'centered',
        'v_unit': Unit('km / s'),
        'r_unit': Unit('kpc'),
        # emission line meta pars
        'wavelength': 656.28, # nm; halpha
        'lam_unit': 'nm',
        'z': 0.3,
        'R': 5000.,
        'sky_sigma': 0.5, # pixel counts for mock data vector
        'psf': psf # same as above
    }

    print('Setting up test datacube and true Halpha image')
    datacube, vmap, true_im = mocks.setup_likelihood_test(
        true_pars, datacube_pars
        )
    Nspec, Nx, Ny = datacube.shape
    lambdas = datacube.lambdas
    datacube.set_psf(datacube_pars['psf'])

    imap_kwargs = {
        'flux': true_flux,
        'hlr': true_hlr,
    }
    # ipdb.set_trace()
    imap = intensity.build_intensity_map(
        'inclined_exp', datacube, imap_kwargs
        )

    # now render using true sini
    theta_pars = {
        'g1': true_pars['g1'],
        'g2': true_pars['g2'],
        'theta_int': true_pars['theta_int'],
        'sini': true_pars['sini']
    }

    render = imap.render(theta_pars, datacube, pars=None)

    # quick plot of render vs. interpolated image
    outfile = os.path.join(outdir, 'render-vs-interpolated-im.png')
    plot_interpolated_im_compare(
        render, pix_scale, (Nx, Ny), outfile=outfile, show=show
        )

    # Image will inherit render shape
    gal_im = gs.Image(render, scale=pix_scale)
    gal_int_im = gs.InterpolatedImage(gal_im)
    # interp_im = gal_int_im.drawImage(
    #     scale=pix_scale, nx=Nx, ny=Ny, method='no_pixel'
    #     ).array

    psf = datacube.get_psf()
    psf_im = psf.drawImage(scale=pix_scale, nx=Nx, ny=Ny)

    obs = gs.Convolve([psf, gal_int_im])
    # no_pixel bc the interpolated image is already pixelated
    obs_im = obs.drawImage(
        scale=pix_scale, nx=Nx, ny=Ny, method='no_pixel'
        )

    print('Plotting galsim PSF & gal image')
    outfile = os.path.join(outdir, 'component-images-imap.png')
    out_im_dc = plot_components(
        psf_im, gal_im, obs_im, outfile=outfile, show=show,
        title='DataCube interpolated image render'
        )

    #------------------------------------------------------------------
    print('Starting kl render vs. galsim render test')
    plt.imshow(
        out_im_dc-out_im, origin='lower', norm=colors.CenteredNorm(),
        cmap='RdBu'
        )
    plt.colorbar()
    plt.title('DataCube image render - GalSim only image render')
    plt.gcf().set_size_inches(9,8)

    outfile = os.path.join(outdir, 'image-compare.png')
    plt.savefig(outfile, bbox_inches='tight', dpi=300)
    if show is True:
        plt.show()
    else:
        plt.close()

    plt.imshow(
        100*(out_im_dc-out_im)/out_im, origin='lower',
        cmap='RdBu',
        vmin=-10,
        vmax=10
        )
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('% Error', rotation=270)
    plt.title('DataCube image render - GalSim only image render ()')
    plt.gcf().set_size_inches(9,8)

    outfile = os.path.join(outdir, 'image-compare-perr.png')
    plt.savefig(outfile, bbox_inches='tight', dpi=300)
    if show is True:
        plt.show()
    else:
        plt.close()

    return 0

if __name__ == '__main__':
    args = parse_args()

    print('Starting tests')
    rc = main(args)

    if rc == 0:
        print('All tests ran succesfully')
    else:
        print(f'Tests failed with return code of {rc}')
