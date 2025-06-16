from unittest import TestCase
import numpy as np
import galsim as gs
import matplotlib.pyplot as plt
from astropy.units import Unit
from mpl_toolkits.axes_grid1 import make_axes_locatable

from kl_tools.parameters import ImagePars
from kl_tools.galaxy_fitter import estimate_gal_properties
from kl_tools.plotting import plot
from kl_tools.utils import get_test_dir, MidpointNormalize, build_map_grid

class TestGalaxyFitter(TestCase):

    def setUp(self):

        self.show = False
        self.vb = False
        self.outdir = get_test_dir() / 'out/galaxy_fitter'
        self.outdir.mkdir(exist_ok=True, parents=True)

        return

    def test_galaxy_fitter_faceon(self):
        # Test the galaxy fitter with a simple, face-on example

        use_psf = True
        use_galsim = True

        Nrow = 30
        Ncol = 50
        pixel_scale = 1 # arcsec
        # pixel_scale = 0.5 # arcsec
        image_pars = ImagePars(
            shape=(Nrow, Ncol),
            indexing='ij',
            pixel_scale=pixel_scale
        )

        Nx = image_pars.Nx
        Ny = image_pars.Ny

        if use_psf:
            psf_sigma = 2*pixel_scale # arcsec
            psf = gs.Gaussian(
                sigma=psf_sigma,
                flux=1.0,
            )
        else:
            psf = None

        flux = 1.0
        scale_radius = 2.0 # arcsec
        offset = (-10, 5) # pixels

        if use_galsim is True:
            # non-inclined exponential disk
            obj = gs.Exponential(
                flux=flux,
                scale_radius=scale_radius
            )

            if use_psf is True:
                obj = gs.Convolution(obj, psf)
            else:
                psf = None

            image = obj.drawImage(
                scale=image_pars.pixel_scale,
                offset=gs.PositionD(offset[0], offset[1]),
                nx=Nx,
                ny=Ny,
                # method='no_pixel'
            ).array

        else:
            # centered at (0, 0) in pixel coordinates
            X, Y = build_map_grid(Nx, Ny, indexing='xy')
            R = np.sqrt((X-offset[0])**2 + (Y-offset[1])**2)

            # parameters for synthetic exponential profile
            I0 = 1.23
            r_s = scale_radius / pixel_scale

            image = I0 * np.exp(-R / r_s)

        # add noise
        im_peak = image.flatten().max()

        noise_std = 0.05 * im_peak
        noise = np.random.normal(
            loc=0.0,
            scale=noise_std,
            size=image.shape
        )
        image_noisy = image + noise

        vmin = min(image.flatten().min(), image_noisy.flatten().min())
        vmax = max(image.flatten().max(), image_noisy.flatten().max())
        norm = MidpointNormalize(vmin=vmin, vmax=vmax)
        cmap = 'RdBu'

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))

        im1 = ax1.imshow(image, origin='lower', cmap=cmap, norm=norm)
        ax1.set_title('Original image')
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im1, cax=cax)

        im2 = ax2.imshow(image_noisy, origin='lower', cmap=cmap, norm=norm)
        ax2.set_title('Noisy image')
        divider = make_axes_locatable(ax2)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im2, cax=cax)

        diff = image_noisy - image
        norm = MidpointNormalize(vmin=np.min(diff), vmax=np.max(diff))
        im3 = ax3.imshow(image_noisy-image, origin='lower', cmap=cmap, norm=norm)
        ax3.set_title('Difference image')
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im3, cax=cax)

        outfile = self.outdir / 'simulated_faceon.png'
        plot(self.show, out_file=outfile, overwrite=True)

        guess = {
            'flux': 10, # add some error
            'scale_radius': 2 * scale_radius, # add some error
            'sini': 0.5,  # no inclination
            'theta': np.pi/6,  # no rotation
            'x': 0.5*offset[0],  # x offset in pixels
            'y': 0.5*offset[1],  # y offset in pixels
        }
        bounds = {
            'flux': (1e-4, 1e4),  # flux must be positive
            'scale_radius': (0.0, 10),  # scale radius must be positive
            # 'x': (-Nx/2, Nx/2),  # x offset in pixels
            # 'y': (-Ny/2, Ny/2),  # y offset in pixels
        }
        result = estimate_gal_properties(
            image_noisy,
            image_pars,
            guess,
            bounds,
            sersic_n=1.0,
            psf=psf,
            optimizer='differential_evolution',
            optimize_kwargs={'tol': 1e-6}
        )

        fit_pars = result['params']
        model_image = result['model_image']

        if self.vb is True:
            print(result.keys())
            print('fit_pars:', fit_pars)
            print(f"Estimated flux: {fit_pars['flux']:.3f} (true: {flux})")
            print(f"Estimated scale radius: {fit_pars['scale_radius']:.3f} (true: {scale_radius/pixel_scale})")
            print(f"Estimated x offset: {fit_pars['x']:.3f} (true: {offset[0]})")
            print(f"Estimated y offset: {fit_pars['y']:.3f} (true: {offset[1]})")

        vmin = min(model_image.flatten().min(), image_noisy.flatten().min())
        vmax = max(model_image.flatten().max(), image_noisy.flatten().max())
        norm = MidpointNormalize(vmin=vmin, vmax=vmax)
        cmap = 'RdBu'

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))

        im1 = ax1.imshow(image_noisy, origin='lower', cmap=cmap, norm=norm)
        ax1.set_title('Noisy image')
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im1, cax=cax)

        im2 = ax2.imshow(model_image, origin='lower', cmap=cmap, norm=norm)
        ax2.set_title('Best fit image')
        divider = make_axes_locatable(ax2)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im2, cax=cax)

        diff = model_image - image_noisy
        red_chi2 = np.sum((diff/noise_std)**2) / (image_noisy.size - len(fit_pars))
        norm = MidpointNormalize(vmin=np.min(diff), vmax=np.max(diff))
        im3 = ax3.imshow(diff, origin='lower', cmap=cmap, norm=norm)
        ax3.set_title(f'Residuals; red $\chi^2$ = {red_chi2:.3f}')
        # cax = divider.append_axes("right", size="5%", pad=0.05)
        # plt.colorbar(im3, cax=cax)
        plt.gcf().set_size_inches(18, 4)

        outfile = self.outdir / 'fitted_faceon.png'
        plot(self.show, out_file=outfile, overwrite=True)

        # make sure the red_chi2 is reasonable
        self.assertAlmostEqual(red_chi2, 1.0, delta=0.5)

        return
