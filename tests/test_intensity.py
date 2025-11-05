from unittest import TestCase
import numpy as np
import galsim as gs
import matplotlib.pyplot as plt
from astropy.units import Unit
from mpl_toolkits.axes_grid1 import make_axes_locatable

from kl_tools.parameters import ImagePars
from kl_tools.intensity import IntensityMap, IntensityMapFitter, BasisIntensityMap, InclinedExponential, build_intensity_map
from kl_tools.plotting import plot
from kl_tools.utils import get_test_dir, MidpointNormalize, build_map_grid

class TestIntensityMap(TestCase):

    def setUp(self):

        self.show = False
        self.outdir = get_test_dir() / 'out/intensity'
        self.outdir.mkdir(exist_ok=True, parents=True)

        return

    def test_intensity_map_init(self):
        # Test the initialization of the IntensityMap subclasses

        # not typically used in reality for abstract classes

        name = 'test'
        continuum = None
        imap = IntensityMap(name, continuum)
        self.assertTrue(imap.name == name)
        self.assertTrue(imap.continuum == continuum)

        # static continuum image
        continuum = np.random.randn(10, 10)
        imap = IntensityMap(name, continuum)

        # dynamic continuum image
        continuum = lambda shape: np.random.randn(shape[0], shape[1])
        imap = IntensityMap(name, continuum)

        return

    def test_build_intensity_map(self):
        # test the factory method for imap classes

        imap_kwargs = {
            'flux': 1.0,
            'hlr': 1.0,
        }
        imap_exp = build_intensity_map('inclined_exp', imap_kwargs)
        self.assertTrue(isinstance(imap_exp, InclinedExponential))
        self.assertTrue(imap_exp.flux == imap_kwargs['flux'])
        self.assertTrue(imap_exp.hlr == imap_kwargs['hlr'])

        # now for the basis imaps

        # first, shapelets
        shapelets_kwargs = {
            'basis_type': 'shapelets',
            'basis_kwargs': {'nmax': 12, 'beta': 0.5}
        }
        imap_shapelets = build_intensity_map('basis', shapelets_kwargs)
        self.assertTrue(isinstance(imap_shapelets, BasisIntensityMap))
        self.assertTrue(
            imap_shapelets.basis_type == shapelets_kwargs['basis_type']
            )
        self.assertTrue(
            imap_shapelets.basis_kwargs == shapelets_kwargs['basis_kwargs']
            )

        # next, exp_shapelets
        exp_shapelets_kwargs = {
            'basis_type': 'exp_shapelets',
            'basis_kwargs': {'nmax': 10, 'beta': 0.5}
        }
        imap_exp_shapelets = build_intensity_map('basis', exp_shapelets_kwargs)
        self.assertTrue(isinstance(imap_exp_shapelets, BasisIntensityMap))
        self.assertTrue(
            imap_exp_shapelets.basis_type == exp_shapelets_kwargs['basis_type']
            )
        self.assertTrue(
            imap_exp_shapelets.basis_kwargs == exp_shapelets_kwargs['basis_kwargs']
            )

        # next, sersiclets
        sersiclets_kwargs = {
            'basis_type': 'sersiclets',
            'basis_kwargs': {'index': 1, 'nmax': 10, 'beta': 0.5}
        }
        imap_sersiclets = build_intensity_map('basis', sersiclets_kwargs)
        self.assertTrue(isinstance(imap_sersiclets, BasisIntensityMap))
        self.assertTrue(
            imap_sersiclets.basis_type == sersiclets_kwargs['basis_type']
            )
        self.assertTrue(
            imap_sersiclets.basis_kwargs == sersiclets_kwargs['basis_kwargs']
            )

        return

    def test_basis_fitter_faceon(self):
        # test the IntensityMapFitter on a simple example

        plot_dir = self.outdir / 'test_basis_fitter_faceon'
        plot_dir.mkdir(exist_ok=True, parents=True)

        # same image setup each time
        pixel_scale = 0.5
        Nx, Ny = 50, 30
        Nrow, Ncol = Ny, Nx
        shape = (Nrow, Ncol)
        image_pars = ImagePars(shape, pixel_scale=pixel_scale)

        # first, make a simple exponential image
        exp_kwargs = {
            'flux': 1.0,
            'hlr': 2.0, # arcsec
        }
        imap_exp = build_intensity_map('inclined_exp', exp_kwargs)

        theta_pars = {
            'sini': 0.0,
            'theta_int': 0.0,
            'g1': 0.0,
            'g2': 0.0,
            'x0': 0.0,
            'y0': 0.0,
        }
        pars = {}
        im_base = imap_exp.render(image_pars, theta_pars, pars=pars)
        im_peak = np.max(im_base.flatten())

        noise_std = 0.001 * im_peak
        noise = np.random.normal(loc=0.0, scale=noise_std, size=im_base.shape)
        im = im_base + noise

        vmin = min(im.flatten().min(), im_base.flatten().min())
        vmax = max(im.flatten().max(), im_base.flatten().max())
        norm = MidpointNormalize(vmin, vmax)
        cmap = 'RdBu'

        plt.subplot(121)
        im1 = plt.imshow(im_base, origin='lower', norm=norm, cmap=cmap)
        # plt.colorbar()
        plt.title('True inclined exp (sini=0)')

        plt.subplot(122)
        im2 = plt.imshow(im, origin='lower', norm=norm, cmap=cmap)
        # plt.colorbar()

        plt.colorbar(im2, fraction=0.02, pad=0.04, aspect=40)
        plt.title(f'Inclined exp + noise (std={noise_std:.4f})')
        plt.gcf().set_size_inches(10, 5)

        outfile = plot_dir / 'exp_faceon.png'
        plot(self.show, outfile, overwrite=True)

        # now fit the image with the exp shapelets

        # start with just the ground state
        Nmax = 1
        exp_shapelet_kwargs = {
            'basis_type': 'exp_shapelets',
            'basis_kwargs': {
                'nmax': Nmax,
                'beta': 2
                # 'beta': exp_kwargs['hlr'] / pixel_scale,
                },
        }
        imap_exp_shapelets = build_intensity_map('basis', exp_shapelet_kwargs)

        fit = imap_exp_shapelets.render(
            image_pars, theta_pars, pars=pars, image=im
            )

        vmin = min(fit.flatten().min(), im.flatten().min())
        vmax = max(fit.flatten().max(), im.flatten().max())
        norm = MidpointNormalize(vmin, vmax)
        cmap = 'RdBu'

        plt.subplot(131)
        im1 = plt.imshow(im, origin='lower', norm=norm, cmap=cmap)
        # plt.colorbar()
        plt.title('Inclined exp (sini=0) + noise')

        plt.subplot(132)
        im2 = plt.imshow(fit, origin='lower', norm=norm, cmap=cmap)
        # plt.colorbar()
        plt.title('Fitted exp shapelets')

        plt.subplot(133)
        residuals = fit-im
        red_chi2 = np.sum((residuals/noise_std)**2) / (im.size - len(pars))
        im3 = plt.imshow(fit-im, origin='lower', norm=norm, cmap=cmap)
        # plt.colorbar()
        plt.title('Residuals; red_chi2 = {:.3f}'.format(red_chi2))

        plt.suptitle(f'Inclined exp (sini=0) + noise; Nmax={Nmax}')
        plt.gcf().colorbar(im3, fraction=0.02, pad=0.04, aspect=40)
        plt.gcf().set_size_inches(15, 5)

        outfile = plot_dir / 'fit_exp_shapelets_faceon_Nmax{Nmax}.png'
        plot(self.show, outfile, overwrite=True)

        # let's make sure there's not an obviously better solution than our fit
        scale_factor = np.linspace(0.001, 2.0, 1000)
        scales = []
        red_chi2s = []
        for i, scale in enumerate(scale_factor):
            this_fit = scale * fit
            residuals = this_fit-im
            red_chi2 = np.sum((residuals/noise_std)**2) / (im.size - len(pars))

            scales.append(scale)
            red_chi2s.append(red_chi2)
            if i == 0:
                best_chi2 = red_chi2
                best_scale = scale
            else:
                if red_chi2 < best_chi2:
                    best_chi2 = red_chi2
                    best_scale = scale

        plt.plot(scales, red_chi2s)
        plt.axvline(best_scale, color='r', linestyle='--')
        plt.xlabel('Fit coefficient scale factor')
        plt.ylabel('Reduced chi2')
        plt.title(
            f'Scale factor vs. reduced chi2\nBest scale = {best_scale:.3f}\nbest red chi2 = {best_chi2:.3f}'
        )

        outfile = plot_dir / 'scale_vs_red_chi2.png'
        plot(self.show, outfile, overwrite=True)

        # now with a few extra modes
        Nmax = 5
        exp_shapelet_kwargs = {
            'basis_type': 'exp_shapelets',
            'basis_kwargs': {'nmax': Nmax}
        }
        imap_exp_shapelets = build_intensity_map('basis', exp_shapelet_kwargs)

        fit = imap_exp_shapelets.render(
            image_pars, theta_pars, pars=pars, image=im
            )

        vmin = min(fit.flatten().min(), im.flatten().min())
        vmax = max(fit.flatten().max(), im.flatten().max())
        norm = MidpointNormalize(vmin, vmax)
        cmap = 'RdBu'
        plt.subplot(131)
        im1 = plt.imshow(im, origin='lower', norm=norm, cmap=cmap)
        # plt.colorbar()
        plt.title('Inclined exp (sini=0) + noise')

        plt.subplot(132)
        im2 = plt.imshow(fit, origin='lower', norm=norm, cmap=cmap)
        # plt.colorbar()
        plt.title('Fitted exp shapelets')

        plt.subplot(133)
        residuals = fit-im
        red_chi2 = np.sum((residuals/noise_std)**2) / (im.size - len(pars))
        im3 = plt.imshow(fit-im, origin='lower', norm=norm, cmap=cmap)
        # plt.colorbar()
        plt.title('Residuals; red_chi2 = {:.3f}'.format(red_chi2))

        plt.suptitle(f'Inclined exp (sini=0) + noise; Nmax={Nmax}')
        plt.gcf().colorbar(im3, fraction=0.02, pad=0.04, aspect=40)
        plt.gcf().set_size_inches(15, 5)

        outfile = plot_dir / 'fit_exp_shapelets_faceon_Nmax{Nmax}.png'
        plot(self.show, outfile, overwrite=True)

        return

    def test_basis_fitter_faceon_best_fit(self):
        # test the basis fitter on a simple example, with the best beta/coeff

        plot_dir = self.outdir / 'test_basis_fitter_faceon_best_fit'
        plot_dir.mkdir(exist_ok=True, parents=True)

        use_psf = False
        use_galsim = True
        if use_psf:
            psf_sigma = 0.01 # arcsec
            psf = gs.Gaussian(
                sigma=psf_sigma,
                flux=1.0,
            )
        else:
            psf = None

        Nrow = 30
        Ncol = 50
        pixel_scale = 0.5 # arcsec
        image_pars = ImagePars(
            shape=(Nrow, Ncol),
            indexing='ij',
            pixel_scale=pixel_scale
        )

        Nx = image_pars.Nx
        Ny = image_pars.Ny

        flux = 1.0
        scale_radius = 2.0 # arcsec

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
                nx=Nx,
                ny=Ny,
                method='no_pixel'
            ).array

        else:
            # centered at (0, 0) in pixel coordinates
            X, Y = build_map_grid(Nx, Ny, indexing='xy')
            R = np.sqrt(X**2 + Y**2)

            # parameters for synthetic exponential profile
            I0 = 1.23
            r_s = scale_radius / pixel_scale

            image = I0 * np.exp(-R / r_s)

        # add noise
        im_peak = image.flatten().max()

        noise_std = 0.001 * im_peak
        noise = np.random.normal(
            loc=0.0,
            scale=noise_std,
            size=image.shape
        )
        image_noisy = image + noise

        # make the base image comparison
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

        outfile = plot_dir / 'exp_faceon_image.png'
        plot(self.show, outfile, overwrite=True)

        # the following were determined from a test run scanning
        # over beta & a single coefficient for the above (very low-noise) image 
        # using just the ground state of the exp_shapelets basis but *not* using
        # either of the BasisIntensityMap or IntensityMapFitter classes
        best_beta = 2.0040679359
        best_coeff = 0.0176352705
        best_red_chi2 = 1.0667858430

        # now let's try it with the proper classes to see if they agree
        imap = BasisIntensityMap(
            basis_type='exp_shapelets',
            basis_kwargs={
                'nmax': 1,
                'psf': psf,
                'beta': best_beta
                }
            )

        # perfectly face-on, no other transformations
        theta_pars = {
            'x0': 0.0,
            'y0': 0.0,
            'sini': 0.0,
            'theta_int': 0.0,
            'g1': 0.0,
            'g2': 0.0,
        }
        pars = None
        fitted_image = imap.render(
            image_pars,
            theta_pars,
            pars,
            image=image_noisy,
        )

        diff = fitted_image - image_noisy
        N_free_pars = image_noisy.size - 1
        fitted_red_chi2 = np.sum((diff / noise_std)**2) / N_free_pars

        # now plot the fitted image and residuals
        vmin = min(fitted_image.flatten().min(), image_noisy.flatten().min())
        vmax = max(fitted_image.flatten().max(), image_noisy.flatten().max())
        norm = MidpointNormalize(vmin=vmin, vmax=vmax)
        cmap = 'RdBu'
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
        im1 = ax1.imshow(image_noisy, origin='lower', cmap=cmap, norm=norm)
        ax1.set_title('Noisy image')
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im1, cax=cax)
        im2 = ax2.imshow(fitted_image, origin='lower', cmap=cmap, norm=norm)
        ax2.set_title('Fitted image')
        divider = make_axes_locatable(ax2)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im2, cax=cax)
        diff = fitted_image - image_noisy
        norm = MidpointNormalize(vmin=np.min(diff), vmax=np.max(diff))
        im3 = ax3.imshow(diff, origin='lower', cmap=cmap, norm=norm)    
        ax3.set_title('Residuals')
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im3, cax=cax)
        plt.suptitle(
            f'Fitted image with exp_shapelets basis\n'
            f'Best beta = {best_beta:.3f}, coeff = {best_coeff:.3f}\n'
            f'Reduced chi2 = {fitted_red_chi2:.3f}'
        )
        plt.gcf().set_size_inches(12, 4)

        outfile = plot_dir / 'exp_faceon_fit.png'
        plot(self.show, outfile, overwrite=True)

        # check that the fitted image is close to the original
        self.assertTrue(
            np.isclose(
                fitted_red_chi2,
                best_red_chi2,
                rtol=5e-2,
                atol=1e-1
                ),
        )

        return
    
    def _make_faceon_image(
            self,
            image_pars,
            flux=1.0,
            hlr=2.0,
            noise_frac=0.001,
            psf=None
            ):
        '''
        Make a face-on inclined exponential image with noise.
        '''

        Nrow = image_pars.Nrow
        Ncol = image_pars.Ncol
        pixel_scale = image_pars.pixel_scale
        image_pars = ImagePars(
            shape=(Nrow, Ncol),
            indexing='ij',
            pixel_scale=pixel_scale
        )

        Nx = image_pars.Nx
        Ny = image_pars.Ny

        # non-inclined exponential disk
        obj = gs.Exponential(
            flux=flux,
            half_light_radius=hlr, # arcsec
        )

        if psf is not None:
            obj = gs.Convolution(obj, psf)

        image = obj.drawImage(
            scale=image_pars.pixel_scale,
            nx=Nx,
            ny=Ny,
        ).array
        im_peak = image.flatten().max()

        # set noise to a fraction of the peak flux
        noise_std = noise_frac * im_peak
        noise = np.random.normal(
            loc=0.0,
            scale=noise_std,
            size=image.shape
        )
        image_noisy = image + noise

        return image_noisy, noise_std

    def test_plot_intensity_map(self):
        # test the plot() method for imap classes
        pass