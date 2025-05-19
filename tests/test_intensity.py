from unittest import TestCase
import numpy as np
import os
import matplotlib.pyplot as plt
from astropy.units import Unit

from kl_tools.mocks import setup_likelihood_test
from kl_tools.parameters import ImagePars
from kl_tools.intensity import IntensityMap, IntensityMapFitter, BasisIntensityMap, InclinedExponential, build_intensity_map
from kl_tools.plotting import plot
from kl_tools.utils import get_test_dir

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

        plot_dir = self.outdir / 'test_basis_fitter'
        plot_dir.mkdir(exist_ok=True, parents=True)

        # same image setup each time
        pixel_scale = 0.2
        Nx, Ny = 40, 30
        Nrow, Ncol = Ny, Nx
        shape = (Nrow, Ncol)
        image_pars = ImagePars(shape, pixel_scale=pixel_scale)

        # first, make a simple exponential image
        exp_kwargs = {
            'flux': 1.0,
            'hlr': 1.0, # arcsec
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

        noise = (0.1 * im_peak) * np.random.random(shape)
        im = im_base + noise

        plt.subplot(121)
        plt.imshow(im_base, origin='lower')
        plt.colorbar()
        plt.title('True inclined exp (sini=0)')

        plt.subplot(122)
        plt.imshow(im, origin='lower')
        plt.colorbar()
        plt.title('Inclined exp + noise')
        plt.gcf().set_size_inches(10, 5)

        outfile = plot_dir / 'exp_faceon.png'
        plot(self.show, outfile, overwrite=True)

        # now fit the image with the exp shapelets
        exp_shapelet_kwargs = {
            'basis_type': 'exp_shapelets',
            'basis_kwargs': {'nmax': 3} # don't need many for face-on
        }
        imap_exp_shapelets = build_intensity_map('basis', exp_shapelet_kwargs)

        fit = imap_exp_shapelets.render(
            image_pars, theta_pars, pars=pars, image=im
            )

        plt.subplot(131)
        plt.imshow(im, origin='lower')
        plt.colorbar()
        plt.title('Inclined exp (sini=0) + noise')

        plt.subplot(132)
        plt.imshow(fit, origin='lower')
        plt.colorbar()
        plt.title('Fitted exp shapelets')

        plt.subplot(133)
        plt.imshow(fit-im, origin='lower')
        plt.colorbar()
        plt.title('Residuals')
        plt.gcf().set_size_inches(15, 5)

        outfile = plot_dir / 'fit_exp_shapelets_faceon.png'
        plot(self.show, outfile, overwrite=True)

        return

    def test_plot_intensity_map(self):
        # test the plot() method for imap classes
        pass