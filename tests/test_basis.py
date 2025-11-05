import unittest
import numpy as np
import galsim as gs

import kl_tools.basis as kl_basis
from kl_tools.parameters import ImagePars
from kl_tools.utils import get_test_dir

class TestBasis(unittest.TestCase):
    def setUp(self):
        self.show = False
        self.outdir = get_test_dir() / 'out/basis'
        self.outdir.mkdir(parents=True, exist_ok=True)

        return

    def test_basis_init(self):
        # Instantiate the parent basis class

        # NOTE: this would fail in terms of rendering a basis as
        # the basis class is an abstract class, but it should still
        # instantiate without error
        name = 'test_basis'

        basis = kl_basis.Basis(name)
        self.assertTrue(basis.name == name)
        self.assertTrue(basis.offset == None)
        self.assertTrue(basis.beta == None)

        basis_offset = kl_basis.Basis(name, offset=(0,1))
        basis_offset = kl_basis.Basis(name, offset=(2.4, -3.8))
        basis_beta = kl_basis.Basis(name, beta=1.0)
        basis_beta = kl_basis.Basis(name, beta=8)

        basis_all = kl_basis.Basis(name, offset=(0,1), beta=1.3)
        self.assertTrue(basis_all.offset == (0,1))
        self.assertTrue(basis_all.beta == 1.3)

        # now make sure it fails if we try to render it to an image
        coefficients = np.random.randn(10)
        nx, ny = 20, 20
        nrow, ncol = ny, nx
        pixel_scale = 0.2 # arcsec/pixel
        image_pars = ImagePars((nrow, ncol), pixel_scale=pixel_scale)
        plane = 'obs'
        try:
            basis_all.render_im(coefficients, image_pars, plane=plane)
        except ValueError:
            # value error because the len(coefficients) != basis.nmax, which is 
            # None
            pass

        return

    def test_polar_basis_init(self):
        # Instantiate the PolarBasis class

        # almost identical to the basis class, but with additional structure 
        # for polar modes
        name = 'polar_test'
        nmax = 4
        polar = kl_basis.PolarBasis(
            name, nmax=nmax, offset=(0,1), beta=1.3
            )
        self.assertTrue(polar.name == name)
        self.assertTrue(polar.nmax == nmax)
        self.assertTrue(polar.N == nmax**2)
        self.assertTrue(polar.offset == (0,1))
        self.assertTrue(polar.beta == 1.3)
        self.assertEqual(polar.n_to_lm(0), (0, 0))
        self.assertEqual(polar.n_to_lm(1), (1, -1))
        self.assertEqual(polar.n_to_lm(2), (1, 0))
        self.assertEqual(polar.n_to_lm(3), (1, 1))

        # now try without setting nmax explicitly
        polar = kl_basis.PolarBasis(
            name, offset=(0,1), beta=1.3
            )
        self.assertEqual(polar.nmax, None)
        self.assertEqual(polar.N, None)

        return

    def test_sersiclet_basis_init(self):
        # Instantiate the SersicletBasis class

        index = 2 # Sersic index
        nmax = 3 # maximum number of quantum levels
        sersiclets = kl_basis.SersicletBasis(
            index, nmax=nmax, beta=1.2
            )

        self.assertEqual(sersiclets.nmax, nmax)
        self.assertEqual(sersiclets.index, index)
        self.assertEqual(sersiclets.N, nmax**2)

        # now try without setting nmax explicitly
        sersiclets = kl_basis.SersicletBasis(
            index, beta=1.4
            )

        self.assertEqual(sersiclets.nmax, None)
        self.assertEqual(sersiclets.N, None)

        return

    def test_exp_shapelet_basis_init(self):
        # Instantiate the ExpShapeletBasis class

        nmax = 3 # maximum number of quantum levels
        offset = (0.3, -1.2)
        beta = 1.2
        exp_shapelets = kl_basis.ExpShapeletBasis(
            nmax=nmax, beta=beta, offset=offset
            )

        self.assertEqual(exp_shapelets.beta, beta)
        self.assertEqual(exp_shapelets.nmax, nmax)
        self.assertEqual(exp_shapelets.offset, offset)
        self.assertEqual(exp_shapelets.N, nmax**2)

        # now try without setting nmax explicitly
        exp_shapelets = kl_basis.ExpShapeletBasis(
            beta=1.4, offset=(0.2, 0.1)
            )

        self.assertEqual(exp_shapelets.nmax, None)
        self.assertEqual(exp_shapelets.N, None)

        return

    def test_shapelet_basis_init(self):
        # Instantiate the ShapeletBasis class

        nmax = 3 # maximum number of Nx/Ny eigenstates
        offset = (0.3, -1.2)
        beta = 1.2
        exp_shapelets = kl_basis.ShapeletBasis(
            nmax=nmax, beta=beta, offset=offset
            )

        self.assertEqual(exp_shapelets.beta, beta)
        self.assertEqual(exp_shapelets.nmax, nmax)
        self.assertEqual(exp_shapelets.offset, offset)
        self.assertEqual(exp_shapelets.N, (nmax+1)*(nmax+2)/2)

        # now try without setting nmax explicitly
        exp_shapelets = kl_basis.ShapeletBasis(
            beta=1.4, offset=(0.2, 0.1)
            )

        self.assertEqual(exp_shapelets.nmax, None)
        self.assertEqual(exp_shapelets.N, None)

        return

    def test_get_basis_func(self):
        # Test the get_basis_func() method for each of the standard classes

        nmax = 3
        beta = 1.5

        sersiclets = kl_basis.SersicletBasis(1, nmax=nmax, beta=beta)
        exp_shapelets = kl_basis.ExpShapeletBasis(nmax=nmax, beta=beta)
        shapelets = kl_basis.ShapeletBasis(nmax=nmax, beta=beta)

        # test the get_basis_func() method
        x = np.arange(-2*beta, 2*beta, 0.1)
        y = np.arange(-2*beta, 2*beta, 0.1)
        X, Y = np.meshgrid(x, y)
        for basis in [sersiclets, exp_shapelets, shapelets]:
            for n in range(basis.N):
                # as none of these require a PSF, we can just call the function
                # without any image parameters
                val = basis.get_basis_func(n, X, Y)

        return

    def test_plot_basis_funcs(self):
        # Test the plot_basis_funcs() method for each of the standard classes

        nmax = 3
        beta = 1
        nx, ny = 25,  25
        nrow, ncol = ny, nx
        pix_scale = 0.5
        size = (10,6)

        image_pars = ImagePars((nrow, ncol), pixel_scale=pix_scale)

        sersiclets = kl_basis.SersicletBasis(1, nmax=nmax, beta=beta)
        exp_shapelets = kl_basis.ExpShapeletBasis(nmax=nmax, beta=beta)
        shapelets = kl_basis.ShapeletBasis(nmax=nmax, beta=beta)

        # test the plot_basis_funcs() method
        for basis in [sersiclets, exp_shapelets, shapelets]:
            outfile = self.outdir / f'{basis.name}-basis-funcs.png'
            basis.plot_basis_funcs(
                image_pars, show=self.show, outfile=outfile, size=size
            )

        #-----------------------------------------------------------------
        # Redo the above, but now with a PSF convolution on the basis

        psf = gs.Gaussian(sigma=2*pix_scale)

        sersiclets_psf = kl_basis.SersicletBasis(
            1, nmax=nmax, beta=beta, psf=psf
            )
        exp_shapelets_psf = kl_basis.ExpShapeletBasis(
            nmax=nmax, beta=beta, psf=psf
            )
        shapelets_psf = kl_basis.ShapeletBasis(
            nmax=nmax, beta=beta, psf=psf
            )

        # test the plot_basis_funcs() method
        for basis in [sersiclets_psf, exp_shapelets_psf, shapelets_psf]:
            outfile = self.outdir / f'{basis.name}-basis-funcs-psf.png'
            basis.plot_basis_funcs(
                image_pars, show=self.show, outfile=outfile, size=size
            )

        return
