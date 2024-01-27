import unittest
import numpy as np
import os
from astropy.units import Unit

from kl_tools.mocks import MockObservation, DefaultMockObservation
import kl_tools.utils as utils

import pudb

class TestMockObservation(unittest.TestCase):
    def setUp(self) -> None:
        # Set up any necessary objects or data for the tests

        self.true_pars = {
            'g1': 0.01,
            'g2': -0.02,
            'theta_int': np.pi / 3,
            'sini': 0.8,
            'v0': 9,
            'vcirc': 220,
            'rscale': 5.5,
        }

        true_flux = 1e5
        true_hlr = 4.5

        self.datacube_pars = {
            # image meta pars
            'Nx': 40, # pixels
            'Ny': 40, # pixels
            'pix_scale': 0.5, # arcsec / pixel
            # intensity meta pars
            'true_flux': true_flux, # counts
            'true_hlr': true_hlr, # pixels
            'imap_type': 'inclined_exp',
            # velocty meta pars
            'v_model': 'centered',
            'v_unit': Unit('km/s'),
            # emission line meta pars
            'wavelength': 656.28, # nm; halpha
            'lam_unit': 'nm',
            'z': 0.3,
            'R': 5000.,
            's2n': 1000000,
        }

        # Create a MockObservation object
        self.mock_obs = MockObservation(
            self.true_pars, self.datacube_pars
            )

        # Create a MockObservation object with an extended set of true parameters
        self.true_pars_extended = {
            'g1': 0.025,
            'g2': -0.0125,
            'theta_int': np.pi / 6,
            'sini': 0.7,
            'v0': 5,
            'vcirc': 200,
            'rscale': 5,
            'flux': true_flux,
            'hlr': true_hlr,
            'x0': 0.5,
            'y0': -1,
        }

        self.datacube_pars_extended = self.datacube_pars.copy()
        self.datacube_pars_extended.update({
            'v_model': 'offset'
        })

        self.mock_obs_extended = MockObservation(
            self.true_pars_extended, self.datacube_pars_extended
        )

        # make a mock observation that uses basis functions to create the true image
        self.true_pars_basis = {
            'g1': 0.01,
            'g2': -0.02,
            'theta_int': np.pi / 3,
            'sini': 0.3,
            'v0': 9,
            'vcirc': 180,
            'rscale': 3.5,
            'x0': 0.5,
            'y0': -1,
        }

        self.datacube_pars_basis = self.datacube_pars.copy()
        self.datacube_pars_basis.update({
            'basis_type': 'exp_shapelets',
            'use_basis_as_truth': True,
            'basis_kwargs': {
                'plane': 'obs',
                'Nmax': 5,
                'beta': 0.15,
            }
        })

        self.mock_obs_basis = MockObservation(
            self.true_pars_basis, self.datacube_pars_basis
        )

        return

    def tearDown(self) -> None:
        # Clean up any resources used by the tests
        del self.true_pars
        del self.datacube_pars
        del self.mock_obs

        del self.true_pars_extended
        del self.datacube_pars_extended
        del self.mock_obs_extended

        del self.true_pars_basis
        del self.datacube_pars_basis
        del self.mock_obs_basis

        return

    def test_true_pars(self) -> None:
        # Test the true_pars property of MockObservation
        assert self.true_pars == self.mock_obs.true_pars

    def test_datacube_pars(self) -> None:
        # Test the datacube_pars property of MockObservation
        assert self.datacube_pars == self.mock_obs.datacube_pars

    def test_datacube(self) -> None:
        # Test the datacube property of MockObservation
        pass

    def test_generate_datacube(self) -> None:
        # Test the generate_datacube method of MockObservation
        # Add your test code here
        pass

    def test_plots(self) -> None:
        '''
        Make plots of the mock datacube
        '''

        outdir = os.path.join(os.path.dirname(__file__), 'plots', 'test_mocks')
        utils.make_dir(outdir)

        outfile = os.path.join(outdir, 'mock-datacube.png')
        self.mock_obs.datacube.plot(show=False, outfile=outfile)

        outfile = os.path.join(outdir, 'mock-datacube-extended.png')
        self.mock_obs_extended.datacube.plot(show=False, outfile=outfile)

        outfile = os.path.join(outdir, 'mock-datacube-basis.png')
        self.mock_obs_basis.datacube.plot(show=False, outfile=outfile)

        return

class TestDefaultMockObservation(unittest.TestCase):
    def setUp(self) -> None:
        # Set up any necessary objects or data for the tests

        self.mock_obs = DefaultMockObservation()
        self.mock_obs_basis = DefaultMockObservation(imap='basis')
        self.mock_obs_psf = DefaultMockObservation(psf=False)

        try:
            wrong = DefaultMockObservation(imap='other')
        except ValueError:
            pass

        return

    def tearDown(self) -> None:
        # Clean up any resources used by the tests
        pass

    def test_setup_inclined_exp(self) -> None:
        # Test the _setup_inclined_exp method of DefaultMockObservation
        # Add your test code here
        pass

    def test_setup_basis(self) -> None:
        # Test the _setup_basis method of DefaultMockObservation
        # Add your test code here
        pass

    def test_setup_likelihood_test(self) -> None:
        # Test the setup_likelihood_test function
        # Add your test code here
        pass

    def test_fill_test_datacube(self) -> None:
        # Test the _fill_test_datacube function
       
        pass

    def test_setup_simple_emission_line(self) -> None:
        # Test the setup_simple_emission_line function
        # Add your test code here
        pass

    def test_plots(self) -> None:
        '''
        Make plots of the default mock datacube
        '''

        outdir = os.path.join(os.path.dirname(__file__), 'plots', 'test_mocks')
        utils.make_dir(outdir)

        outfile = os.path.join(outdir, 'default-mock-datacube.png')
        self.mock_obs.datacube.plot(show=False, outfile=outfile)

        outfile = os.path.join(outdir, 'default-mock-datacube-basis.png')
        self.mock_obs_basis.datacube.plot(show=False, outfile=outfile)

        outfile = os.path.join(outdir, 'default-mock-datacube-psf.png')
        self.mock_obs_psf.datacube.plot(show=False, outfile=outfile)

        return

if __name__ == '__main__':
    unittest.main()