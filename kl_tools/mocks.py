import numpy as np
from astropy.units import Unit
import galsim as gs
from abc import abstractmethod
from typing import Union
import os

from kl_tools.parameters import Pars
from kl_tools.emission import EmissionLine
import kl_tools.cube as cube
from kl_tools.cube import CubePars, DataCube
from kl_tools.velocity import VelocityMap
from kl_tools.likelihood import DataCubeLikelihood
import kl_tools.intensity as intensity
import kl_tools.utils as utils

import ipdb
import pudb

class MockObservation(object):
    '''
    This class is meant to organize the specification, construction, and
    diagnostics of a mock datacube observation to be used for various
    test scripts. This is to ensure consistency in input between multiple
    tests
    '''

    # TODO: This may need to be restructured to allow for more general models
    _req_true_pars = ['g1', 'g2', 'theta_int', 'sini', 'v0', 'vcirc', 'rscale']
    _opt_true_pars = ['flux', 'hlr', 'beta', 'x0', 'y0']
    _req_datacube_pars = ['Nx', 'Ny', 'pix_scale', 'true_flux', 'true_hlr', 'z', 'R', 'wavelength', 'v_model', 'imap_type']
    _opt_datacube_pars = ['s2n', 'sky_sigma', 'continuum_template', 'v_unit', 'r_unit', 'lam_unit', 'psf', 'basis_type', 'basis_kwargs', 'use_basis_as_truth']

    def __init__(self, true_pars:dict, datacube_pars: dict) -> None:
        '''
        true_pars: dict
            A dictionary of parameters that are to be sampled over, along with
            their true values
        datacube_pars: dict
            A dictionary of parameters needed for mock datacube construction
        '''

        utils.check_fields(
            true_pars, self._req_true_pars, self._opt_true_pars, 'true_pars'
            )
        utils.check_fields(
            datacube_pars, self._req_datacube_pars, self._opt_datacube_pars, 'datacube_pars'
            )

        self.true_pars = true_pars
        self.datacube_pars = datacube_pars
        self.generate_datacube()

        return

    def generate_datacube(self) -> None:
        '''
        Generate the mock datacube
        '''
        datacube, true_vmap, true_im = setup_likelihood_test(
            self.true_pars, self.datacube_pars
            )

        try:
            datacube.set_psf(self.datacube_pars['psf'])
        except KeyError:
            pass

        self.datacube = datacube
        self.true_vmap = true_vmap
        self.true_im = true_im

        return

    @property
    def psf(self) -> gs.GSObject:
        try:
            return self.datacube_pars['psf']
        except KeyError:
            return None

    def plot(self, plot_kwargs:dict) -> None:
        '''
        Plot the mock observation, both the stack and individual channels

        plot_kwargs: dict
            A dictionary of plotting kwargs to pass to the datacube.plot()
            method
        '''

        self.datacube.plot(**plot_kwargs)

        return

class DefaultMockObservation(MockObservation):
    '''
    This is a specific mock observation that is meant to be used for
    baseline testing
    '''

    _true_pars = {
        'g1': 0.025,
        'g2': -0.0125,
        'theta_int': np.pi / 6,
        'sini': 0.7,
        'v0': 5,
        'vcirc': 200,
        'rscale': 5,
    }

    # exp disk parameters
    _true_flux = 1.8e4
    _true_hlr = 3.5

    _base_datacube_pars = {
        # image meta pars
        'Nx': 40, # pixels
        'Ny': 40, # pixels
        'pix_scale': 0.5, # arcsec / pixel
        # intensity meta pars
        'true_flux': _true_flux, # counts
        'true_hlr': _true_hlr, # pixels
        # velocty meta pars
        'v_model': 'centered',
        'v_unit': Unit('km/s'),
        'r_unit': Unit('kpc'),
        # emission line meta pars
        'wavelength': 656.28, # nm; halpha
        'lam_unit': 'nm',
        'z': 0.3,
        'R': 5000.,
        's2n': 1000000,
        'true_flux': 1.8e4,
        'true_hlr': 3.5,

    }

    _datacube_exp_pars = {
        'imap_type': 'inclined_exp',
    }

    _datacube_basis_pars = {
        'imap_type': 'basis',
        'basis_type': 'exp_shapelets',
        'use_basis_as_truth': True,
        'basis_kwargs': {
            'Nmax': 6,
            'plane': 'obs',
            'beta': 0.15,
        }
    }

    _psf = gs.Gaussian(fwhm=0.8)

    def __init__(self, imap: str='inclined_exp', psf: bool=True) -> None:
        '''
        imap: str
            The type of intensity map to use for the mock observation. If 'basis', then the true intensity map is the defined basis fit to the galsim-produced image defined in the default _datacube_pars image type
        psf: bool
            Set to include the default PSF in the datacube model generation
        '''

        if imap == 'inclined_exp':
            self._setup_inclined_exp()
        elif imap == 'basis':
            self._setup_basis()
        else:
            raise ValueError('imap must be inclined_exp or basis')

        if psf is True:
            self.datacube_pars.update({'psf': self._psf})

        # datacube_pars is set according to imap type in the respective setup method
        if imap == 'inclined_exp':
            exp_pars = {
                'flux': self._true_flux,
                'hlr': self._true_hlr,
            }
            true_pars = {**self._true_pars, **exp_pars}
        elif imap == 'basis':
            true_pars = self._true_pars

        super().__init__(true_pars, self.datacube_pars)

        return

    def _setup_inclined_exp(self) -> None:
        '''
        Setup a mock observation using an inclined exponential disk
        '''

        self.datacube_pars = {
            **self._base_datacube_pars, **self._datacube_exp_pars
        }

        return

    def _setup_basis(self) -> None:
        '''
        Setup a mock observation using a basis intensity map
        '''

        self.datacube_pars = {
            **self._base_datacube_pars, **self._datacube_basis_pars
        }

        return

# NOTE: This is kept for backwards compatability
def setup_likelihood_test(true_pars: dict, datacube_pars: dict) -> Union[DataCube, VelocityMap, gs.Image]:
    '''
    Setup a test datacube, velocity map, and intensity map given
    true physical parameters and some datacube_parsprameters

    true_pars: dict
        A dictionary of parameters needed to construct a true image
    datacube_pars: dict
        A dictionary of all parameters needed for datacube construction

        Needed fields in datacube_pars:
        [Nx, Ny, pix_scale, true_flux, true_hlr, r_unit,
         v_unit, v_model]

        Need one of the following in datacube_pars:
        [sky_sigma, s2n]
    '''

    # first, make copies so we don't modify the input dicts
    true_pars = true_pars.copy()
    datacube_pars = datacube_pars.copy()

    # setup mock emission line, w/ halpha & CWI defaults
    if 'wavelength' in datacube_pars:
        wavelength = datacube_pars['wavelength']
    else:
        wavelength = 656.28 # nm
    if 'R' in datacube_pars:
        R = datacube_pars['R']
    else:
        R = 5000.
    if 'z' in datacube_pars:
        z = datacube_pars['z']
    else:
        z = 0.3
    width = 1 # nm
    lines = [setup_simple_emission_line(
        wavelength, Unit('nm'), R, z, width
        )]


    if ('sky_sigma' not in datacube_pars) and ('s2n' not in datacube_pars):
        raise KeyError('Must pass one of sky_sigma or s2n!')

    if ('sky_sigma' in datacube_pars) and ('s2n' in datacube_pars):
        raise KeyError('Can only pass one of sky_sigma and s2n!')

    if 's2n' in datacube_pars:
        sky_sigma = datacube_pars['true_flux'] / datacube_pars['s2n']
        datacube_pars['sky_sigma'] = sky_sigma

    # setup mock bandpasses
    throughput = 1.
    zp = 30.
    lblue = (wavelength * (1.+z)) - (width / 2.)
    lred  = (wavelength * (1.+z)) + (width / 2.)
    dlam = 0.1 # nm
    bandpasses = cube.setup_simple_bandpasses(
        lblue, lred, dlam, throughput=throughput, zp=zp, unit='nm'
        )

    # setup datacube pars
    pars = CubePars({
        'pix_scale': datacube_pars['pix_scale'],
        'bandpasses': bandpasses,
        'emission_lines': lines
    })

    # setup blank datacube
    # assert (width % dlam) == 0 # doesn't work bc of float issues...
    Nspec = int(width / dlam)
    shape = (Nspec, datacube_pars['Nx'], datacube_pars['Ny'])
    data = np.zeros(shape)
    datacube = DataCube(data, pars=pars)

    # now fill datacube slices
    datacube, vmap, true_im = _fill_test_datacube(
        datacube, true_pars, datacube_pars
        )

    return datacube, vmap, true_im

def _fill_test_datacube(datacube: DataCube, true_pars: dict, pars: dict) -> Union[DataCube, VelocityMap, gs.Image]:
    '''
    TODO: Restructure to allow for more general truth input
    '''

    Nspec, Nx, Ny = datacube.shape

    # make true light profile an exponential disk
    imap_pars = {
        'flux': pars['true_flux'],
        'hlr': pars['true_hlr'],
    }

    try:
        imap_type = pars['imap_type']
    except KeyError:
        imap_type = 'inclined_exp'

    try:
        use_basis = pars['use_basis_as_truth']
    except KeyError:
        use_basis = False

    if 'psf' in pars:
        psf = pars['psf']
    else:
        psf = None

    # a slight abuse of API call here, passing a dummy datacube to
    # instantiate the desired type as truth (likely inclined exp)
    imap = intensity.build_intensity_map(imap_type, datacube, imap_pars)
    true_im = imap.render(true_pars, datacube, pars)

    # TODO: TESTING!!!
    # This alows us to draw the test datacube from basis instead
    if use_basis is True:
        print('WARNING: Using basis for true image as test.')
        print('WARNING: Make sure this is intentional!')

        # what we're doing here is making a dummy datacube with the truth
        # image that we then pass to the imap functions for fitting to the
        # desired basis funcs, so we just need a single dummy lambda slice
        basis_pars = CubePars({
            'pix_scale': datacube.pix_scale,
            'bandpasses': [datacube.pars['bandpasses'][0]],
            'emission_lines': datacube.pars['emission_lines']
        })

        basis_datacube = DataCube(
            data=true_im,
            pars=basis_pars
            )

        basis_type = pars['basis_type']
        kwargs = pars['basis_kwargs']
        basis_imap = intensity.BasisIntensityMap(
            basis_datacube, basis_type, basis_kwargs=kwargs
            )

        # Now make new truth image from basis MLE fit
        true_im = basis_imap.render(true_pars, basis_datacube, pars)

    vel_pars = {}
    for name in true_pars.keys():
        vel_pars[name] = true_pars[name]
    try:
        vel_pars['v_unit'] = pars['v_unit']
    except KeyError:
        vel_pars['v_unit'] = Unit('km/s')
    try:
        vel_pars['r_unit'] = pars['r_unit']
    except KeyError:
        vel_pars['r_unit'] = Unit('kpc')

    vmap = VelocityMap(pars['v_model'], vel_pars)

    X, Y = utils.build_map_grid(Nx, Ny)
    Vnorm = vmap('obs', X, Y, normalized=True)

    # We use this one for the return map
    V = vmap('obs', X, Y)

    data = datacube.data
    sed = datacube.get_sed()
    lambdas = datacube.lambdas

    # numba won't allow a scipy.interp1D object
    sed_array = np.array([sed.x, sed.y])

    # cont_array
    if 'continuum_template' in pars:
        continuum = pars['continuum_template']
    else:
        continuum = None

    sky_sig = pars['sky_sigma']

    for i in range(Nspec):
        zfactor = 1. / (1 + Vnorm)

        obs_array = DataCubeLikelihood._compute_slice_model(
            lambdas[i], sed_array, zfactor, true_im, continuum,
            psf=psf, pix_scale=pars['pix_scale']
            )

        obs_im = gs.Image(obs_array, scale=pars['pix_scale'])

        noise = gs.GaussianNoise(sigma=sky_sig)
        obs_im.addNoise(noise)

        data[i,:,:] = obs_im.array

    # datacube class won't let us modify data, so just create a new one
    datacube = DataCube(data, pars=datacube.pars)

    # set weight maps according to added noise
    datacube.set_weights(1. / sky_sig)
    datacube.set_masks(0)

    return datacube, V, true_im

def setup_simple_CWI_datacube(Nx=30, Ny=30):
    '''
    Make a very simple mock datacube to use for testing
    MCMC runs on simulations, tuned to instrumental parameters
    for the Cosmic Web Imager (CWI)

    Uses halpha emission line
    '''

    # setup mock emission line
    halpha = 656.28 # nm
    R = 5000.
    z = 0.3
    width = 1 # nm
    lines = [setup_simple_emission_line(
        halpha, Unit('nm'), R, z, width
        )]

    # setup datacube pars
    pix_scale = 0.
    pars = CubePars()

    # TODO: finish!
    # ...

    # return mock_datacube
    return

def setup_simple_emission_line(wavelength, unit, R, z, width,
                               Nbins=100):
    '''
    wavelength: float
        Emission line wavelength in a vacuum
    unit: astropy.units.Unit
        The unit of the line wavelength
    R: float
        The spectral resolution of the instrument
    z: float
        The observed redshift of the emission line
    width: float
        The width of the line SED in given lambda units
    Nbins: int
        The number of bins in the line SED

    returns: EmissionLine
        An EmissionLine object that stores all metadata
        for the line, including a simple SED
    '''

    sed_resolution = width / Nbins
    obs_line = wavelength * (1.+z)

    lblue = obs_line - width/2
    lred  = obs_line + width/2

    line_pars = {
        'value': wavelength,
        'R': R,
        'z': z,
        'unit': unit
    }

    sed_pars = {
        'lblue': lblue,
        'lred': lred,
        'resolution': sed_resolution,
        'unit': unit
    }

    return EmissionLine(line_pars, sed_pars)
