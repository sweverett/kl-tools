import numpy as np
from astropy.units import Unit
import galsim as gs

from emission import EmissionLine
import cube
from cube import CubePars, DataCube
from velocity import VelocityMap
from likelihood import DataCubeLikelihood
import intensity
import utils

import ipdb
import pudb

def setup_likelihood_test(true_pars, meta):
    '''
    Setup a test datacube, velocity map, and intensity map given
    true physical parameters and some metaprameters

    true_pars: dict
        A dictionary of {'parameter': val} pairs for parameters
        needed to construct a model datacube
    meta: MetaPars or MCMCPars
        A parameter object that contains all needed meta pars
        for datacube construction

        Needed fields in meta:
        [Nx, Ny, pix_scale, true_flux, true_hlr, r_unit,
         v_unit, v_model]

        Need one of the following in meta:
        [sky_sigma, s2n]
    '''

    # setup mock emission line, w/ halpha & CWI defaults
    if 'wavelength' in meta:
        wavelength = meta['wavelength']
    else:
        wavelength = 656.28 # nm
    if 'R' in meta:
        R = meta['R']
    else:
        R = 5000.
    if 'z' in meta:
        z = meta['z']
    else:
        z = 0.3
    width = 1 # nm
    lines = [setup_simple_emission_line(
        wavelength, Unit('nm'), R, z, width
        )]

    if ('sky_sigma' not in meta) and ('s2n' not in meta):
        raise KeyError('Must pass one of sky_sigma or s2n!')

    if ('sky_sigma' in meta) and ('s2n' in meta):
        raise KeyError('Can only pass one of sky_sigma and s2n!')

    if 's2n' in meta:
        sky_sigma = meta['intensity']['true_flux'] / meta['s2n']
        meta['sky_sigma'] = sky_sigma

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
        'pix_scale': meta['pix_scale'],
        'bandpasses': bandpasses,
        'emission_lines': lines
    })

    # setup blank datacube
    # assert (width % dlam) == 0 # doesn't work bc of float issues...
    Nspec = int(width / dlam)
    shape = (Nspec, meta['Nx'], meta['Ny'])
    data = np.zeros(shape)
    datacube = DataCube(data, pars=pars)

    # now fill datacube slices
    datacube, vmap, true_im = _fill_test_datacube(
        datacube, true_pars, meta
        )

    return datacube, vmap, true_im

def _fill_test_datacube(datacube, true_pars, pars):
    '''
    TODO: Restructure to allow for more general truth input
    '''

    Nspec, Nx, Ny = datacube.shape

    # make true light profile an exponential disk
    imap_pars = {
        'flux': pars['intensity']['true_flux'],
        'hlr': pars['intensity']['true_hlr'],
    }

    imap_type = pars['intensity']['type']
    use_basis = pars['intensity']['use_basis_as_truth']

    if 'psf' in pars:
        psf = pars['psf']
    else:
        psf = None

    # a slight abuse of API call here, passing a dummy datacube to
    # instantiate the desired type as truth (likely inclined exp)
    imap = intensity.build_intensity_map(imap_type, datacube, imap_pars)
    true_im = imap.render(true_pars, datacube, pars)

    # TODO: TESTING!!!
    # This alows us to draw the test datacube from shapelets instead
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

        basis_type = pars['intensity']['basis']
        kwargs = pars['intensity']['basis_kwargs']
        shapelet_imap = intensity.BasisIntensityMap(
            basis_datacube, basis_type, basis_kwargs=kwargs
            )

        # Now make new truth image from shapelet MLE fit
        true_im = shapelet_imap.render(true_pars, basis_datacube, pars)

    vel_pars = {}
    for name in true_pars.keys():
        vel_pars[name] = true_pars[name]
    vel_pars['v_unit'] = pars['v_unit']
    vel_pars['r_unit'] = pars['r_unit']

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

    for i in range(Nspec):
        zfactor = 1. / (1 + Vnorm)

        obs_array = DataCubeLikelihood._compute_slice_model(
            lambdas[i], sed_array, zfactor, true_im, continuum,
            psf=psf, pix_scale=pars['pix_scale']
            )

        obs_im = gs.Image(obs_array, scale=pars['pix_scale'])

        noise = gs.GaussianNoise(sigma=pars['sky_sigma'])
        obs_im.addNoise(noise)

        data[i,:,:] = obs_im.array

    # datacube class won't let us modify data, so just create a new one
    datacube = DataCube(data, pars=datacube.pars)

    # set weight maps according to added noise
    datacube.set_weights(1. / pars['sky_sigma'])
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

    return mock_datacube

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
