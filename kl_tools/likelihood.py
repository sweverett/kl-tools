from abc import abstractmethod
import numpy as np
import os
from time import time
from scipy.interpolate import interp1d
import scipy
from scipy.sparse import identity
from astropy.units import Unit
from argparse import ArgumentParser
import galsim as gs
from galsim.angle import Angle, radians
import matplotlib.pyplot as plt
from numba import njit

import utils
import priors
import intensity
from parameters import Pars
# import parameters
from velocity import VelocityMap
from cube import DataVector, DataCube

import pudb

# TODO: Make LogLikelihood a base class w/ abstract call,
# and make current implementation for IFU

parser = ArgumentParser()

parser.add_argument('--show', action='store_true', default=False,
                    help='Set to show test plots')
parser.add_argument('--test', action='store_true', default=False,
                    help='Set to run tests')

# TODO: rename!
class LogBase(object):
    '''
    Base class for a probability dist that holds the meta and
    sampled parameters
    '''

    def __init__(self, parameters, datavector):
        '''
        parameters: Pars
            Pars instance that holds all parameters needed for MCMC
            run, including SampledPars and MetaPars
        datavector: DataCube, etc.
            Arbitrary data vector that subclasses from DataVector.
            If DataCube, truncated to desired lambda bounds
        '''

        utils.check_type(parameters, 'pars', Pars)
        utils.check_type(datavector, 'datavector', DataVector)

        self.parameters = parameters
        self.datavector = datavector

        return

    @property
    def meta(self):
        '''
        While this is the default name for the meta parameters, we still
        allow for "pars" instead to be consistent with the older version
        '''
        return self.parameters.meta

    @property
    def pars(self):
        '''
        To be consistent with the older version of the code, we allow for
        the metaparameters simply as "pars" after instantiation
        '''
        return self.parameters.meta

    @property
    def sampled(self):
        return self.parameters.sampled

    @property
    def pars_order(self):
        return self.parameters.sampled.pars_order

    def pars2theta(self, pars):
        return self.sampled.pars2theta(pars)

    def theta2pars(self, theta):
        return self.sampled.theta2pars(theta)

    @abstractmethod
    def __call__(self):
        '''
        Must be implemented in actual classes
        '''
        pass

class LogPosterior(LogBase):
    '''
    Class to evaluate the log posterior of a datacube
    with the log prior and log likelihood constructed given
    the passed sampled & meta parameters
    '''

    def __init__(self, parameters, datavector):
        '''
        parameters: Pars
            Pars instance that holds all parameters needed for MCMC
            run, including SampledPars and MetaPars
        datavector: DataCube, etc.
            Arbitrary data vector that subclasses from DataVector.
            If DataCube, truncated to desired lambda bounds
        '''

        super(LogPosterior, self).__init__(parameters, datavector)

        self.log_prior = LogPrior(parameters)
        self.log_likelihood = LogLikelihood(parameters, datavector)

        self.ndims = len(parameters.sampled)

        return

    def blob(self, prior, likelihood):
        '''
        Set what the blob returns

        For the base class, this will just be the (prior, likelihood)
        tuple
        '''

        return (prior, likelihood)

    def __call__(self, theta, data, pars):
        '''
        Natural log of the posterior dist. Target function for chosen
        sampler

        theta: list
            Sampled parameters. Order defined in self.pars_order
        data: DataCube, etc.
            Arbitrary data vector. If DataCube, truncated
            to desired lambda bounds
        '''

        logprior = self.log_prior(theta)

        if logprior == -np.inf:
            return -np.inf, self.blob(-np.inf, -np.inf)

        else:
            loglike = self.log_likelihood(theta, data)

        return logprior + loglike, self.blob(logprior, loglike)

class LogPrior(LogBase):
    def __init__(self, parameters):
        '''
        pars: Pars
            A parameters.Pars object, containing both
            sampled & meta parameters
        '''

        # Can't use parent constructor as the prior doesn't
        # have access to the datavector
        utils.check_type(parameters, 'pars', Pars)
        self.parameters = parameters

        self.priors = parameters.meta['priors']

        return

    def __call__(self, theta):
        '''
        theta: list
            Sampled MCMC parameters
        '''

        pars_order = self.parameters.sampled.pars_order

        logprior = 0

        for name, prior in self.priors.items():
            indx = pars_order[name]
            logprior += prior(theta[indx], log=True)

        return logprior

class LogLikelihood(LogBase):

    def __init__(self, parameters, datavector):
        '''
        parameters: Pars
            Pars instance that holds all parameters needed for MCMC
            run, including SampledPars and MetaPars
        datavector: DataCube, etc.
            Arbitrary data vector that subclasses from DataVector.
            If DataCube, truncated to desired lambda bounds
        '''

        super(LogLikelihood, self).__init__(parameters, datavector)

        # Sometimes we want to marginalize over parts of the posterior explicitly
        self._setup_marginalization(self.meta)

        return

    def _setup_marginalization(self, pars):
        '''
        Check to see if we need to sample from a marginalized posterior

        pars: MetaPars
            Dictionary full of run parameters
        '''

        # TODO: May want something more general in the future, but for now
        #the only likely candidate is the intensity map
        # self.marginalize = {}

        # names = ['intensity']
        # for name in names:
        #     if hasattr(pars, f'marginalize_{name}'):
        #         self.marginalize[name] = pars[f'marginalize_{name}']
        # else:
        #     self.marginalize[name] = False

        # simple version:
        key = 'marginalize_intensity'
        if hasattr(pars, key):
            self.marginalize_intensity = pars[key]
        else:
            self.marginalize_intensity = False

        return

    def __call__(self, theta, datavector):
        '''
        Do setup and type / sanity checking here
        before calling the abstract method _loglikelihood,
        which will have a different implementation for each class

        theta: list
            Sampled parameters. Order defined in self.pars_order
        datavector: DataCube, etc.
            Arbitrary data vector that subclasses from DataVector.
            If DataCube, truncated to desired lambda bounds
        '''

        # unpack sampled params
        theta_pars = self.theta2pars(theta)

        model_datacube, v_array, i_array = self._setup_model_datacube(
            theta_pars, datavector
            )

        # numba can't handle interp tables, so we do it ourselves
        sed_array = self._setup_sed(self.meta)

        # NOTE: This doesn't currently work with numba
        inv_cov = self._setup_inv_cov_matrix()

        # if we are computing the marginalized posterior over intensity
        # map parameters, then we need to scale this likelihood by a
        # determinant factor
        if self.marginalize_intensity is True:
            log_det = self._compute_log_det(imap)
        else:
            log_det = 1.

        return (-0.5 * log_det) + self._log_likelihood(
            datavector, model_datacube, inv_cov
            )

    def _compute_log_det(self, imap):
        # TODO: it would be better to pass inv_cov directly,
        # but for now we are only using diagonal inv_cov matrices
        # anyway
        # log_det = imap.fitter.compute_marginalization_det(inv_cov=inv_cov, log=True)
        log_det = imap.fitter.compute_marginalization_det(pars=self.pars, log=True)

        if log_det == 0:
            print('Warning: determinant is 0. Cannot compute ' +\
                'marginalized posterior over intensity basis functions')

        return log_det

    # @abstractmethod
    # def _log_likelihood(theta, datavector, pars={}):
    #     '''
    #     Natural log of the likelihood. Target function for chosen
    #     sampler

    #     theta: list
    #         Sampled parameters, order defined by pars_order
    #     datavector: DataCube, etc.
    #         Arbitrary data vector that subclasses from DataVector.
    #         If DataCube, truncated to desired lambda bounds
    #     pars: dict
    #         Dictionary of any other parameters needed to evaluate
    #         the likelihood
    #     '''
    #     pass

    @classmethod
    def _log_likelihood(cls, datavector, model, inv_cov):
        '''
        datavector: DataCube, etc.
            The datavector that is compared against the model
        model: DataCube, etc.
            The model datacube object, truncated to desired lambda bounds
        inv_cov: np.ndarray, sp.sparse.spmatrix
            A (Nx*Ny, Nx*Ny) inverse covariance matrix for the image pixels
        '''

        Nspec = datavector.Nspec
        Nx, Ny = datavector.Nx, datavector.Ny

        loglike = 0

        # can figure out how to remove this for loop later
        # will be fast enough with numba anyway
        for i in range(Nspec):

            diff = (datavector.slice(i) - model.slice(i)).reshape(Nx*Ny)
            chi2 = diff.T.dot(inv_cov.dot(diff))

            loglike += -0.5*chi2

        # Actually slower due to extra matrix evals...
        # diff_2 = (datavector.data - model.data).reshape(Nspec, Nx*Ny)
        # chi2_2 = diff_2.dot(inv_cov.dot(diff_2.T))
        # loglike2 = -0.5*chi2_2.trace()

        return loglike

    def _setup_model_datacube(self, theta_pars, datavector):
        '''
        theta_pars: dict
            Dictionary of sampled pars
        datavector: DataCube, etc.
            Arbitrary data vector that subclasses from DataVector.
            If DataCube, truncated to desired lambda bounds
        '''

        # Datavector may not be a datacube itself
        Nx, Ny = self.pars['Nx'], self.pars['Ny']
        Nspec = self.pars['Nspec']

        lambdas = np.array(self.pars['lambdas'])

        # create grid of pixel centers in image coords
        X, Y = utils.build_map_grid(Nx, Ny)

        # create 2D velocity & intensity maps given sampled transformation
        # parameters
        vmap = self._setup_vmap(theta_pars, self.meta)
        imap = self._setup_imap(theta_pars, datavector, self.meta)

        # evaluate maps at pixel centers in obs plane
        if 'use_numba' in self.pars:
            use_numba = self.pars['use_numba']
        else:
            use_numba = False

        v_array = vmap(
            'obs', X, Y, normalized=True, use_numba=use_numba
            )

        i_array = imap.render(theta_pars, datavector, self.meta)

        sed_array = self._setup_sed(self.meta)

        # create datacube
        shape = (Nspec, Nx, Ny)

        model_datacube = self._construct_model_datacube(
            shape, lambdas, v_array, i_array
            )

        return model_datacube, v_array, i_array

    @classmethod
    def _setup_vmap(cls, theta_pars, pars, model_name='default'):
        '''
        theta_pars: dict
        A dict of the sampled mcmc params for both the velocity
        map and the tranformation matrices
        pars: dict
        A dict of anything else needed to compute the posterior
        model_name: str
        The model name to use when constructing the velocity map
        '''

        vmodel = theta_pars

        for name in ['r_unit', 'v_unit']:
            if name in pars:
                vmodel[name] = Unit(pars[name])
            else:
                raise AttributeError(f'pars must have a value for {name}!')

        return VelocityMap(model_name, vmodel)

    @classmethod
    def _setup_imap(cls, theta_pars, datacube, pars):
        '''
        theta_pars: dict
            A dict of the sampled mcmc params for both the velocity
            map and the tranformation matrices
        pars: dict
            A dict of anything else needed to compute the posterior
        model_name: str
            The model name to use when constructing the velocity map
        '''

        imap_pars = pars['intensity'].copy()
        imap_type = imap_pars['type']
        del imap_pars['type']

        return intensity.build_intensity_map(imap_type, datacube, imap_pars)

    def _construct_model_datacube(self, shape, lambdas, v_array, i_array):
        '''
        Create the model datacube from model slices, using the evaluated
        velocity and intensity maps, SED, etc.

        shape: tuple
            A tuple of format (Nspec, Nx, Ny)
        lambdas: list of tuples
            A list of (lambda_b, lambda_r) tuples for each datacube slice
        v_array: np.array (2D)
            The vmap evaluated at image pixel positions for sampled pars.
            (Must be normalzied)
        i_array: np.array (2D)
            The imap evaluated at image pixel positions for sampled pars
        '''

        Nspec, Nx, Ny = shape

        data = np.zeros(shape)

        sed_array = self._setup_sed(self.meta)

        for i in range(Nspec):
            zfactor = 1. / (1 + v_array)

            obs_array = self._compute_slice_model(
                lambdas[i], sed_array, zfactor, i_array
            )

            # NB: here you could do something fancier, such as a
            # wavelength-dependent PSF
            # obs_im = gs.Image(obs_array, scale=pars['pix_scale'])
            # obs_im = ...

            data[i,:,:] = obs_array

        pix_scale = self.meta['pix_scale']
        bandpasses = self.meta['bandpasses']
        model_datacube = DataCube(
            data=data, bandpasses=bandpasses, pix_scale=pix_scale
        )

        return model_datacube

    @classmethod
    def _compute_slice_model(cls, lambdas, sed, zfactor, imap):
        '''
        Compute datacube slice given lambda range, sed, redshift factor
        per pixel, and the intemsity map

        lambdas: tuple
            The wavelength tuple (lambda_blue, lambda_red) for a
            given slice
        sed: np.ndarray
            A 2D numpy array with axis=0 being the lambda values of
            a 1D interpolation table, with axis=1 being the corresponding
            SED values
        zfactor: np.ndarray (2D)
            A 2D numpy array corresponding to the (normalized by c) velocity
            map at each pixel
        imap: np.ndarray (2D)
            A 2D numpy array corresponding to the source intensity map
            at the emission line
        '''

        lblue, lred = lambdas[0], lambdas[1]

        # Get mean SED vlaue in slice range
        # NOTE: We do it this way as numba won't allow
        #       interp1d objects
        sed_b = cls._interp1d(sed, lblue*zfactor)
        sed_r = cls._interp1d(sed, lred*zfactor)

        # Numba won't let us use np.mean w/ axis=0
        mean_sed = (sed_b + sed_r) / 2.
        int_sed = (lred - lblue) * mean_sed
        model = imap * int_sed

        return model

    @classmethod
    def _setup_sed(cls, meta):
        '''
        numba can't handle most interpolators, so create
        a numpy one
        '''

        sed = meta['sed']

        return np.array([sed.x, sed.y])

    @classmethod
    def _interp1d(cls, table, values, kind='linear'):
        '''
        Interpolate table(value)

        table: np.ndarray
            A 2D numpy array with axis=0 being the x-values and
            axis=1 being the function evaluated at x
        values: np.array
            The values to interpolate the table on
        '''

        if kind == 'linear':
            # just use numpy linear interpolation, as it works with numba
            interp = np.interp(values, table[0], table[1])
        else:
            raise ValueError('Non-linear interpolations not yet implemented!')

        return interp

    def _setup_inv_cov_matrix(self):
        '''
        Build covariance matrix for slice images

        pars: dict
            dictionary containing parameters needed to build cov matrix

        # TODO: For now, a single cov matrix for each slice. However, this could be
                generalized to a list of cov matrices
        '''

        Nx, Ny = self.pars['Nx'], self.pars['Ny']
        Npix = Nx*Ny

        # TODO: For now, treating pixel covariance as diagonal
        #       and uniform for a given slice
        sigma = self.pars['cov_sigma']

        # full matrix, but very inefficient for a diagonal
        # cov matrix
        # inv_cov = (1./sigma)**2 * np.identity(Npix)

        # uses scipy sparse matrices
        inv_cov = (1./sigma)**2 * identity(Npix)

        return inv_cov

def _setup_test_sed(pars):
    '''
    Build interpolation table for normalized emission line SED

    pars: dict contianing emission line parameters

    TODO: See if constructing the SED once and passing it in args
          will cause problems for optimizations
    '''

    start = pars['sed_start']
    end = pars['sed_end']
    res = pars['sed_resolution']

    lambdas = np.arange(start, end+res, res)

    # Model emission line SED as gaussian
    mu  = pars['line_value']
    std = pars['line_std']
    unit = pars['line_unit']

    norm = 1. / (std * np.sqrt(2.*np.pi))
    gauss = norm * np.exp(-0.5*(lambdas - mu)**2 / std**2)

    sed = interp1d(lambdas, gauss)

    return sed

def setup_likelihood_test(true_pars, pars, shape, lambdas):

    throughput = pars['bandpass_throughput']
    unit = pars['bandpass_unit']
    zp = pars['bandpass_zp']

    bandpasses = []
    for l1, l2 in lambdas:
        bandpasses.append(gs.Bandpass(
            throughput, unit, blue_limit=l1, red_limit=l2, zeropoint=zp
            ))

    assert shape[0] == len(bandpasses)

    sed = _setup_test_sed(pars)

    datacube, vmap, true_im = _setup_test_datacube(
        shape, lambdas, bandpasses, sed, true_pars, pars
        )

    return datacube, sed, vmap, true_im

def _setup_test_datacube(shape, lambdas, bandpasses, sed, true_pars, pars):
    '''
    TODO: Restructure to allow for more general truth input
    '''

    Nspec, Nx, Ny = shape[0], shape[1], shape[2]

    imap_pars = {
        'flux': pars['true_flux'],
        'hlr': pars['true_hlr']
    }

    # a slight abuse of API call here, passing a dummy datacube to
    # instantiate an inclined exponential as truth
    dc = DataCube(shape=shape, bandpasses=bandpasses)
    imap = intensity.build_intensity_map('inclined_exp', dc, imap_pars)
    true_im = imap.render(true_pars, dc, pars)

    # TODO: TESTING!!!
    #This alows us to draw the test datacube from shapelets instead
    if pars['intensity']['type'] == 'basis':
        try:
            use_basis = pars['intensity']['use_basis_as_truth']

            if use_basis is True:
                print('WARNING: Using basis for true image as test')
                ps = pars['pix_scale']
                dc = DataCube(
                    shape=(1,Nx,Ny), bandpasses=[bandpasses[0]], data=true_im, pix_scale=ps
                    )

                basis_type = pars['intensity']['basis_type']
                kwargs = pars['intensity']['basis_kwargs']
                shapelet_imap = intensity.BasisIntensityMap(
                    dc, basis_type, basis_kwargs=kwargs)

                # Now make new truth image from shapelet MLE fit
                true_im = shapelet_imap.render(true_pars, dc, pars)

        except KeyError:
            pass

    vel_pars = {}
    for name in true_pars.keys():
        vel_pars[name] = true_pars[name]
    vel_pars['v_unit'] = pars['v_unit']
    vel_pars['r_unit'] = pars['r_unit']

    vmap = VelocityMap('default', vel_pars)

    X, Y = utils.build_map_grid(Nx, Ny)
    Vnorm = vmap('obs', X, Y, normalized=True)

    # We use this one for the return map
    V = vmap('obs', X, Y)

    data = np.zeros(shape)

    # numba won't allow a scipy.interp1D object
    sed_array = np.array([sed.x, sed.y])

    for i in range(Nspec):
        zfactor = 1. / (1 + Vnorm)

        obs_array = LogLikelihood._compute_slice_model(
            lambdas[i], sed_array, zfactor, true_im
            )

        obs_im = gs.Image(obs_array, scale=pars['pix_scale'])

        noise = gs.GaussianNoise(sigma=pars['cov_sigma'])
        obs_im.addNoise(noise)

        data[i,:,:] = obs_im.array

    pix_scale = pars['pix_scale']
    datacube = DataCube(
        data=data, bandpasses=bandpasses, pix_scale=pix_scale
        )

    return datacube, V, true_im

def setup_test_pars(nx, ny):
    '''
    Initialize a test set of true_pars and pars for
    reasonable values

    nx: int
        Size of image on x-axis
    ny: int
        Size of image on y-axis
    '''

    true_pars = {
        'g1': 0.05,
        'g2': -0.025,
        'theta_int': np.pi / 3,
        'sini': 0.8,
        'v0': 10.,
        'vcirc': 200,
        'rscale': 5,
    }

    # additional args
    halpha = 656.28 # nm
    R = 5000.
    z = 0.3
    pars = {
        'true_flux': 1e5, # counts
        'true_hlr': 5, # pixels
        'v_unit': Unit('km / s'),
        'r_unit': Unit('kpc'),
        'z': z,
        'spec_resolution': R,
        'line_std': halpha * (1.+z) / R, # emission line SED std; nm
        'line_value': 656.28, # emission line SED std; nm
        'line_unit': Unit('nm'),
        'sed_start': 650,
        'sed_end': 660,
        'sed_resolution': 0.025,
        'sed_unit': Unit('nm'),
        'cov_sigma': 1, # pixel counts; dummy value
        'bandpass_throughput': '.2',
        'bandpass_unit': 'nm',
        'bandpass_zp': 30,
        'use_numba': False,
        'pix_scale': 1.,
        'priors': {
            'g1': priors.GaussPrior(0., 0.3),#, clip_sigmas=2),
            'g2': priors.GaussPrior(0., 0.3),#, clip_sigmas=2),
            'theta_int': priors.UniformPrior(0., np.pi),
            'sini': priors.UniformPrior(0., 1.),
            'v0': priors.UniformPrior(0, 20),
            'vcirc': priors.GaussPrior(200, 20, zero_boundary='positive'),# clip_sigmas=2),
            'rscale': priors.UniformPrior(0, 10),
        },
        'intensity': {
            'type': 'inclined_exp',

        },
        }
    pars['Nx'] = nx
    pars['Ny'] = ny

    return true_pars, pars

def main(args):

    show = args.show

    outdir = os.path.join(utils.TEST_DIR, 'likelihood')
    utils.make_dir(outdir)

    true_pars = {
        'g1': 0.1,
        'g2': 0.2,
        'sini': 0.75,
        'theta_int': np.pi / 3,
        'v0': 10,
        'vcirc': 200,
        'rscale': 5,
    }

    # additional args needed for prior / likelihood evaluation
    pars = {
        'Nx': 30, # pixels
        'Ny': 30, # pixels
        'pix_scale': 1., # arcsec / pix
        'true_flux': 1e4, # counts
        'true_hlr': 3, # pixels
        'v_unit': Unit('km / s'),
        'r_unit': Unit('kpc'),
        'line_std': 2, # emission line SED std; nm
        'line_value': 656.28, # emission line SED std; nm
        'line_unit': Unit('nm'),
        'sed_start': 655,
        'sed_end': 657.5,
        'sed_resolution': 0.025,
        'sed_unit': Unit('nm'),
        'cov_sigma': 1., # pixel counts; dummy value
        'bandpass_throughput': '0.2',
        'bandpass_unit': 'nm',
        'bandpass_zp': 30,
        'priors': {
            'g1': priors.GaussPrior(0., 0.1),# clip_sigmas=3),
            'g2': priors.GaussPrior(0., 0.1),# clip_sigmas=3),
            'theta_int': priors.UniformPrior(0., np.pi),
            'sini': priors.UniformPrior(0., 1.),
            'v0': priors.UniformPrior(0, 20),
            'vcirc': priors.GaussPrior(200, 20),
            'rscale': priors.UniformPrior(0, 10),
        },
        'intensity': {
            'type': 'inclined_exp'
        },
        'psf': gs.Gaussian(fwhm=3), # fwhm in pixels
    }

    li, le, dl = 654, 657, 1
    lambdas = [(l, l+dl) for l in range(li, le, dl)]

    Nx, Ny = pars['Nx'], pars['Ny']
    Nspec = len(lambdas)
    shape = (Nspec, Nx, Ny)

    print('Setting up test datacube and true Halpha image')
    datacube, sed, vmap, true_im = setup_likelihood_test(
        true_pars, pars, shape, lambdas
        )

    pars['sed'] = sed

    outfile = os.path.join(outdir, 'true-im.png')
    print(f'Saving true intensity profile in obs plane to {outfile}')
    plt.imshow(true_im, origin='lower')
    plt.colorbar()
    plt.title('True Halpha profile in obs plane')
    plt.savefig(outfile, bbox_inches='tight')
    if show is True:
        plt.show()
    else:
        plt.close()

    outfile = os.path.join(outdir, 'vmap.png')
    print(f'Saving true vamp in obs plane to {outfile}')
    plt.imshow(vmap, origin='lower')
    plt.colorbar(label='v / c')
    plt.title('True velocity map in obs plane')
    plt.savefig(outfile, bbox_inches='tight')
    if show is True:
        plt.show()
    else:
        plt.close()

    outfile = os.path.join(outdir, 'datacube.fits')
    print(f'Saving test datacube to {outfile}')
    datacube.write(outfile)

    outfile = os.path.join(outdir, 'datacube-slice.png')
    print(f'Saving example datacube slice im to {outfile}')
    lc = Nspec // 2
    s = datacube.slices[lc]
    plt.imshow(s._data, origin='lower')
    plt.colorbar()
    plt.title(f'Test datacube slice at lambda={lambdas[lc]}')
    plt.savefig(outfile, bbox_inches='tight')
    if show is True:
        plt.show()
    else:
        plt.close()

    theta = pars2theta(true_pars)

    print('Computing log posterior for correct theta')
    ptrue = log_posterior(theta, datacube, pars)[0]
    chi2_true = -2.*ptrue / (Nx*Ny*Nspec - len(theta))
    print(f'Posterior value = {ptrue:.2f}')

    print('Computing log posterior for random scatter about correct theta')
    N = 1000
    p = []
    chi2 = []
    new_thetas = []
    radius = 0.25
    for i in range(N):
        scale = radius * np.array(theta)
        new_theta = theta + scale * np.random.rand(len(theta))
        new_thetas.append(new_theta)
        p.append(log_posterior(new_theta, datacube, pars)[0])
        chi2.append(-2.*p[i] / (Nx*Ny*Nspec - len(new_theta)))
    if N <= 10:
        print(f'Posterior values:\n{p}')

    # outfile = os.path.join(outdir, 'posterior-dist-ball.png')
    outfile = os.path.join(outdir, 'posterior-dist-ball.png')
    print('Plotting hist of reduced chi2 vs. chi2 at truth to {outfile}')
    cmin = np.min(chi2)
    cmax = np.max(chi2)
    Nbins = 20
    bins = np.linspace(cmin, cmax, num=Nbins, endpoint=True)
    plt.hist(chi2, ec='k', bins=bins)
    plt.axvline(chi2_true, ls='--', c='k', lw=2, label='Eval at true theta')
    plt.legend()
    plt.xlabel('Reduced Chi2')
    plt.ylabel('Counts')
    plt.yscale('log')
    plt.title('Reduced chi2 evaluations for a random ball centered at truth\n '
              f'with radius = {radius} * truth')
    plt.savefig(outfile, bbox_inches='tight')
    if show is True:
        plt.show()
    else:
        plt.close()

    outfile = os.path.join(outdir, 'theta-diff.png')
    print('Plotting diff between true theta and MAP')
    best = new_thetas[int(np.where(chi2 == np.min(chi2))[0])]
    plt.plot(100. * (best-theta) / theta, 'o', c='k', markersize=5)
    plt.axhline(0, c='k', ls='--', lw=2)
    xx = range(0, len(best))
    plt.fill_between(
        xx, -10*np.ones(len(xx)), 10*np.ones(len(xx)), color='gray', alpha=0.25
        )
    my_xticks = len(best)*['']
    for name, indx in PARS_ORDER.items():
        my_xticks[indx] = name
    plt.xticks(xx, my_xticks)
    plt.xlabel('theta params')
    plt.ylabel('Percent Error (MAP - true)')
    plt.title('% Error in MAP vs. true sampled params')
    plt.savefig(outfile, bbox_inches='tight')
    if show is True:
        plt.show()
    else:
        plt.close()

    return 0

if __name__ == '__main__':
    args = parser.parse_args()

    if args.test is True:
        print('Starting tests')
        rc = main(args)

        if rc == 0:
            print('All tests ran succesfully')
        else:
            print(f'Tests failed with return code of {rc}')
