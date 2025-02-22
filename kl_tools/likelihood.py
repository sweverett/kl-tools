from abc import abstractmethod
import numpy as np
from copy import deepcopy
from astropy.units import Unit
import galsim as gs

import kl_tools.utils as utils
import kl_tools.intensity as intensity
from kl_tools.parameters import Pars
from kl_tools.velocity import VelocityMap
from kl_tools.cube import DataVector, DataCube

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

        # add an internal field to the meta pars dictionary
        parameters.meta['_likelihood'] = {}

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

    def __init__(self, parameters, datavector, likelihood='default',
                 prior='default'):
        '''
        parameters: Pars
            Pars instance that holds all parameters needed for MCMC
            run, including SampledPars and MCMCPars
        datavector: DataCube, etc.
            Arbitrary data vector that subclasses from DataVector.
            If DataCube, truncated to desired lambda bounds
        likelihood: str
            Type of likelihood model to build; see LIKELIHOOD_TYPES
        prior: str
            Type of prior model to build; see PRIOR_TYPES
        '''

        super(LogPosterior, self).__init__(parameters, datavector)

        self.log_prior = build_prior_model(
            prior, parameters
            )
        self.log_likelihood = build_likelihood_model(
            likelihood, parameters, datavector
            )

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

        logpost = logprior + loglike

        # In rare cases, particularly basis function imap modeling,
        # NaN's can occur from pseudo-inverse solving. We try to catch
        # it when it happens but we keep this as a fail-safe
        if np.isnan(logpost):
            logpost = -np.inf

        return logpost, self.blob(logprior, loglike)

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

class LogCDFPrior(LogPrior):
    '''
    A prior class that returns the sampled value given defined
    cumulative probability distributions for each sampled parameter
    '''

    def __call__(self, quantile_cube):
        '''
        quantile_cube: np.ndarray
            An array of CDF quantile values for each parameter. See:
            https://johannesbuchner.github.io/UltraNest/priors.html?highlight=prior
        '''

        # prepare the output array, which has the same shape
        transformed_pars = np.empty_like(quantile_cube)

        pars_order = self.parameters.sampled.pars_order

        for name, prior in self.priors.items():
            indx = pars_order[name]
            transformed_pars[indx] = prior(
                quantile_cube[indx], log=True, quantile=True
                )

        return transformed_pars

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

        # setup model corresponding to datavector type
        model, imap = self._setup_model(
            theta_pars, datavector, return_imap=True
            )

        # NOTE: in rare cases, the model will return a NaN datacube due
        # to failure to find an inverse of the design matrix when using
        # basis function imap modeling. This is usually due to an issue
        # with sampling the scale factor, so we reject this sample
        if np.isnan(model.data).any():
            return -np.inf

        # if we are computing the marginalized posterior over intensity
        # map parameters, then we need to scale this likelihood by a
        # determinant factor
        if self.marginalize_intensity is True:
            log_det = self._compute_log_det(imap)
        else:
            log_det = 1.

        return (-0.5 * log_det) + self._log_likelihood(
            theta, datavector, model
            )

    def _call_no_args(self, theta):
        '''
        TODO: A hack to make ultranest work. Will document later if it
        is successful
        '''

        return self(theta, self.datavector)

    def _compute_log_det(self, imap):
        # TODO: Adapt this to work w/ new DataCube's w/ weight maps!
        # log_det = imap.fitter.compute_marginalization_det(inv_cov=inv_cov, log=True)
        log_det = imap.fitter.compute_marginalization_det(pars=self.meta, log=True)

        if log_det == 0:
            print('Warning: determinant is 0. Cannot compute ' +\
                'marginalized posterior over intensity basis functions')

        return log_det

    def setup_vmap(self, theta_pars):
        '''
        theta_pars: dict
            A dict of the sampled mcmc params for both the velocity
            map and the tranformation matrices
        model_name: str
            The model name to use when constructing the velocity map
        '''

        try:
            model_name = self.pars['velocity']['model']
        except KeyError:
            model_name = 'default'

        # no extras for this func
        return self._setup_vmap(theta_pars, self.meta, model_name)

    @classmethod
    def _setup_vmap(cls, theta_pars, meta_pars, model_name):
        '''
        See setup_vmap()
        '''

        vmodel = theta_pars.copy()

        for name in ['r_unit', 'v_unit']:
            if name in meta_pars['units']:
                vmodel[name] = Unit(meta_pars['units'][name])
            else:
                raise AttributeError(f'pars must have a value for {name}!')

        return VelocityMap(model_name, vmodel)

    @classmethod
    def _setup_sed(cls, theta_pars, datavector):
        '''
        numba can't handle most interpolators, so create
        a numpy one
        '''

        # sed pars that are sample-able
        _sed_pars = ['z', 'R']

        # if we are marginalizing over SED pars, modify stored
        # SED with the sample
        line_pars_update = {}

        for par in _sed_pars:
            if par in theta_pars:
                line_pars_update[par] = theta_pars[par]

        if len(line_pars_update) == 0:
            line_pars_update = None

        # NOTE: Right now, this will error if more than one
        # emission lines are stored (as we don't have a line
        # index to pass here), but can improve in future
        sed = datavector.get_sed(line_pars_update=line_pars_update)

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

    @abstractmethod
    def _log_likelihood(self, theta, datavector, model):
        '''
        Natural log of the likelihood. Target function for chosen
        sampler

        theta: list
            Sampled parameters, order defined by pars_order
        datavector: DataCube, etc.
            Arbitrary data vector that subclasses from DataVector.
            If DataCube, truncated to desired lambda bounds
        model: Datacube, etc.
            The model given theta that matches the structure of
            the input datacube
        '''
        raise NotImplementedError('Must use a LogLikelihood subclass ' +\
                                  'that implements _log_likelihood()!')

    @abstractmethod
    def _setup_model(self, theta_pars, datavector):
        '''
        A method that creates a model datacube for the given
        theta_pars draw and input datvector. Must be implemented
        for each datavector type

        theta_pars: dict
            Dictionary of sampled pars
        datavector: DataCube, etc.
            Arbitrary data vector that subclasses from DataVector.
            If DataCube, truncated to desired lambda bounds

        returns: model (Datacube, etc.)
            The model given theta that matches the structure of
            the input datacube
        '''
        raise NotImplementedError('Must use a LogLikelihood subclass ' +\
                                  'that implements _setup_model()!')

class DataCubeLikelihood(LogLikelihood):
    '''
    An implementation of a LogLikelihood for a DataCube datavector
    '''

    def _log_likelihood(self, theta, datacube, model):
        '''
        theta: list
            Sampled parameters, order defined by pars_order
        datacube: DataCube
            The datacube datavector, truncated to desired lambda bounds
        model: DataCube
            The model datacube object, truncated to desired lambda bounds
        '''

        Nspec = datacube.Nspec
        Nx, Ny = datacube.Nx, datacube.Ny
        masks = datacube.masks # a list of masks for each slice

        # a (Nspec, Nx*Ny, Nx*Ny) inverse covariance matrix for the image pixels
        inv_cov = self._setup_inv_cov_list(datacube)

        loglike = 0

        # can figure out how to remove this for loop later
        # will be fast enough with numba anyway
        for i in range(Nspec):

            msk = masks[i,:,:]
            # ...


            diff = (
                datacube.data[i] - model.data[i]
                ).reshape(Nx*Ny)
            chi2 = diff.T.dot(inv_cov[i].dot(diff))

            loglike += -0.5*chi2

        # NOTE: Actually slower due to extra matrix evals...
        # diff_2 = (datacube.data - model.data).reshape(Nspec, Nx*Ny)
        # chi2_2 = diff_2.dot(inv_cov.dot(diff_2.T))
        # loglike2 = -0.5*chi2_2.trace()

        return loglike

    def _setup_model(self, theta_pars, datacube, return_imap=False):
        '''
        Setup the model datacube given the input datacube datacube

        theta_pars: dict
            Dictionary of sampled pars
        datacube: DataCube
            Datavector datacube truncated to desired lambda bounds
        return_imap: bool
            Set to return the tuple (model, imap)
        '''

        Nx, Ny = datacube.Nx, datacube.Ny
        Nspec = datacube.Nspec

        # create grid of pixel centers in image coords
        X, Y = utils.build_map_grid(Nx, Ny)

        # create 2D velocity & intensity maps given sampled transformation
        # parameters
        vmap = self.setup_vmap(theta_pars)
        imap = self.setup_imap(theta_pars, datacube)

        try:
            use_numba = self.meta['run_options']['use_numba']
        except KeyError:
            use_numba = False

        # evaluate maps at pixel centers in obs plane
        v_array = vmap(
            'obs', X, Y, normalized=True, use_numba=use_numba
            )

        # get both the emission line and continuum image
        i_array, cont_array = imap.render(
            theta_pars, datacube, self.meta, im_type='both'
            )

        model_datacube = self._construct_model_datacube(
            theta_pars, v_array, i_array, cont_array, datacube
            )

        # TODO: I don't like this implementation, but for now will handle the case in which basis function marginalization requires imap return
        if return_imap is True:
            return model_datacube, imap
        else:
            return model_datacube

    def setup_imap(self, theta_pars, datacube):
        '''
        theta_pars: dict
            A dict of the sampled mcmc params for both the velocity
            map and the tranformation matrices
        datacube: DataCube
            Datavector datacube truncated to desired lambda bounds
        '''

        # In some instances, the intensity  map will not change
        # between samples
        try:
            static_imap = self.meta['_likelihood']['static_imap']
        except KeyError:
            static_imap = False

        if static_imap is True:
            if self.meta['_likelihood']['static_imap'] is True:
                return self.meta['_likelihood']['imap']

        # if not (or it is the first sample), generate imap and then
        # check if it will be static
        imap = self._setup_imap(theta_pars, datacube, self.meta)

        # add static_imap check if not yet set
        if 'static_imap' not in self.meta['_likelihood']:
            self._check_for_static_imap(imap)

        return imap

    @classmethod
    def _setup_imap(cls, theta_pars, datacube, meta):
        '''
        See setup_imap(). Only runs if a new imap for the sample
        is needed
        '''

        # Need to check if any basis func parameters are
        # being sampled over
        pars = meta.copy_with_sampled_pars(theta_pars)

        imap_pars = deepcopy(pars['intensity'])
        imap_type = imap_pars['type']
        del imap_pars['type']

        return intensity.build_intensity_map(imap_type, datacube, imap_pars)

    def _check_for_static_imap(self, imap):
        '''
        Check if given intensity model will require a new imap for each
        sample. If not, internally store first imap and use it for
        future samples

        imap: IntensityMap
            The generated intensity map object from _setup_imap()
        '''

        try:
            self.meta['_likelihood'].update({
                'static_imap': imap.is_static,
                'imap': imap
            })
        except KeyError:
            self.meta['_likelihood'] = {
                'static_imap': imap.is_static,
                'imap': imap
            }

        return

    def _construct_model_datacube(self, theta_pars, v_array, i_array,
                                  cont_array, datacube):
        '''
        Create the model datacube from model slices, using the evaluated
        velocity and intensity maps, SED, etc.

        theta_pars: dict
            A dict of the sampled mcmc params for both the velocity
            map and the tranformation matrices
        v_array: np.array (2D)
            The vmap evaluated at image pixel positions for sampled pars.
            (Must be normalzied)
        i_array: np.array (2D)
            The imap evaluated at image pixel positions for sampled pars
        cont_array: np.array (2D)
            The imap of the fitted or modeled continuum
        datacube: DataCube
            Datavector datacube truncated to desired lambda bounds
        '''

        Nspec, Nx, Ny = datacube.shape

        data = np.zeros(datacube.shape)

        lambdas = np.array(datacube.lambdas)

        sed_array = self._setup_sed(theta_pars, datacube)

        psf = datacube.get_psf()

        # get kinematic redshift correct per imap image pixel
        zfactor = 1. / (1 + v_array)

        for i in range(Nspec):
            data[i,:,:] = self._compute_slice_model(
                lambdas[i], sed_array, zfactor, i_array, cont_array,
                pix_scale=datacube.pix_scale, psf=psf
            )

        model_datacube = DataCube(
            data=data, pars=datacube.pars
        )

        return model_datacube

    @classmethod
    def _compute_slice_model(cls, lambdas, sed, zfactor, imap, continuum,
                             pix_scale, psf=None):
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
            An array corresponding to the source intensity map
            at the emission line
        imap: np.ndarray (2D)
            An array corresponding to the continuum model
        pix_scale: float
            The image pixel scale
        psf: galsim.GSObject
            A galsim object representing the PSF to convolve by
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

        # TODO: could generalize in future, but for now assume
        #       a constant PSF for exposures
        # if (psf is not None) and (np.sum(model) != 0):
        if psf is not None:
            # plt.subplot(131)
            # plt.imshow(model, origin='lower')
            # plt.colorbar()
            # plt.title('pre-psf model')
            # This fails if the model has no flux, which
            # can happen for very wrong redshift samples
            nx, ny = imap.shape[0], imap.shape[1]
            model_im = gs.Image(model, scale=pix_scale)
            gal = gs.InterpolatedImage(model_im)
            conv = gs.Convolve([psf, gal])

            model = conv.drawImage(
                nx=ny, ny=nx, method='no_pixel', scale=pix_scale
                ).array
            # plt.subplot(132)
            # plt.imshow(pmodel, origin='lower')
            # plt.colorbar()
            # plt.title('post-psf model')
            # plt.subplot(133)
            # plt.imshow(pmodel-model, origin='lower')
            # plt.colorbar()
            # plt.title('post-pre psf model')
            # plt.show()

        # for now, continuum is modeled as lambda-independent
        # TODO: This assumes that the continuum template (if passed)
        #       is *post* psf-convolution
        if continuum is not None:
            model += continuum

        return model

    def _setup_inv_cov_list(self, datacube):
        '''
        Build inverse covariance matrices for slice images

        returns: List of (Nx*Ny, Nx*Ny) scipy sparse matrices
        '''

        # for now, we'll let each datacube class do this
        # to allow for differences in weightmap definitions
        # between experiments

        return datacube.get_inv_cov_list()

# NOTE: This is where you must register a new prior model
PRIOR_TYPES = {
    'default': LogPrior,
    'prior': LogPrior,
    'cdf_prior': LogCDFPrior
    }

def get_prior_types():
    return PRIOR_TYPES

def get_sampler_prior_type(sampler):
    sampler_prior = {
        'emcee': 'prior',
        'zeus': 'prior',
        'poco': 'prior',
        'metropolis': 'prior',
        'multinest': 'cdf_prior',
        'ultranest': 'cdf_prior'
    }

    return sampler_prior[sampler]

def build_prior_model(name, parameters):
    '''
    name: str
        Name of prior model type
    parameters: Pars
        Pars instance that holds all parameters needed for MCMC
        run, including SampledPars and MetaPars
    '''

    name = name.lower()

    if name in PRIOR_TYPES.keys():
        # User-defined input construction
        prior = PRIOR_TYPES[name](parameters)
    else:
        raise ValueError(f'{name} is not a registered prior model!')

    return prior

# NOTE: This is where you must register a new likelihood model
LIKELIHOOD_TYPES = {
    'default': DataCubeLikelihood,
    'datacube': DataCubeLikelihood
    }

def get_likelihood_types():
    return LIKELIHOOD_TYPES

def build_likelihood_model(name, parameters, datavector):
    '''
    name: str
        Name of likelihood model type
    parameters: Pars
        Pars instance that holds all parameters needed for MCMC
        run, including SampledPars and MetaPars
    datavector: DataCube, etc.
        Arbitrary data vector that subclasses from DataVector.
        If DataCube, truncated to desired lambda bounds
    '''

    name = name.lower()

    if name in LIKELIHOOD_TYPES.keys():
        # User-defined input construction
        likelihood = LIKELIHOOD_TYPES[name](parameters, datavector)
    else:
        raise ValueError(f'{name} is not a registered likelihood model!')

    return likelihood
