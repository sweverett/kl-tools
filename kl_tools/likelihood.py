from abc import abstractmethod
import numpy as np
import os
import sys
sys.path.insert(0, './grism_modules')
from copy import deepcopy
from time import time
from scipy.interpolate import interp1d
import scipy
from astropy.units import Unit
from argparse import ArgumentParser
import galsim as gs
from galsim.angle import Angle, radians
import matplotlib.pyplot as plt
from numba import njit
from mpi4py import MPI
_mpi = MPI
_mpi_comm = _mpi.COMM_WORLD
_mpi_size = _mpi_comm.Get_size()
_mpi_rank = _mpi_comm.Get_rank()

import utils
import priors
import intensity
from parameters import Pars, MetaPars
# import parameters
from velocity import VelocityMap
from cube import DataVector, DataCube
import grism as grism
import kltools_grism_module_2 as m
m.set_mpi_info(_mpi_size, _mpi_rank)

import ipdb

# TODO: Make LogLikelihood a base class w/ abstract call,
# and make current implementation for IFU

parser = ArgumentParser()

parser.add_argument('--show', action='store_true', default=False,
                    help='Set to show test plots')
parser.add_argument('--test', action='store_true', default=False,
                    help='Set to run tests')

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

    def __init__(self, parameters, datavector, likelihood='default'):
        '''
        parameters: Pars
            Pars instance that holds all parameters needed for MCMC
            run, including SampledPars and MCMCPars
        datavector: DataCube, etc.
            Arbitrary data vector that subclasses from DataVector.
            If DataCube, truncated to desired lambda bounds
        '''

        super(LogPosterior, self).__init__(parameters, datavector)

        self.log_prior = LogPrior(parameters)
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

        # setup model corresponding to datavector type
        model = self._setup_model(theta_pars, datavector)

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

        # a (Nspec, Nx*Ny, Nx*Ny) inverse covariance matrix for the image pixels
        inv_cov = self._setup_inv_cov_list(datacube)

        loglike = 0

        # can figure out how to remove this for loop later
        # will be fast enough with numba anyway
        for i in range(Nspec):

            diff = (datacube.data[i] - model.data[i]).reshape(Nx*Ny)
            chi2 = diff.T.dot(inv_cov[i].dot(diff))

            loglike += -0.5*chi2

        # NOTE: Actually slower due to extra matrix evals...
        # diff_2 = (datacube.data - model.data).reshape(Nspec, Nx*Ny)
        # chi2_2 = diff_2.dot(inv_cov.dot(diff_2.T))
        # loglike2 = -0.5*chi2_2.trace()

        return loglike

    def _setup_model(self, theta_pars, datacube):
        '''
        Setup the model datacube given the input datacube datacube

        theta_pars: dict
            Dictionary of sampled pars
        datacube: DataCube
            Datavector datacube truncated to desired lambda bounds
        '''

        Nx, Ny = datacube.Nx, datacube.Ny
        Nspec = datacube.Nspec

        # create grid of pixel centers in image coords
        X, Y = utils.build_map_grid(Nx, Ny)

        # create 2D velocity & intensity maps given sampled transformation
        # parameters
        vmap = self.setup_vmap(theta_pars)
        imap = self.setup_imap(theta_pars, datacube)

        # TODO: temp for debugging!
        self.imap = imap

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
                psf=psf, pix_scale=datacube.pix_scale
            )

        model_datacube = DataCube(
            data=data, pars=datacube.pars
        )

        return model_datacube

    @classmethod
    def _compute_slice_model(cls, lambdas, sed, zfactor, imap, continuum,
                             psf=None, pix_scale=None):
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
        psf: galsim.GSObject
            A galsim object representing the PSF to convolve by
        pix_scale: float
            The image pixel scale. Required if convolving by PSF
        '''

        # they must come in pairs for PSF convolution
        if (psf is not None) and (pix_scale is None):
            raise Exception('Must pass a pix_scale if convovling by PSF!')

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
        if psf is not None:
            nx, ny = imap.shape[0], imap.shape[1]
            model_im = gs.Image(model, scale=pix_scale)
            gal = gs.InterpolatedImage(model_im)
            conv = gs.Convolve([psf, gal])
            model = conv.drawImage(
                nx=ny, ny=nx, method='no_pixel', scale=pix_scale
                ).array

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

class GrismLikelihood(LogLikelihood):
    '''
    An implementation of a LogLikelihood for a Grism KL measurement

    Note: a reminder of LogLikelihood interface
    method:
        - __init__(Pars:parameters, DataVector:datavector)
        x _setup_marginalization(MetaPars:pars)
        - __call__(list:theta, DataVector:datavector)
        x _compute_log_det(imap)
        - setup_vmap(dict:theta_pars, str:model_name)
        - _setup_vmap(dict:theta_pars, MetaPars:meta_pars, str:model_name)
        > _setup_sed(cls, theta_pars, datacube)
        - _interp1d(np.ndarray:table, np.ndarray:values, kind='linear')
        o _log_likelihood(list:theta, DataVector:datavector, DataCube:model)
        o _setup_model(list:theta_pars, DataVector:datavector) 

    attributes:
        From LogBase
        - self.parameters: parameters.Pars
        - self.datavector: cube.DataVector
        - self.meta/pars: parameters.MCMCPars
        - self.sampled: parameters.SampledPars
    '''
    def __init__(self, parameters, datavector):
        ''' Initialization

        Besides the usual likelihood.LogLikelihood initialization, we further
        initialize the grism.GrismModelCube object, fsor the sake of speed
        '''
        super(GrismLikelihood, self).__init__(parameters, datavector)
        # init with an empty model cube
        _lrange = self.meta['model_dimension']['lambda_range']
        _dl = self.meta['model_dimension']['lambda_res']
        Nspec = len(np.arange(_lrange[0], _lrange[1], _dl))
        Nx = self.meta['model_dimension']['Nx']
        Ny = self.meta['model_dimension']['Ny']
        # init with an empty modelcube object
        self.modelcube = grism.GrismModelCube(self.parameters, 
            datacube=np.zeros([Nspec, Nx, Ny]))
        self.set_obs_methods(datavector)
        return

    def set_obs_methods(self, datavector):
        self.modelcube.set_obs_methods(datavector.get_config_list())

    #def __call__(self, theta, datavector):
    def __call__(self, theta):
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

        #start_time = time()*1000

        # unpack sampled params
        theta_pars = self.theta2pars(theta)

        # setup model corresponding to cube.DataCube type
        #model = self._setup_model(theta_pars, datavector)
        self._setup_model(theta_pars, datavector)

        #print("---- build model | %.2f ms -----" % (time()*1000 - start_time))

        # if we are computing the marginalized posterior over intensity
        # map parameters, then we need to scale this likelihood by a
        # determinant factor
        if self.marginalize_intensity is True:
            log_det = self._compute_log_det(self.imap)
        else:
            log_det = 1.

        return (-0.5 * log_det) + self._log_likelihood(theta, datavector)

    def _log_likelihood(self, theta, datavector):
        '''
        theta: list
            Sampled parameters, order defined by pars_order
        datavector: DataVector
            The grism datavector
        model: DataCube
            The model datacube object, truncated to desired lambda bounds
        '''

        #start_time = time()*1000
        # a (Nspec, Nx*Ny, Nx*Ny) inverse covariance matrix for the image pixels
        inv_cov = self._setup_inv_cov_list(datavector)

        loglike = 0

        # can figure out how to remove this for loop later
        # will be fast enough with numba anyway
        for i in range(datavector.Nobs):

            _img, _noise = self.modelcube.observe(i, force_noise_free=True)
            diff = (datavector.get_data(i) - _img)
            chi2 = np.sum(diff**2/inv_cov[i])

            loglike += -0.5*chi2

        #print("---- calculate dchi2 | %.2f ms -----" % (time()*1000 - start_time))
        # NOTE: Actually slower due to extra matrix evals...
        # diff_2 = (datacube.data - model.data).reshape(Nspec, Nx*Ny)
        # chi2_2 = diff_2.dot(inv_cov.dot(diff_2.T))
        # loglike2 = -0.5*chi2_2.trace()

        return loglike

    def _setup_model(self, theta_pars, datavector):
        '''
        Setup the model datacube given the input theta_pars and datavector

        theta_pars: dict
            Dictionary of sampled pars
        datavector: DataVector
            GrismDataVector object
        '''
        Nx = self.meta['model_dimension']['Nx']
        Ny = self.meta['model_dimension']['Ny']
        model_scale = self.meta['model_dimension']['scale']

        # create grid of pixel centers in image coords
        X, Y = utils.build_map_grid(Nx, Ny, indexing='xy', scale=model_scale)

        # create 2D velocity & intensity maps given sampled transformation
        # parameters
        vmap = self.setup_vmap(theta_pars)
        imap = self.setup_imap(theta_pars, datavector)

        # TODO: temp for debugging!
        self.imap = imap

        try:
            use_numba = self.meta['run_options']['use_numba']
        except KeyError:
            use_numba = False

        # evaluate maps at pixel centers in obs plane
        v_array = vmap(
            'obs', X, Y, normalized=True, use_numba=use_numba
            )
        self.v_array = v_array
        # get both the emission line and continuum image
        _pars = self.meta.copy_with_sampled_pars(theta_pars)
        _pars['run_options']['imap_return_gal'] = True
        i_array, gal = imap.render(theta_pars, datavector, _pars)
        self.i_array = i_array
        self._construct_model_datacube(theta_pars, v_array, i_array, gal)
        
    def setup_imap(self, theta_pars, datavector):
        '''
        theta_pars: dict
            A dict of the sampled mcmc params for both the velocity
            map and the tranformation matrices
        datacube: DataCube
            Datavector datacube truncated to desired lambda bounds
        '''
        imap = self._setup_imap(theta_pars, datavector, self.meta)
        return imap

    @classmethod
    def _setup_imap(cls, theta_pars, datavector, meta):
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
        imap_pars['theory_Nx'] = pars['model_dimension']['Nx']
        imap_pars['theory_Ny'] = pars['model_dimension']['Ny']
        imap_pars['scale'] = pars['model_dimension']['scale']

        return intensity.build_intensity_map(imap_type, datavector, imap_pars)


    def _construct_model_datacube(self, theta_pars, v_array, i_array, gal):
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
        _lrange = self.meta['model_dimension']['lambda_range']
        _dl = self.meta['model_dimension']['lambda_res']
        lambdas = [(l, l+_dl) for l in np.arange(_lrange[0], _lrange[1], _dl)]
        lambda_cen = np.array([(l[0]+l[1])/2.0 for l in lambdas])
        Nspec = len(lambdas)
        #Nspec, Nx, Ny = datavector.shape

        #data = np.zeros(datavector.shape)

        #lambdas = np.array(datavector.lambdas)

        sed = self._setup_sed(self.meta)

        # build Doppler-shifted datacube
        # self.lambda_cen = observed frame lambda grid
        # w_mesh = rest frame wavelengths evaluated on observed frame grid
        # To make energy conserved, dc_array in units of 
        # photons / (s cm2)
        w_mesh = np.outer(lambda_cen, 1./(1.+ v_array))
        w_mesh = w_mesh.reshape(lambda_cen.shape+v_array.shape)
        # photons/s/cm2 in the 3D grid
        dc_array = sed.spectrum(w_mesh.flatten()) * _dl
        dc_array = dc_array.reshape(w_mesh.shape) * \
                        i_array[np.newaxis, :, :] /\
                        (1+v_array[np.newaxis, :, :]) 

        self.modelcube.set_data(dc_array, gal, sed)
        #model_datacube = grism.GrismModelCube(self.parameters,
        #    datacube=dc_array, gal=gal, sed=sed)

        #return model_datacube

    def _setup_inv_cov_list(self, datavector):
        '''
        '''

        # for now, we'll let each datacube class do this
        # to allow for differences in weightmap definitions
        # between experiments

        return datavector.get_inv_cov_list()

    @classmethod
    def _setup_sed(cls, meta):
        # self.meta
        return grism.GrismSED(meta['sed'])


class GrismLikelihood_test(LogLikelihood):
    '''
    An implementation of a LogLikelihood for a Grism KL measurement

    Note: light-weight version
    method:
        - __init__(Pars:parameters, DataVector:datavector)
        x _setup_marginalization(MetaPars:pars)
        - __call__(list:theta, DataVector:datavector)
        x _compute_log_det(imap)
        - setup_vmap(dict:theta_pars, str:model_name)
        - _setup_vmap(dict:theta_pars, MetaPars:meta_pars, str:model_name)
        > _setup_sed(cls, theta_pars, datacube)
        - _interp1d(np.ndarray:table, np.ndarray:values, kind='linear')
        o _log_likelihood(list:theta, DataVector:datavector, DataCube:model)
        o _setup_model(list:theta_pars, DataVector:datavector) 

    attributes:
        From LogBase
        - self.parameters: parameters.Pars
        - self.datavector: None
        - self.meta/pars: parameters.MCMCPars
        - self.sampled: parameters.SampledPars
    '''
    def __init__(self, parameters, datavector):
        ''' Initialization

        Besides the usual likelihood.LogLikelihood initialization, we further
        initialize the grism.GrismModelCube object, fsor the sake of speed
        '''
        # init parent class
        super(GrismLikelihood_test, self).__init__(parameters, datavector)
        _lr = self.meta['model_dimension']['lambda_range']
        _li, _lf = _lr[0], _lr[1]
        self._dl = self.meta['model_dimension']['lambda_res']
        self._lambdas = np.array([(l, l+self._dl) for l in np.arange(_li, _lf, self._dl)])
        self.mNspec = self._lambdas.shape[0]
        self.mNx = self.meta['model_dimension']['Nx']
        self.mNy = self.meta['model_dimension']['Ny']
        self.mscale = self.meta['model_dimension']['scale']
        self.Nobs = datavector.Nobs
        self.obs_types = []
        self.config_list = []
        # init the data vector in C++ module
        for i in range(self.Nobs):
            _config = datavector.get_config(i)
            _config['model_Nx'] = self.mNx
            _config['model_Ny'] = self.mNy
            _config['model_Nlam'] = self.mNspec
            _config['model_scale'] = self.mscale
            _type = _config['type']
            self.obs_types.append(_type)
            self.config_list.append(_config)

    def initialize_observations(self):
        if (self.Nobs != m.get_Nobs()):
            if(m.get_Nobs()>0):
                print(f'clean {m.get_Nobs()} observations')
                m.clear_observation()
            for i in range(self.Nobs):
                _config = self.config_list[i]
                _type = self.obs_types[i]
                _bp = gs.Bandpass(_config.get('bandpass'),
                    wave_type=_config.get('wave_type', 'nm'))
                _bp_list = _bp(self._lambdas)

                if _type=='photometry':
                    m.add_image_observation(_config, 
                        self.datavector.get_data(i), 
                        self.datavector.get_noise(i))
                elif _type=='grism':
                    m.add_grism_observation(
                    _config,
                    self._lambdas,
                    _bp_list,
                    self.datavector.get_data(i), 
                    self.datavector.get_noise(i))
            #self.obs_initialized = True
        assert (m.get_Nobs() == self.Nobs), "Inconsistent # of obs!"

    def __call__(self, theta, data):
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

        #start_time = time()*1000

        # unpack sampled params
        theta_pars = self.theta2pars(theta)
        if (m.get_Nobs() != self.Nobs):
            self.initialize_observations()
        # setup model corresponding to cube.DataCube type
        dc_array, gal, sed = self._setup_model(theta_pars)

        #print("---- build model | %.2f ms -----" % (time()*1000 - start_time))

        # if we are computing the marginalized posterior over intensity
        # map parameters, then we need to scale this likelihood by a
        # determinant factor
        if self.marginalize_intensity is True:
            log_det = self._compute_log_det(self.imap)
        else:
            log_det = 1.

        return (-0.5 * log_det) + self._log_likelihood(dc_array,gal,sed)

    def _log_likelihood(self, dc_array, gal, sed):
        '''
        theta: list
            Sampled parameters, order defined by pars_order
        datavector: DataVector
            The grism datavector
        model: DataCube
            The model datacube object, truncated to desired lambda bounds
        '''

        #start_time = time()*1000

        loglike = 0

        # can figure out how to remove this for loop later
        # will be fast enough with numba anyway
        for i in range(self.Nobs):
            _img, _noise = self.get_simulated_image(i, dc_array, gal, sed)
            loglike += -0.5*m.get_chi2(i, _img)

        #print("---- calculate dchi2 | %.2f ms -----" % (time()*1000 - start_time))
        # NOTE: Actually slower due to extra matrix evals...
        # diff_2 = (datacube.data - model.data).reshape(Nspec, Nx*Ny)
        # chi2_2 = diff_2.dot(inv_cov.dot(diff_2.T))
        # loglike2 = -0.5*chi2_2.trace()

        return loglike

    def _setup_model(self, theta_pars):
        '''
        Setup the model datacube given the input theta_pars and datavector

        theta_pars: dict
            Dictionary of sampled pars
        '''
        # create grid of pixel centers in image coords
        X, Y = utils.build_map_grid(self.mNx, self.mNy, 
            indexing='xy', scale=self.mscale)

        # create 2D velocity & intensity maps given sampled transformation
        # parameters
        vmap = self.setup_vmap(theta_pars)
        imap = self.setup_imap(theta_pars)

        # TODO: temp for debugging!
        self.imap = imap

        try:
            use_numba = self.meta['run_options']['use_numba']
        except KeyError:
            use_numba = False

        # evaluate maps at pixel centers in obs plane
        v_array = vmap(
            'obs', X, Y, normalized=True, use_numba=use_numba
            )
        self.v_array = v_array
        # get both the emission line and continuum image
        _pars = self.meta.copy_with_sampled_pars(theta_pars)
        _pars['run_options']['imap_return_gal'] = True
        i_array, gal = imap.render(theta_pars, None, _pars)
        self.i_array = i_array
        dc_array, sed = self._construct_model_datacube(theta_pars, 
            v_array, i_array, gal)
        return dc_array, gal, sed

    def setup_imap(self, theta_pars):
        '''
        theta_pars: dict
            A dict of the sampled mcmc params for both the velocity
            map and the tranformation matrices
        datacube: DataCube
            Datavector datacube truncated to desired lambda bounds
        '''
        imap = self._setup_imap(theta_pars, self.meta)
        return imap

    @classmethod
    def _setup_imap(cls, theta_pars, meta):
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
        imap_pars['theory_Nx'] = pars['model_dimension']['Nx']
        imap_pars['theory_Ny'] = pars['model_dimension']['Ny']
        imap_pars['scale'] = pars['model_dimension']['scale']

        return intensity.build_intensity_map(imap_type, None, imap_pars)


    def _construct_model_datacube(self, theta_pars, v_array, i_array, gal):
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
        lambda_cen = np.mean(self._lambdas, axis=1)
        sed = self._setup_sed(self.meta)
        # build Doppler-shifted datacube
        # self.lambda_cen = observed frame lambda grid
        # w_mesh = rest frame wavelengths evaluated on observed frame grid
        # To make energy conserved, dc_array in units of 
        # photons / (s cm2)
        w_mesh = np.outer(lambda_cen, 1./(1.+ v_array))
        w_mesh = w_mesh.reshape(lambda_cen.shape+v_array.shape)
        # photons/s/cm2 in the 3D grid
        dc_array = sed.spectrum(w_mesh.flatten()) * self._dl
        dc_array = dc_array.reshape(w_mesh.shape) * \
                        i_array[np.newaxis, :, :] /\
                        (1+v_array[np.newaxis, :, :]) 

        return dc_array, sed

    @classmethod
    def _setup_sed(cls, meta):
        # self.meta
        return grism.GrismSED(meta['sed'])

    def _build_PSF_model(self, config, **kwargs):
        ''' Generate PSF model

        Inputs:
        =======
        kwargs: keyword arguments for building psf model
            - lam: wavelength in nm
            - scale: pixel scale

        Outputs:
        ========
        psf_model: GalSim PSF object

        '''
        _type = config.get('psf_type', 'none').lower()
        if _type != 'none':
            if _type == 'airy':
                lam = kwargs.get('lam', 1000) # nm
                scale_unit = kwargs.get('scale_unit', gs.arcsec)
                return gs.Airy(lam=lam, diam=config['diameter']/100,
                                scale_unit=scale_unit)
            elif _type == 'airy_mean':
                scale_unit = kwargs.get('scale_unit', gs.arcsec)
                #return gs.Airy(config['psf_fwhm']/1.028993969962188, 
                #               scale_unit=scale_unit)
                lam = kwargs.get("lam_mean", 1000) # nm
                return gs.Airy(lam=lam, diam=config['diameter']/100,
                                scale_unit=scale_unit)
            elif _type == 'moffat':
                beta = config.get('psf_beta', 2.5)
                fwhm = config.get('psf_fwhm', 0.5)
                return gs.Moffat(beta=beta, fwhm=fwhm)
            else:
                raise ValueError(f'{psf_type} has not been implemented yet!')
        else:
            return None
    def _getNoise(self, config):
        ''' Generate image noise based on parameter settings

        Outputs:
        ========
        noise: GalSim Noise object
        '''
        random_seed = config.get('random_seed', int(time()))
        rng = gs.BaseDeviate(random_seed+1)

        _type = config.get('noise_type', 'ccd').lower()
        if _type == 'ccd':
            sky_level = config.get('sky_level', 0.65*1.2)
            read_noise = config.get('read_noise', 8.5)
            gain = config.get('gain', 1.0)
            exp_time = config.get('exp_time', 1.0)
            noise = gs.CCDNoise(rng=rng, gain=self.gain, 
                                read_noise=read_noise, 
                                sky_level=sky_level*exp_time/gain)
        elif _type == 'gauss':
            sigma = config.get('noise_sigma', 1.0)
            noise = gs.GaussianNoise(rng=rng, sigma=sigma)
        elif _type == 'poisson':
            sky_level = config.get('sky_level', 0.65*1.2)
            noise = gs.PoissonNoise(rng=rng, sky_level=sky_level)
        else:
            raise ValueError(f'{self.noise_type} not implemented yet!')
        return noise
    def get_simulated_image(self, index, dc_array, gal, sed, force_noise_free=True):
        _cfg = self.config_list[index]
        _bp = gs.Bandpass(_cfg.get('bandpass'), 
                wave_type=_cfg.get('wave_type', 'nm'))
        if self.obs_types[index] == 'photometry':
            gal_chromatic = gal * sed.spectrum
            # convolve with PSF
            psf = self._build_PSF_model(_cfg, 
                  lam=_bp.calculateEffectiveWavelength())
            if psf is not None:
                gal_chromatic = gs.Convolve([gal_chromatic, psf])
            _img = gal_chromatic.drawImage(nx=_cfg['Nx'], 
                ny=_cfg['Ny'], scale=_cfg['pix_scale'], 
                method='auto', area=np.pi*(_cfg['diameter']/2.0)**2, 
                exptime=_cfg['exp_time'], gain=_cfg['gain'],
                bandpass=_bp)
        elif self.obs_types[index] == 'grism':
            _imgArray = np.zeros([_cfg['Ny'], _cfg['Nx']], 
                dtype=np.float64, order='C')
            m.get_dispersed_image(index, dc_array, _imgArray)
            _img = gs.Image(_imgArray, dtype = np.float64, 
                scale=_cfg['pix_scale'])
            psf = self._build_PSF_model(_cfg, lam_mean=np.mean(self._lambdas))
            if psf is not None:
                _gal = gs.InterpolatedImage(_img, scale=_cfg['pix_scale'])
                grism_gal = gs.Convolve([_gal, psf])
                _img = grism_gal.drawImage(nx=_cfg['Nx'], ny=_cfg['Ny'], scale=_cfg['pix_scale'])
        else:
            print(f'Invalid type {self.obs_type[index]}')
            exit(-1)
        if force_noise_free:
            return _img.array, None
        else:
            noise = self._getNoise(_cfg)
            _img_with_Noise = _img.copy()
            _img_with_Noise.addNoise(noise)
            noise_img = _img_with_Noise - _img
            if _cfg.get('apply_to_data', True):
                return _img_with_Noise.array, noise_img.array
            else:
                return _img.array, noise_img.array

def get_likelihood_types():
    return LIKELIHOOD_TYPES

# NOTE: This is where you must register a new likelihood model
LIKELIHOOD_TYPES = {
    'default': DataCubeLikelihood,
    'datacube': DataCubeLikelihood,
    'grism': GrismLikelihood,
    'grism_test': GrismLikelihood_test
    }

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

    mcmc_pars = {
        'units': {
            'v_unit': Unit('km / s'),
            'r_unit': Unit('kpc'),
        },
        'priors': {
            'g1': priors.GaussPrior(0., 0.1, clip_sigmas=2),
            'g2': priors.GaussPrior(0., 0.1, clip_sigmas=2),
            'theta_int': priors.UniformPrior(0., np.pi),
            'sini': priors.UniformPrior(0., 1.),
            'v0': priors.UniformPrior(0, 20),
            'vcirc': priors.GaussPrior(200, 20, clip_sigmas=3),
            'rscale': priors.UniformPrior(0, 10),
        },
        'intensity': {
            # For this test, use truth info
            'type': 'inclined_exp',
            'flux': 3.8e4, # counts
            'hlr': 3.5,
        },
        'velocity': {
            'model': 'centered'
        },
        'run_options': {
            'use_numba': False,
            }
    }

    # create dummy datacube for tests
    data = np.zeros((100,10,10))
    pix_scale = 1
    bandpasses = [gs.Bandpass(
        1., 'A', blue_limit=1e4, red_limit=2e4)
                  for i in range(100)
                  ]
    datacube = DataCube(
        data, pix_scale=pix_scale, bandpasses=bandpasses
        )

    datacube.set_psf(gs.Gaussian(fwhm=0.8, flux=1.))

    sampled_pars = list(true_pars)
    pars = Pars(sampled_pars, mcmc_pars)
    pars_order = pars.sampled.pars_order

    print('Creating LogPosterior object w/ default likelihood')
    log_posterior = LogPosterior(pars, datacube, likelihood='datacube')

    print('Creating LogPosterior object w/ datacube likelihood')
    log_posterior = LogPosterior(pars, datacube, likelihood='datacube')

    print('Calling Logposterior w/ random theta')
    theta = np.random.rand(len(sampled_pars))

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
