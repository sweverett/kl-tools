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
#from mpi4py import MPI
#_mpi = MPI
#_mpi_comm = _mpi.COMM_WORLD
#_mpi_size = _mpi_comm.Get_size()
#_mpi_rank = _mpi_comm.Get_rank()

import kl_tools.utils as utils
import kl_tools.priors as priors
import kl_tools.intensity as intensity
from kl_tools.parameters import Pars, MetaPars
from kl_tools.velocity import VelocityMap
import kl_tools.cube as cube
from kl_tools.cube import DataCube
from kl_tools.datavector import DataVector, FiberDataVector
import kl_tools.grism_modules.grism as grism
import kl_tools.emission as emission
import kltools_grism_module_2 as m
#m.set_mpi_info(_mpi_size, _mpi_rank)

import ipdb

# TODO: Make LogLikelihood a base class w/ abstract call,
# and make current implementation for IFU

parser = ArgumentParser()

parser.add_argument('--show', action='store_true', default=False,
                    help='Set to show test plots')
parser.add_argument('--test', action='store_true', default=False,
                    help='Set to run tests')

CubeLists = []
CubeParsLists = []
def init_CubePars_lists(cubepars_lists):
    assert type(cubepars_lists) is list, "Input cubepars_lists must be list!"
    global CubeParsLists
    CubeParsLists = cubepars_lists

def init_Cube_lists(cube_lists):
    assert type(cube_lists) is list, "Input cube_lists must be list!"
    global CubeLists
    CubeLists = cube_lists

def get_Cube(i):
    global CubeLists
    assert i<len(CubeLists), print(f'Requesting cube {i} out of {len(CubeLists)}!')
    return CubeLists[i]

def get_CubePars(i):
    global CubeParsLists
    assert i<len(CubeParsLists), print(f'Requesting cubepar {i} out of {len(CubeParsLists)}!')
    return CubeParsLists[i]

GlobalDataVector = []
def init_GlobalDataVector(dv_list):
    assert type(dv_list) is list, "Input dv_list must be list!"
    global GlobalDataVector
    GlobalDataVector = dv_list
def get_GlobalDataVector(i):
    global GlobalDataVector
    assert i<len(GlobalDataVector), f'Requesting GlobalDataVector {i} out of {len(GlobalDataVector)}!'
    return GlobalDataVector[i]

Global_lambdas_hires = []
Global_lambdas = []
def init_GlobalLambdas_hires(lambdas_hires):
    global Global_lambdas_hires
    Global_lambdas_hires = lambdas_hires
def get_GlobalLambdas_hires(i):
    global Global_lambdas_hires
    assert i<len(Global_lambdas_hires), f'Index {i} out of range of {len(Global_lambdas)}!'
    return Global_lambdas_hires[i]
def init_GlobalLambdas(lambdas):
    global Global_lambdas
    Global_lambdas = lambdas
def get_GlobalLambdas(i):
    global Global_lambdas
    assert i<len(Global_lambdas), f'Index {i} out of range of {len(Global_lambdas)}!'
    return Global_lambdas[i]

class LogBase(object):
    '''
    Base class for a probability dist that holds the meta and
    sampled parameters
    '''

    def __init__(self, parameters, datavector):
        '''
        parameters: `parameters.Pars`
            Pars instance that holds all parameters needed for MCMC
            run, including `SampledPars` and `MetaPars`
        datavector: any class sub-classes from `datavector.DataVector`
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

    def __init__(self, parameters, datavector, likelihood='default', **kwargs):
        '''
        parameters: `parameters.Pars` object
            Pars instance that holds all parameters needed for MCMC
            run, including `SampledPars` and `MCMCPars`
        datavector: any class sub-classes from `datavector.DataVector`
            If DataCube, truncated to desired lambda bounds
        '''
        # set self.parameters and self.datavector (to Null)
        super(LogPosterior, self).__init__(parameters, DataVector())

        self.log_prior = LogPrior(parameters)
        self.log_likelihood = build_likelihood_model(
            likelihood, parameters, datavector, **kwargs
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
        data: DataVector, etc.
            Arbitrary data vector. If DataCube, truncated
            to desired lambda bounds
        pars: Pars object
        '''

        logprior = self.log_prior(theta)
        theta_pars = self.theta2pars(theta)
        # ad hoc prior on g1/g2/eint1/eint2
        if ('g1' in theta_pars.keys()) and ('g2' in theta_pars.keys()):
            if np.abs(theta_pars['g1'] + 1j*theta_pars['g2'])>1.:
                logprior = -np.inf
        if ('eint1' in theta_pars.keys()) and ('eint2' in theta_pars.keys()):
            if np.abs(theta_pars['eint1'] + 1j*theta_pars['eint2'])>1.:
                logprior = -np.inf
        if logprior == -np.inf:
            return -np.inf, self.blob(-np.inf, -np.inf)

        else:
            loglike = self.log_likelihood(theta, data)
        if np.isnan(loglike):
            print("loglike is NaN at theta = ", theta)
            exit(-1)

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

    def __init__(self, parameters, datavector, **kwargs):
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

    def setup_vmap(self, theta_pars, meta):
        '''
        theta_pars: dict
            A dict of the sampled mcmc params for both the velocity
            map and the tranformation matrices
        model_name: str
            The model name to use when constructing the velocity map
        '''

        try:
            #model_name = self.pars['velocity']['model']
            model_name = meta['velocity']['model']
        except KeyError:
            model_name = 'default'
        for key in ['v0', 'vcirc', 'rscale']:
            if theta_pars.get(key, None) is None:
                theta_pars[key] = meta['velocity'][key]
        for key in ['theta_int', 'sini']:
            if theta_pars.get(key, None) is None:
                theta_pars[key] = meta[key]
        # setup vmap offset parameters
        # Note: the sampled parameters are fractional offset dx/y_kin
        # while the parameters used in vmap model are absolute x0/y0 center pos
        if model_name == 'offset':
            for k_sample, k_model, k_intoff in zip(['dx_kin', 'dy_kin'], ['x0', 'y0'], ['dx_spec', 'dy_spec']):
                # if not sampled, filled with the fiducial or default
                if theta_pars.get(k_sample, None) is None:
                    theta_pars[k_sample] = meta['velocity'].get(k_sample, 0.0)
                # transform from fractional difference to absolute offset
                theta_pars[k_model] = theta_pars[k_sample]*theta_pars['rscale']
                # add 2D spec offset on top of the velocity offset
                # (they are different)
                if theta_pars.get(k_intoff, None) is None:
                    theta_pars[k_intoff] = meta['intensity'].get(k_intoff, 0.)
                if theta_pars.get('hlr', None) is None:
                    theta_pars['hlr'] = meta['intensity'].get('hlr')
                theta_pars[k_model] += theta_pars[k_intoff]*theta_pars['hlr']

        # no extras for this func
        #return self._setup_vmap(theta_pars, self.meta, model_name)
        return self._setup_vmap(theta_pars, meta, model_name)

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
        # if (psf is not None) and (np.sum(model) != 0):
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

# class GrismLikelihood(LogLikelihood):
#     '''
#     An implementation of a LogLikelihood for a Grism KL measurement

#     Note: a reminder of LogLikelihood interface
#     method:
#         - __init__(Pars:parameters, DataVector:datavector)
#         x _setup_marginalization(MetaPars:pars)
#         - __call__(list:theta, DataVector:datavector)
#         x _compute_log_det(imap)
#         - setup_vmap(dict:theta_pars, str:model_name)
#         - _setup_vmap(dict:theta_pars, MetaPars:meta_pars, str:model_name)
#         > _setup_sed(cls, theta_pars, datacube)
#         - _interp1d(np.ndarray:table, np.ndarray:values, kind='linear')
#         o _log_likelihood(list:theta, DataVector:datavector, DataCube:model)
#         o _setup_model(list:theta_pars, DataVector:datavector)

#     attributes:
#         From LogBase
#         - self.parameters: parameters.Pars
#         - self.datavector: cube.DataVector
#         - self.meta/pars: parameters.MCMCPars
#         - self.sampled: parameters.SampledPars
#     '''
#     def __init__(self, parameters, datavector):
#         ''' Initialization

#         Besides the usual likelihood.LogLikelihood initialization, we further
#         initialize the grism.GrismModelCube object, fsor the sake of speed
#         '''
#         super(GrismLikelihood, self).__init__(parameters, datavector)
#         # init with an empty model cube
#         _lrange = self.meta['model_dimension']['lambda_range']
#         _dl = self.meta['model_dimension']['lambda_res']
#         Nspec = len(np.arange(_lrange[0], _lrange[1], _dl))
#         Nx = self.meta['model_dimension']['Nx']
#         Ny = self.meta['model_dimension']['Ny']
#         # init with an empty modelcube object
#         self.modelcube = grism.GrismModelCube(self.parameters,
#             datacube=np.zeros([Nspec, Nx, Ny]))
#         self.set_obs_methods(datavector)
#         return

#     def set_obs_methods(self, datavector):
#         self.modelcube.set_obs_methods(datavector.get_config_list())

#     #def __call__(self, theta, datavector):
#     def __call__(self, theta):
#         '''
#         Do setup and type / sanity checking here
#         before calling the abstract method _loglikelihood,
#         which will have a different implementation for each class

#         theta: list
#             Sampled parameters. Order defined in self.pars_order
#         datavector: DataCube, etc.
#             Arbitrary data vector that subclasses from DataVector.
#             If DataCube, truncated to desired lambda bounds
#         '''

#         #start_time = time()*1000

#         # unpack sampled params
#         theta_pars = self.theta2pars(theta)

#         # setup model corresponding to cube.DataCube type
#         #model = self._setup_model(theta_pars, datavector)
#         self._setup_model(theta_pars, datavector)

#         #print("---- build model | %.2f ms -----" % (time()*1000 - start_time))

#         # if we are computing the marginalized posterior over intensity
#         # map parameters, then we need to scale this likelihood by a
#         # determinant factor
#         if self.marginalize_intensity is True:
#             log_det = self._compute_log_det(self.imap)
#         else:
#             log_det = 1.

#         return (-0.5 * log_det) + self._log_likelihood(theta, datavector)

#     def _log_likelihood(self, theta, datavector):
#         '''
#         theta: list
#             Sampled parameters, order defined by pars_order
#         datavector: DataVector
#             The grism datavector
#         model: DataCube
#             The model datacube object, truncated to desired lambda bounds
#         '''

#         #start_time = time()*1000
#         # a (Nspec, Nx*Ny, Nx*Ny) inverse covariance matrix for the image pixels
#         inv_cov = self._setup_inv_cov_list(datavector)

#         loglike = 0

#         # can figure out how to remove this for loop later
#         # will be fast enough with numba anyway
#         for i in range(datavector.Nobs):

#             _img, _noise = self.modelcube.observe(i, force_noise_free=True)
#             diff = (datavector.get_data(i) - _img)
#             chi2 = np.sum(diff**2/inv_cov[i])

#             loglike += -0.5*chi2

#         #print("---- calculate dchi2 | %.2f ms -----" % (time()*1000 - start_time))
#         # NOTE: Actually slower due to extra matrix evals...
#         # diff_2 = (datacube.data - model.data).reshape(Nspec, Nx*Ny)
#         # chi2_2 = diff_2.dot(inv_cov.dot(diff_2.T))
#         # loglike2 = -0.5*chi2_2.trace()

#         return loglike

#     def _setup_model(self, theta_pars, datavector):
#         '''
#         Setup the model datacube given the input theta_pars and datavector

#         theta_pars: dict
#             Dictionary of sampled pars
#         datavector: DataVector
#             GrismDataVector object
#         '''
#         Nx = self.meta['model_dimension']['Nx']
#         Ny = self.meta['model_dimension']['Ny']
#         model_scale = self.meta['model_dimension']['scale']

#         # create grid of pixel centers in image coords
#         X, Y = utils.build_map_grid(Nx, Ny, indexing='xy', scale=model_scale)

#         # create 2D velocity & intensity maps given sampled transformation
#         # parameters
#         vmap = self.setup_vmap(theta_pars)
#         imap = self.setup_imap(theta_pars, datavector)

#         # TODO: temp for debugging!
#         self.imap = imap

#         try:
#             use_numba = self.meta['run_options']['use_numba']
#         except KeyError:
#             use_numba = False

#         # evaluate maps at pixel centers in obs plane
#         v_array = vmap(
#             'obs', X, Y, normalized=True, use_numba=use_numba
#             )
#         self.v_array = v_array
#         # get both the emission line and continuum image
#         _pars = self.meta.copy_with_sampled_pars(theta_pars)
#         _pars['run_options']['imap_return_gal'] = True
#         i_array, gal = imap.render(theta_pars, datavector, _pars)
#         self.i_array = i_array
#         self._construct_model_datacube(theta_pars, v_array, i_array, gal)

#     def setup_imap(self, theta_pars, datavector):
#         '''
#         theta_pars: dict
#             A dict of the sampled mcmc params for both the velocity
#             map and the tranformation matrices
#         datacube: DataCube
#             Datavector datacube truncated to desired lambda bounds
#         '''
#         imap = self._setup_imap(theta_pars, datavector, self.meta)
#         return imap

#     @classmethod
#     def _setup_imap(cls, theta_pars, datavector, meta):
#         '''
#         See setup_imap(). Only runs if a new imap for the sample
#         is needed
#         '''

#         # Need to check if any basis func parameters are
#         # being sampled over
#         pars = meta.copy_with_sampled_pars(theta_pars)

#         imap_pars = deepcopy(pars['intensity'])
#         imap_type = imap_pars['type']
#         del imap_pars['type']
#         imap_pars['theory_Nx'] = pars['model_dimension']['Nx']
#         imap_pars['theory_Ny'] = pars['model_dimension']['Ny']
#         imap_pars['scale'] = pars['model_dimension']['scale']

#         return intensity.build_intensity_map(imap_type, datavector, imap_pars)


#     def _construct_model_datacube(self, theta_pars, v_array, i_array, gal):
#         '''
#         Create the model datacube from model slices, using the evaluated
#         velocity and intensity maps, SED, etc.

#         theta_pars: dict
#             A dict of the sampled mcmc params for both the velocity
#             map and the tranformation matrices
#         v_array: np.array (2D)
#             The vmap evaluated at image pixel positions for sampled pars.
#             (Must be normalzied)
#         i_array: np.array (2D)
#             The imap evaluated at image pixel positions for sampled pars
#         cont_array: np.array (2D)
#             The imap of the fitted or modeled continuum
#         datacube: DataCube
#             Datavector datacube truncated to desired lambda bounds
#         '''
#         _lrange = self.meta['model_dimension']['lambda_range']
#         _dl = self.meta['model_dimension']['lambda_res']
#         lambdas = [(l, l+_dl) for l in np.arange(_lrange[0], _lrange[1], _dl)]
#         lambda_cen = np.array([(l[0]+l[1])/2.0 for l in lambdas])
#         Nspec = len(lambdas)
#         #Nspec, Nx, Ny = datavector.shape

#         #data = np.zeros(datavector.shape)

#         #lambdas = np.array(datavector.lambdas)

#         sed = self._setup_sed(self.meta)

#         # build Doppler-shifted datacube
#         # self.lambda_cen = observed frame lambda grid
#         # w_mesh = rest frame wavelengths evaluated on observed frame grid
#         # To make energy conserved, dc_array in units of
#         # photons / (s cm2)
#         w_mesh = np.outer(lambda_cen, 1./(1.+ v_array))
#         w_mesh = w_mesh.reshape(lambda_cen.shape+v_array.shape)
#         # photons/s/cm2 in the 3D grid
#         dc_array = sed.spectrum(w_mesh.flatten()) * _dl
#         dc_array = dc_array.reshape(w_mesh.shape) * \
#                         i_array[np.newaxis, :, :] /\
#                         (1+v_array[np.newaxis, :, :])

#         self.modelcube.set_data(dc_array, gal, sed)
#         #model_datacube = grism.GrismModelCube(self.parameters,
#         #    datacube=dc_array, gal=gal, sed=sed)

#         #return model_datacube

#     def _setup_inv_cov_list(self, datavector):
#         '''
#         '''

#         # for now, we'll let each datacube class do this
#         # to allow for differences in weightmap definitions
#         # between experiments

#         return datavector.get_inv_cov_list()

#     @classmethod
#     def _setup_sed(cls, meta):
#         # self.meta
#         return grism.GrismSED(meta['sed'])


class GrismLikelihood(LogLikelihood):
    '''
    An implementation of a LogLikelihood for a Grism KL measurement

    Note: light-weight version
    '''
    def __init__(self, parameters, datavector, **kwargs):
        ''' Initialization
        Set up GrismPars and GrismModelCube objects by providing a
        GrismDataVector object or fiducial parameters
        '''

        '''
        super(FiberLikelihood, self).__init__(parameters, DataVector())
        _meta = parameters.meta.copy()
        _mdim = {'model_dimension': _meta['model_dimension'],
                 'run_options': _meta['run_options']}
        ### Case 1: Build data vector from input fiducial parameters
        if datavector is None:
            print("FiberLikelihood: Generating data vector from fid pars...")
            ### preparing observation configuration
            _conf = _meta.pop('obs_conf')
            _fid = kwargs["sampled_theta_fid"]
            self.Nobs = len(_conf)
            init_Cube_lists([cube.FiberModelCube(_mdim, c) for c in _conf])
            self._set_model_dimension()
            ### generate fiducial images and DataVector object
            _d, _n = self.get_images(_fid, force_noise_free=False, return_noise=True)
            _header = {'NEXTEN': 2*self.Nobs, 'OBSNUM': self.Nobs}
            init_GlobalDataVector([FiberDataVector(
                 header=_header, data_header=_conf,
                 data=_d, noise=_n)])
            # set the fiducial images to C++ routine
            print("FiberLikelihood: Caching the (fiducial) data vector...")
        ### Case 2: Build data vector from input FiberDataVector object
        else:
            init_GlobalDataVector([datavector])
            self._set_model_dimension()
            if _meta.get('obs_conf', None) is not None:
                print(f'Use obs_conf from datavector not parameters')
                _meta.pop(obs_conf)
            _conf = datavector.get_config_list()
            self.Nobs = datavector.Nobs
            # init_CubePars_lists([cube.FiberPars(meta_copy.pars, obs_conf_list[i]) for i in range(self.Nobs)])
            # init_Cube_lists([cube.FiberModelCube(get_CubePars(i)) for i in range(self.Nobs)])
            init_Cube_lists([cube.FiberModelCube(_mdim, c) for c in _conf])
        return
        '''
        super(GrismLikelihood, self).__init__(parameters, DataVector())
        _meta = parameters.meta.copy()
        _mdim = {'model_dimension': _meta['model_dimension'],
                 'run_options': _meta['run_options']}
        ### Case 1: Build data vector from input fiducial parameters
        if datavector is None:
            print("GrismLikelihood: Generating data vector from fid pars...")
            _conf = _meta.pop('obs_conf')
            _fid = kwargs["sampled_theta_fid"]
            self.Nobs = len(_conf)
            init_Cube_lists([cube.GrismModelCube(_mdim, c) for c in _conf])
            self._set_model_dimension()
            # set GrismPars and GrismModelCube list
            # if parameters.meta.get('obs_conf', None) is None:
            #     raise ValueError(f'parameters must have obs_conf if '+\
            #         f'data vector is not provided!')
            # if kwargs.get('sampled_theta_fid', None) is None:
            #     raise ValueError(f'sampled_theta_fid must be set when '+\
            #         f'data vector is not provided!')
            # meta_copy = parameters.meta.copy()
            # obs_conf_list = meta_copy.pop('obs_conf')
            # self.Nobs = len(obs_conf_list)
            GrismPars_list = [grism.GrismPars(meta_copy.pars, obs_conf_list[i]) for i in range(self.Nobs)]
            GrismModelCube_list = [grism.GrismModelCube(p) for p in GrismPars_list]
            # initialize C++ routine without setting data vector
            print("debug: Nlam = %d/%d"%(GrismPars_list[0].lambdas.shape,
                GrismPars_list[0].bp_array.shape[0]))
            grism.initialize_observations(self.Nobs, GrismPars_list)
            init_Cube_lists(GrismModelCube_list)
            init_CubePars_lists(GrismPars_list)
            # generate fiducial images and DataVector object
            theta_pars_fid = self.theta2pars(kwargs['sampled_theta_fid'])
            dc_array, gal, sed = self._setup_model(theta_pars_fid)
            fid_img_list, fid_noise_list = [], []
            for i in range(self.Nobs):
                img, noise = self.GrismModelCube_list[i].observe(dc_array, force_noise_free=False, datavector=None)
                fid_img_list.append(img)
                fid_noise_list.append(noise)
            _header = {'NEXTEN': 2*self.Nobs, 'OBSNUM': self.Nobs}
            _data_header = obs_conf_list
            self.datavector = grism.GrismDataVector(
                header=_header, data_header=_data_header,
                data=fid_img_list, noise=fid_noise_list)
            # set the fiducial images to C++ routine
            print("GrismLikelihood: Caching the (fiducial) data vector...")
            grism.initialize_observations(self.Nobs, GrismPars_list, datavector=self.datavector, overwrite=True)
            input_from_fid_theta = True
        ### Case 2: Build data vector from input GrismDataVector object
        else:
            init_GlobalDataVector([datavector])
            self._set_model_dimension()
            # ignore the obs_conf from yaml file if we are using existing data
            if _meta.get('obs_conf', None) is not None:
                print(f'Use obs_conf from datavector not parameters')
                _meta.pop('obs_conf')
            _conf = datavector.get_config_list()
            self.Nobs = datavector.Nobs

            GrismPars_list = []
            GrismModelCube_list = []
            for i in range(self.Nobs):
                _gp = grism.GrismPars(_meta.pars, _conf[i])
                _gmc = grism.GrismModelCube(_gp)
                GrismPars_list.append(_gp)
                GrismModelCube_list.append(_gmc)
            # self.GrismPars_list = [grism.GrismPars(meta_copy, conf) for conf in obs_conf_list]
            # self.GrismModelCube_list = [grism.GrismModelCube(p) for p in self.GrismPars_list]
            # initialize C++ routine with setting data vector
            init_Cube_lists(GrismModelCube_list)
            init_CubePars_lists(GrismPars_list)
            grism.initialize_observations(self.Nobs, GrismPars_list, datavector)
            #input_from_fid_theta = False
        return

    # def _set_model_dimension(self):
    #     # build model lambda-y-x grid
    #     # TODO: retrieve header in a consistent way
    #     _lr = self.meta['model_dimension']['lambda_range']
    #     _li, _lf = _lr[0], _lr[1]
    #     self._dl = self.meta['model_dimension']['lambda_res']
    #     self._lambdas = np.array([(l, l+self._dl) for l in np.arange(_li, _lf, self._dl)])
    #     self.mNspec = self._lambdas.shape[0]
    #     self.mNx = self.meta['model_dimension']['Nx']
    #     self.mNy = self.meta['model_dimension']['Ny']
    #     self.mscale = self.meta['model_dimension']['scale']
    #     return
    def _set_model_dimension(self):
        ''' Set up observer-frame lambda-y-x grid
        Two sets of wavelength resolution are initialized:
            - normal resolution where the data are extracted
            - high resolution where the modeling is evaluated on
        '''
        # TODO: retrieve header in a consistent way
        wave_limit = np.atleast_2d(self.meta['model_dimension']['lambda_range'])
        self.N_SED_block = wave_limit.shape[0]
        self.dwave = np.atleast_1d(self.meta['model_dimension']['lambda_res'])
        self.Nsub = np.atleast_1d(self.meta['model_dimension']['super_sampling'])
        if self.dwave.shape[0]==1:
            self.dwave = np.tile(self.dwave, self.N_SED_block)
        if self.Nsub.shape[0]==1:
            self.Nsub = np.tile(self.Nsub, self.N_SED_block)
        assert (self.dwave.shape[0]==self.N_SED_block) & (self.Nsub.shape[0]==self.N_SED_block)
        # wavelength extraction grid (normal resolution)
        lb_lo = [np.arange(l,r,dw) for (l,r),dw in zip(wave_limit, self.dwave)]
        lambdas=[np.array([lb,lb+dw]).T for lb,dw in zip(lb_lo,self.dwave)]
        init_GlobalLambdas(lambdas)
        #self.lambdas=[np.array([lb,lb+dw]).T for lb,dw in zip(lb_lo,self.dwave)]
        self.mNlam = [grid.shape[0] for grid in lambdas]
        # wavelength modeling grid (super resolution)
        lb_hi = [np.arange(lb[0],lb[-1]+dw-dw/Nsub/2,dw/Nsub) for lb,dw,Nsub in zip(lb_lo, self.dwave, self.Nsub)]
        #self.lambdas_hires = [np.array([lb, lb+dw/Nsub]).T for lb,dw,Nsub in zip(lb_hi, self.dwave, self.Nsub)]
        lambdas_hires = [np.array([lb, lb+dw/Nsub]).T for lb,dw,Nsub in zip(lb_hi, self.dwave, self.Nsub)]
        init_GlobalLambdas_hires(lambdas_hires)
        self.mNlam_hires = [grid.shape[0] for grid in lambdas_hires]

        # spatial grid
        self.mNx = self.meta['model_dimension']['Nx']
        self.mNy = self.meta['model_dimension']['Ny']
        self.mscale = self.meta['model_dimension']['scale']
        return

    def get_images(self, theta):
        ''' Get simulated images given parameter values
        '''
        image_list = []
        dv = get_GlobalDataVector(0)
        theta_pars = self.theta2pars(theta)
        dc_array, gal_phot, sed = self._setup_model(theta_pars, None)
        for i in range(self.Nobs):
            fc = get_Cube(i)
            if fc.pars.is_dispersed:
                iblock, obsid = fc.conf['SEDBLKID'], fc.conf['OBSINDEX']
                img, _ = fc.observe(theory_cube=dc_array[iblock], datavector=dv)
            else:
                img, _ = fc.observe(gal_phot=gal_phot, datavector=dv)
            image_list.append(img)
        return image_list

    # def __call__(self, theta, datavector, model):
    #     '''
    #     Do setup and type / sanity checking here
    #     before calling the abstract method _loglikelihood,
    #     which will have a different implementation for each class

    #     theta: list
    #         value of sampled parameters. Order defined in self.pars_order
    #     datavector: DataCube, etc.
    #         Arbitrary data vector that subclasses from DataVector.
    #         If DataCube, truncated to desired lambda bounds
    #     '''
    #     dc_array, gal, sed = model[0], model[1], model[2]


    #     #start_time = time()*1000

    #     # unpack sampled params
    #     #theta_pars = self.theta2pars(theta)
    #     # setup model corresponding to cube.DataCube type
    #     #dc_array, gal, sed = self._setup_model(theta_pars)

    #     #print("---- build model | %.2f ms -----" % (time()*1000 - start_time))

    #     # if we are computing the marginalized posterior over intensity
    #     # map parameters, then we need to scale this likelihood by a
    #     # determinant factor
    #     if self.marginalize_intensity is True:
    #         log_det = self._compute_log_det(self.setup_imap(theta_pars))
    #     else:
    #         log_det = 1.

    #     return (-0.5 * log_det) + self._log_likelihood(dc_array,gal,sed)

    def _log_likelihood(self, theta, datavector, model, timing=False):
        '''
        theta: list
            Sampled parameters, order defined by pars_order
        datavector: DataVector
            The grism datavector
        model: DataCube
            The model datacube object, truncated to desired lambda bounds
        '''

        #start_time = time()*1000
        dc_array, gal_phot, sed = model[0], model[1], model[2]
        # plt.imshow(dc_array[0].sum(axis=1))
        # plt.show()
        loglike = 0
        theta_pars = self.theta2pars(theta)
        dv = get_GlobalDataVector(0)
        start_time = time()*1000

        #fig, axes = plt.subplots(1,self.Nobs)

        for i in range(self.Nobs):
            fc = get_Cube(i)
            if fc.pars.is_dispersed:
                iblock, obsid = fc.conf['SEDBLKID'], fc.conf['OBSINDEX']
                #img, _ = fc.observe(dc_array[iblock], gal, sed)
                img, _ = fc.observe(theory_cube=dc_array[iblock], datavector=dv)
            else:
                #img, _ = fc.observe(None, gal, sed)
                img, _ = fc.observe(gal_phot=gal_phot, datavector=dv)
            mask = dv.get_mask(i)
            if mask == None:
                mask = np.ones_like(img)
            #axes[i].imshow(img, origin='lower')

            data = dv.get_data(i)
            #noise = np.std(dv.get_noise(i))
            noise = dv.get_noise(i)
            chi2 = np.sum(((data-img)*mask/noise)**2)
            loglike += -0.5*chi2
            if timing:
                print("---- calculate dchi2 %d/%d | %.2f ms -----" % (i+1,self.Nobs,time()*1000 - start_time))
        #plt.show()
        # loglike = 0

        # # can figure out how to remove this for loop later
        # # will be fast enough with numba anyway
        # for i in range(self.Nobs):
        #     _img, _noise = self.GrismModelCube_list[i].observe(dc_array)
        #     loglike += -0.5*m.get_chi2(i, _img)

        # #print("---- calculate dchi2 | %.2f ms -----" % (time()*1000 - start_time))
        # # NOTE: Actually slower due to extra matrix evals...
        # # diff_2 = (datacube.data - model.data).reshape(Nspec, Nx*Ny)
        # # chi2_2 = diff_2.dot(inv_cov.dot(diff_2.T))
        # # loglike2 = -0.5*chi2_2.trace()

        return loglike

    def _setup_model(self, theta_pars, datavector, timing=False):
        '''
        Setup the model datacube given the input theta_pars and datavector

        theta_pars: dict
            Dictionary of sampled pars
        '''
        start_time = time()*1000
        _meta_update = self.meta.copy_with_sampled_pars(theta_pars)
        _meta_update['run_options']['imap_return_gal'] = True
        parametriz =  _meta_update['run_options'].get('alignment_params', 'sini_pa')
        if _meta_update['intensity']['type']=='inclined_exp':
            if parametriz=='inc_pa':
                _meta_update['sini'] = np.sin(_meta_update['inc'])
            elif parametriz=='eint':
                eint1 = _meta_update['eint1']
                eint2 = _meta_update['eint2']
                eint = np.abs(eint1 + 1j * eint2)
                _meta_update['theta_int'] = np.angle(eint1 + 1j * eint2)/2.
                _meta_update['sini'] = np.sin(2.*np.arctan(np.sqrt(eint)))
            elif parametriz=='eint_eigen':
                eint1 = (_meta_update['g1+eint1'] - _meta_update['g1-eint1'])/2.
                eint2 = (_meta_update['g2+eint2'] - _meta_update['g2-eint2'])/2.
                g1 = (_meta_update['g1+eint1'] + _meta_update['g1-eint1'])/2.
                g2 = (_meta_update['g2+eint2'] + _meta_update['g2-eint2'])/2.
                eint = np.abs(eint1 + 1j * eint2)
                _meta_update['theta_int'] = np.angle(eint1 + 1j * eint2)/2.
                _meta_update['sini'] = np.sin(2.*np.arctan(np.sqrt(eint)))
                _meta_update['g1'] = g1
                _meta_update['g2'] = g2
            elif parametriz=="sini_pa":
                pass
            else:
                raise ValueError(f'Unsupported parametrization {parametriz}!')
                exit(-1)

        try:
            use_numba = _meta_update['run_options']['use_numba']
        except KeyError:
            use_numba = False

        # create grid of pixel centers in image coordinates
        # 'xy' indexing is the usual imshow y-x scheme
        X, Y = utils.build_map_grid(self.mNx, self.mNy,
            indexing='xy', scale=self.mscale)

        # create 2D velocity & intensity maps given sampled transformation
        # parameters
        vmap = self.setup_vmap(theta_pars, _meta_update)
        # evaluate maps at pixel centers in obs plane
        v_array = vmap('obs', X, Y, normalized=True, use_numba=use_numba)
        if timing:
            print("\t---- render v map | %.2f ms -----" % (time()*1000 - start_time))
        imap = self.setup_imap(theta_pars, _meta_update)
        # get both the emission line and continuum image
        i_arrays, gals = imap.render(theta_pars, None, _meta_update)
        if timing:
            print("\t---- render i map | %.2f ms -----" % (time()*1000 - start_time))
        # debug purpose
        # fig,axes = plt.subplots(1,2)
        # axes[0].imshow(i_array)
        # axes[1].imshow(v_array)
        # plt.show()

        # dc_array, sed = self._construct_model_datacube(theta_pars,
        #     v_array, i_array, gal)

        # create observer-frame SED and the 3D model cubes, block-by-block
        sed = self.setup_sed(_meta_update)
        if timing:
            print("\t---- render SED | %.2f ms -----" % (time()*1000 - start_time))
        dc_array = []
        for iblock in range(self.N_SED_block):
            # high resolution wavelength grid
            #lambda_cen_hires = self.lambdas_hires[iblock].mean(axis=1)
            lambda_cen_hires = get_GlobalLambdas_hires(iblock).mean(axis=1)
            # debug purpose
            # plt.plot(lambda_cen_hires, sed(lambda_cen_hires))
            # plt.show()
            mshape = lambda_cen_hires.shape+v_array.shape
            # apply Doppler shift due to LoS velocity
            w_mesh = np.outer(lambda_cen_hires, 1./(1.+ v_array)).flatten()

            dc = np.zeros(mshape)
            for key in gals.keys():
                if key.startswith('em_'):
                    # evaluate SED (phot/s/cm2) at observer frame (w/ LoS velocity), note that sed itself is phot/s/cm2/nm by default
                    substart_time = time()*1000
                    sed_mesh = (sed(w_mesh, component=key)*self.dwave[iblock]/self.Nsub[iblock]).reshape(mshape)
                    if timing:
                        print(f'\t---- SED on grid [{key}] ({mshape}) | {time()*1000 - substart_time:.2f} ms -----')
                    dc += sed_mesh*i_arrays[key][np.newaxis,:,:]/(1+v_array[np.newaxis,:, :])
                    if timing:
                        print(f'\t---- SED on grid [{key}] ({mshape}) | {time()*1000 - substart_time:.2f} ms -----')
                if key.startswith('cont_'):
                    sed_mesh = sed(lambda_cen_hires, component='continuum')*self.dwave[iblock]/self.Nsub[iblock]
                    dc += np.outer(sed_mesh, i_arrays[key][np.newaxis,:,:]).reshape(mshape)
            # reduce at low resolution grid
            indices = np.arange(0, self.mNlam_hires[iblock], self.Nsub[iblock])
            #dc_array.append(np.add.reduceat(dc, indices, axis=0))
            dc_array.append(np.asarray([np.sum(dc[x:x+self.Nsub[iblock], :, :], axis=0) for x in indices]))
        if timing:
            print("---- build model | %.2f ms -----" % (time()*1000 - start_time))
        return dc_array, gals["phot"], sed

    @classmethod
    def setup_imap(cls, theta_pars, meta):
        '''
        theta_pars: dict
            A dict of the sampled mcmc params for both the velocity
            map and the tranformation matrices
        datacube: DataCube
            Datavector datacube truncated to desired lambda bounds
        '''
        # imap = self._setup_imap(theta_pars, self.meta)
        # return imap
        imap_pars = deepcopy(meta['intensity'])
        imap_type = imap_pars.pop('type')
        imap_pars['theory_Nx'] = meta['model_dimension']['Nx']
        imap_pars['theory_Ny'] = meta['model_dimension']['Ny']
        imap_pars['scale'] = meta['model_dimension']['scale']

        return intensity.build_intensity_map(imap_type, None, imap_pars)

    # @classmethod
    # def _setup_imap(cls, theta_pars, meta):
    #     '''
    #     See setup_imap(). Only runs if a new imap for the sample
    #     is needed
    #     '''

    #     # Need to check if any basis func parameters are
    #     # being sampled over
    #     pars = meta.copy_with_sampled_pars(theta_pars)

    #     imap_pars = deepcopy(pars['intensity'])
    #     imap_type = imap_pars['type']
    #     del imap_pars['type']
    #     imap_pars['theory_Nx'] = pars['model_dimension']['Nx']
    #     imap_pars['theory_Ny'] = pars['model_dimension']['Ny']
    #     imap_pars['scale'] = pars['model_dimension']['scale']

    #     return intensity.build_intensity_map(imap_type, None, imap_pars)
    @classmethod
    def setup_sed(cls, meta):
        _meta_sed_ = deepcopy(meta['sed'])
        _meta_sed_['lblue'] = meta['model_dimension']['lblue']
        _meta_sed_['lred'] = meta['model_dimension']['lred']
        _meta_sed_['resolution'] = meta['model_dimension']['resolution']
        return emission.ObsFrameSED(_meta_sed_)


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
        dc_array = sed(w_mesh.flatten()) * self._dl
        dc_array = dc_array.reshape(w_mesh.shape) * \
                        i_array[np.newaxis, :, :] /\
                        (1+v_array[np.newaxis, :, :])

        return dc_array, sed

    @classmethod
    def _setup_sed(cls, meta):
        # self.meta
        return emission.ObsFrameSED(meta['sed'])

class FiberLikelihood(LogLikelihood):
    '''
    An implementation of a LogLikelihood for a Fiber KL measurement
    Compare to reduced 1D data F_tilde, which is related to model F as
        F_tilde = RF,
    where R is the ""resolution matrix", see DESI spectrum pipeline paper
    and spectro-perfectionism
    '''
    def __init__(self, parameters, datavector, **kwargs):
        ''' Initialization
        Set up CubePars and ModelCube objects by providing a
        FiberDataVector object or fiducial parameters
        - parameters: Pars object
        - datavector: FiberDataVector object
        '''
        super(FiberLikelihood, self).__init__(parameters, DataVector())
        _meta = parameters.meta.copy()
        _mdim = {'model_dimension': _meta['model_dimension'],
                 'run_options': _meta['run_options']}
        ### Case 1: Build data vector from input fiducial parameters
        if datavector is None:
            print("FiberLikelihood: Generating data vector from fid pars...")
            ### preparing observation configuration
            _conf = _meta.pop('obs_conf')
            _fid = kwargs["sampled_theta_fid"]
            self.Nobs = len(_conf)
            init_Cube_lists([cube.FiberModelCube(_mdim, c) for c in _conf])
            self._set_model_dimension()
            ### generate fiducial images and DataVector object
            _d, _n = self.get_images(_fid, force_noise_free=False, return_noise=True)
            _header = {'NEXTEN': 2*self.Nobs, 'OBSNUM': self.Nobs}
            init_GlobalDataVector([FiberDataVector(
                 header=_header, data_header=_conf,
                 data=_d, noise=_n)])
            # set the fiducial images to C++ routine
            print("FiberLikelihood: Caching the (fiducial) data vector...")
        ### Case 2: Build data vector from input FiberDataVector object
        else:
            init_GlobalDataVector([datavector])
            self._set_model_dimension()
            if _meta.get('obs_conf', None) is not None:
                print(f'Use obs_conf from datavector not parameters')
                _meta.pop(obs_conf)
            _conf = datavector.get_config_list()
            self.Nobs = datavector.Nobs
            # init_CubePars_lists([cube.FiberPars(meta_copy.pars, obs_conf_list[i]) for i in range(self.Nobs)])
            # init_Cube_lists([cube.FiberModelCube(get_CubePars(i)) for i in range(self.Nobs)])
            init_Cube_lists([cube.FiberModelCube(_mdim, c) for c in _conf])
        return

    def _set_model_dimension(self):
        # build model lambda-y-x grid
        # Note: lambda can have a several patches
        # TODO: retrieve header in a consistent way
        wave_limit = np.atleast_2d(self.meta['model_dimension']['lambda_range'])
        self.N_SED_block = wave_limit.shape[0]
        self.dwave = np.atleast_1d(self.meta['model_dimension']['lambda_res'])
        self.Nsub = np.atleast_1d(self.meta['model_dimension']['super_sampling'])
        if self.dwave.shape[0]==1:
            self.dwave = np.tile(self.dwave, self.N_SED_block)
        if self.Nsub.shape[0]==1:
            self.Nsub = np.tile(self.Nsub, self.N_SED_block)
        assert (self.dwave.shape[0]==self.N_SED_block) & (self.Nsub.shape[0]==self.N_SED_block)
        # wavelength extraction grid (normal resolution)
        lb_lo = [np.arange(l,r,dw) for (l,r),dw in zip(wave_limit, self.dwave)]
        lambdas=[np.array([lb,lb+dw]).T for lb,dw in zip(lb_lo,self.dwave)]
        init_GlobalLambdas(lambdas)
        #self.lambdas=[np.array([lb,lb+dw]).T for lb,dw in zip(lb_lo,self.dwave)]
        self.mNlam = [grid.shape[0] for grid in lambdas]
        # wavelength modeling grid (super resolution)
        lb_hi = [np.arange(lb[0],lb[-1]+dw-dw/Nsub/2,dw/Nsub) for lb,dw,Nsub in zip(lb_lo, self.dwave, self.Nsub)]
        #self.lambdas_hires = [np.array([lb, lb+dw/Nsub]).T for lb,dw,Nsub in zip(lb_hi, self.dwave, self.Nsub)]
        lambdas_hires = [np.array([lb, lb+dw/Nsub]).T for lb,dw,Nsub in zip(lb_hi, self.dwave, self.Nsub)]
        init_GlobalLambdas_hires(lambdas_hires)
        self.mNlam_hires = [grid.shape[0] for grid in lambdas_hires]

        # spatial grid
        self.mNx = self.meta['model_dimension']['Nx']
        self.mNy = self.meta['model_dimension']['Ny']
        self.mscale = self.meta['model_dimension']['scale']
        return

    def get_images(self, theta, force_noise_free=True, return_noise=False):
        ''' Get simulated images and data given parameter values
        '''
        image_list = []
        noise_list = []
        # unpack sampled params
        theta_pars = self.theta2pars(theta)
        # evaluate 3D model cube
        dc_array, gal, sed = self._setup_model(theta_pars, None)
        # evaluate observed data
        for i in range(self.Nobs):
            fc = get_Cube(i)
            if fc.Fpars.is_dispersed:
                iblock, obsid = fc.conf['SEDBLKID'], fc.conf['OBSINDEX']
                _img, _noise = fc.observe(dc_array[iblock], gal, sed, force_noise_free)
                # apply a flux norm factor to cancel out shear impact on intensity profile, if it's offset fiber
                dx, dy = fc.conf['FIBERDX'], fc.conf['FIBERDY']
                if np.sqrt(dx*dx+dy*dy)>1e-3:
                    _img *= theta_pars.get('ffnorm_%d'%obsid, 1.0)
            else:
                _img, _noise = fc.observe(None, gal, sed, force_noise_free)
            image_list.append(_img)
            noise_list.append(_noise)
        if return_noise:
            return image_list, noise_list
        else:
            return image_list

    def _log_likelihood(self, theta, datavector, model):
        '''
        theta: list
            Sampled parameters, order defined by pars_order
        datavector: DataVector
            The grism datavector
        model: DataCube
            The model datacube object, truncated to desired lambda bounds
        '''
        dc_array, gal, sed = model[0], model[1], model[2]
        loglike = 0
        theta_pars = self.theta2pars(theta)
        dv = get_GlobalDataVector(0)
        for i in range(self.Nobs):
            fc = get_Cube(i)
            if fc.Fpars.is_dispersed:
                iblock, obsid = fc.conf['SEDBLKID'], fc.conf['OBSINDEX']
                img, _ = fc.observe(dc_array[iblock], gal, sed)
                # apply a flux norm factor to cancel out shear impact on intensity profile, if it's offset fiber
                dx, dy = fc.conf['FIBERDX'], fc.conf['FIBERDY']
                if np.sqrt(dx*dx+dy*dy)>1e-3:
                    img *= theta_pars.get('ffnorm_%d'%obsid, 1.0)
            else:
                img, _ = fc.observe(None, gal, sed)
            data = dv.get_data(i)
            noise = np.std(dv.get_noise(i))
            chi2 = np.sum(((data-img)/noise)**2)
            loglike += -0.5*chi2

        return loglike

    def _setup_model(self, theta_pars, datavector):
        '''
        Setup the model datacube given the input theta_pars and datavector

        theta_pars: dict
            Dictionary of sampled pars

        # Three parameterization schemes are available for the Euler angle
        # 1. sini_pa: parameterize through sin(inc) and pa
        # 2. inc_pa: parameterize through inc and pa
        # 3. eint: parametrize through eint_1 and eint_2
        #    **Warning: this parametrization has spin-2 symmetry, so it's missing half of the inclination parameter space, but could be useful when inferring with image-only**
        '''
        _meta_update = self.meta.copy_with_sampled_pars(theta_pars)
        _meta_update['run_options']['imap_return_gal'] = True
        parametriz =  _meta_update['run_options'].get('alignment_params', 'sini_pa')
        if _meta_update['intensity']['type']=='inclined_exp':
            if parametriz=='inc_pa':
                _meta_update['sini'] = np.sin(_meta_update['inc'])
            elif parametriz=='eint':
                eint1 = _meta_update['eint1']
                eint2 = _meta_update['eint2']
                eint = np.abs(eint1 + 1j * eint2)
                _meta_update['theta_int'] = np.angle(eint1 + 1j * eint2)/2.
                _meta_update['sini'] = np.sin(2.*np.arctan(np.sqrt(eint)))
            elif parametriz=='eint_eigen':
                eint1 = (_meta_update['g1+eint1'] - _meta_update['g1-eint1'])/2.
                eint2 = (_meta_update['g2+eint2'] - _meta_update['g2-eint2'])/2.
                g1 = (_meta_update['g1+eint1'] + _meta_update['g1-eint1'])/2.
                g2 = (_meta_update['g2+eint2'] + _meta_update['g2-eint2'])/2.
                eint = np.abs(eint1 + 1j * eint2)
                _meta_update['theta_int'] = np.angle(eint1 + 1j * eint2)/2.
                _meta_update['sini'] = np.sin(2.*np.arctan(np.sqrt(eint)))
                _meta_update['g1'] = g1
                _meta_update['g2'] = g2
            elif parametriz=="sini_pa":
                pass
            else:
                raise ValueError(f'Unsupported parametrization {parametriz}!')
                exit(-1)

        try:
            use_numba = _meta_update['run_options']['use_numba']
        except KeyError:
            use_numba = False

        # create grid of pixel centers in image coordinates
        # 'xy' indexing is the usual imshow y-x scheme
        X, Y = utils.build_map_grid(self.mNx, self.mNy, indexing='xy', scale=self.mscale)
        # create 2D velocity & intensity maps given sampled transformation
        # parameters, and evaluate maps at pixel centers in obs plane
        vmap = self.setup_vmap(theta_pars, _meta_update)
        imap = self.setup_imap(theta_pars, _meta_update)
        v_array = vmap('obs', X, Y, normalized=True, use_numba=use_numba)
        i_array, gal = imap.render(theta_pars, None, _meta_update, im_type='emission')

        # create observer-frame SED and the 3D model cubes, block-by-block
        sed = self.setup_sed(_meta_update)
        dc_array = []
        #dc_array, sed = self._construct_model_datacube(theta_pars, _pars,
        #    v_array, i_array, gal)
        for iblock in range(self.N_SED_block):
            # high resolution wavelength grid
            #lambda_cen_hires = self.lambdas_hires[iblock].mean(axis=1)
            lambda_cen_hires = get_GlobalLambdas_hires(iblock).mean(axis=1)
            mshape = lambda_cen_hires.shape+v_array.shape
            # apply Doppler shift due to LoS velocity
            w_mesh = np.outer(lambda_cen_hires, 1./(1.+ v_array)).flatten()
            # evaluate SED (phot/s/cm2) at observer frame (w/ LoS velocity)
            sed_mesh = (sed(w_mesh)*self.dwave[iblock]/self.Nsub[iblock]).reshape(mshape)
            dc = sed_mesh*i_array[np.newaxis,:,:]/(1+v_array[np.newaxis,:, :])
            # reduce at low resolution grid
            indices = np.arange(0, self.mNlam_hires[iblock], self.Nsub[iblock])
            #dc_array.append(np.add.reduceat(dc, indices, axis=0))
            dc_array.append(np.asarray([np.sum(dc[x:x+self.Nsub[iblock], :, :], axis=0) for x in indices]))
        # build Doppler-shifted datacube
        # self.lambda_cen = observed frame lambda grid
        # w_mesh = rest frame wavelengths evaluated on observed frame grid
        # To make energy conserved, dc_array in units of
        # photons / (s cm2)
        # w_mesh = np.outer(lambda_cen, 1./(1.+ v_array))
        # w_mesh = w_mesh.reshape(lambda_cen.shape+v_array.shape)
        # # photons/s/cm2 in the 3D grid
        # dc_array = sed(w_mesh.flatten()) * self._dl
        # dc_array = dc_array.reshape(w_mesh.shape) * \
        #                 i_array[np.newaxis, :, :] /\
        #                 (1+v_array[np.newaxis, :, :])

        return dc_array, gal, sed

    # def _construct_model_datacube(self, theta_pars, meta, v_array, i_array, gal):
    #     '''
    #     Create the model datacube from model slices, using the evaluated
    #     velocity and intensity maps, SED, etc.

    #     theta_pars: dict
    #         A dict of the sampled mcmc params for both the velocity
    #         map and the tranformation matrices
    #     v_array: np.array (2D)
    #         The vmap evaluated at image pixel positions for sampled pars.
    #         (Must be normalzied)
    #     i_array: np.array (2D)
    #         The imap evaluated at image pixel positions for sampled pars
    #     cont_array: np.array (2D)
    #         The imap of the fitted or modeled continuum
    #     datacube: DataCube
    #         Datavector datacube truncated to desired lambda bounds
    #     '''
    #     lambda_cen = np.mean(self._lambdas, axis=1)
    #     sed = self.setup_sed(meta)
    #     # build Doppler-shifted datacube
    #     # self.lambda_cen = observed frame lambda grid
    #     # w_mesh = rest frame wavelengths evaluated on observed frame grid
    #     # To make energy conserved, dc_array in units of
    #     # photons / (s cm2)
    #     w_mesh = np.outer(lambda_cen, 1./(1.+ v_array))
    #     w_mesh = w_mesh.reshape(lambda_cen.shape+v_array.shape)
    #     # photons/s/cm2 in the 3D grid
    #     dc_array = sed(w_mesh.flatten()) * self._dl
    #     dc_array = dc_array.reshape(w_mesh.shape) * \
    #                     i_array[np.newaxis, :, :] /\
    #                     (1+v_array[np.newaxis, :, :])

    #     return dc_array, sed

    @classmethod
    def setup_sed(cls, meta):
        return emission.ObsFrameSED(meta['sed'])

    @classmethod
    def setup_imap(cls, theta_pars, meta):
        '''
        theta_pars: dict
            A dict of the sampled mcmc params for both the velocity
            map and the tranformation matrices
        meta: MetaPars
            Updated MetaPars that record necessary information
        '''

        # Need to check if any basis func parameters are
        # being sampled over

        imap_pars = deepcopy(meta['intensity'])
        imap_type = imap_pars.pop('type')
        imap_pars['theory_Nx'] = meta['model_dimension']['Nx']
        imap_pars['theory_Ny'] = meta['model_dimension']['Ny']
        imap_pars['scale'] = meta['model_dimension']['scale']

        return intensity.build_intensity_map(imap_type, None, imap_pars)


def get_likelihood_types():
    return LIKELIHOOD_TYPES

# NOTE: This is where you must register a new likelihood model
LIKELIHOOD_TYPES = {
    'default': DataCubeLikelihood,
    'datacube': DataCubeLikelihood,
    # 'grism': GrismLikelihood,
    'grism': GrismLikelihood,
    'fiber': FiberLikelihood,
    }

def build_likelihood_model(name, parameters, datavector, **kwargs):
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
        likelihood = LIKELIHOOD_TYPES[name](parameters, datavector, **kwargs)
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
