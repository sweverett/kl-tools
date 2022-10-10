from abc import abstractmethod
import numpy as np
import os
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

import utils
import priors
import intensity
from parameters import Pars, MetaPars
# import parameters
from velocity import VelocityMap
from cube import DataVector, DataCube
import likelihood
import grism_modules/grism as grism

import ipdb

parser = ArgumentParser()

parser.add_argument('--show', action='store_true', default=False,
                    help='Set to show test plots')
parser.add_argument('--test', action='store_true', default=False,
                    help='Set to run tests')

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
    def __init__(self, parameters, datavector, config_list):
        ''' Initialization

        Besides the usual likelihood.LogLikelihood initialization, we further
        initialize the grism.GrismModelCube object, for the sake of speed
        '''
        super(GrismLikelihood, self).__init__(parameters, datavector)
        # init with an empty model cube
        _lrange = self.meta['model_dimension']['lambda_range']
        _dl = self.meta['model_dimension']['lambda_res']
        Nspec = len(np.arange(_lrange[0], _lrange[1], _dl))
        Nx = self.meta['model_dimension']['Nx']
        Ny = self.meta['model_dimension']['Ny']
        self.modelcube = GrismModelCube(self.parameters, 
            datacube=np.zeros([Nspec, Nx, Ny]))
        self.modelcube.set_obs_methods(datavector.get_config_list())

        return

    def _log_likelihood(self, theta, datavector, model):
        '''
        theta: list
            Sampled parameters, order defined by pars_order
        datavector: DataVector
            The grism datavector
        model: DataCube
            The model datacube object, truncated to desired lambda bounds
        '''

        # a (Nspec, Nx*Ny, Nx*Ny) inverse covariance matrix for the image pixels
        inv_cov = self._setup_inv_cov_list(datavector)

        loglike = 0

        # can figure out how to remove this for loop later
        # will be fast enough with numba anyway
        for i in range(datavector.Nobs):

            _img = model.observe(datavector.get_config(idx=i))
            diff = (datavector.get_data(i) - _img)
            chi2 = diff**2/inv_cov[i]

            loglike += -0.5*chi2

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

        # create grid of pixel centers in image coords
        X, Y = utils.build_map_grid(Nx, Ny)

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

        # get both the emission line and continuum image
        _pars = self.meta.copy_with_sampled_pars(theta_pars)
        _pars['run_options']['imap_return_gal'] = True
        i_array, gal = imap.render(theta_pars, datavector, _pars)

        model_datacube = self._construct_model_datacube(
            theta_pars, v_array, i_array, gal
            )

        return model_datacube

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

        model_datacube = GrismModelCube(self.parameters,
            datacube=dc_array, gal=gal, sed=sed)

        return model_datacube

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
