import numpy as np
from astropy.io import fits
from astropy.table import Table, join, hstack
import os
import sys
import pickle
import fitsio
from astropy.table import Table
from scipy.sparse import identity, dia_matrix
import matplotlib.pyplot as plt
import galsim as gs
import pathlib
import re
import astropy.wcs as wcs
import astropy.units as u
import astropy.constants as constants
from argparse import ArgumentParser
from time import time
from mpi4py import MPI

sys.path.insert(0, '../')
import utils as utils
#import parameters as parameters
import cube as cube
from emission import EmissionLine, LINE_LAMBDAS, SED
import kltools_grism_module_2 as m

import ipdb

parser = ArgumentParser()

parser.add_argument('--show', action='store_true', default=False,
                    help='Set to show test plots')
parser.add_argument('--test', action='store_true', default=False,
                    help='Set to run tests')

### Grism-specific pars here

class GrismPars(cube.CubePars):

    # update the required/optional/generated fields
    _req_fields = ['model_dimension', 'sed', 'intensity', 'velocity', 'priors']
    _opt_fields = ['obs_conf',]
    _gen_fields = ['shape', 'pix_scale', 'bandpasses']

    def __init__(self, pars, obs_conf=None):
        '''
        pars: dict
            Dictionary of meta pars for the GrismCube. Note that this dict has
            different convention than the parent class `cube.CubePars`. The 
            initialization step will translate the dictionary.

            An example structure of pars dict:
            <required fields>
            - model_dimension
                - Nx
                - Ny
                - scale
                - lambda_range
                - lambda_res
                - lambda_unit
            - sed
                - template
                - wave_type
                - flux_type
                - z
                - spectral_range
                - obs_cont_norm
                - lines
                - line_sigma_int
            - intensity
                - type
                - flux
                - hlr
            - velocity
                - model
                - v0
                - vcirc
                - rscale
            - priors
            <optional fields>
            - obs_conf
                - type
                - bandpass
                - instname
                - Nx
                - Ny
                - pixscale
                - diameter
                - exptime
                - gain
                - noisetyp
                - skylevel
                - rdnoise
                - addnoise
                - psftype
                - psffwhm
                - Rspec
                - dispang
                - offset
            <generated fields>
            - shape
            - pix_scale
            - bandpasses
                - lambda_blue
                - lambda_red
                - dlambda
                - throughput
                - zp
                - unit
                - file
        '''
        self._check_pars(pars)
        if obs_conf is None and pars.get('obs_conf', None) is None:
            raise ValueError('Observation configuration is not set!')
        if obs_conf is not None:
            print(f'Overwrite the observation configuration')
            utils.check_type(obs_conf, 'obs_conf', dict)
            pars['obs_conf'] = obs_conf
        # tweak the pars dict such that it fits into `cube.CubePars` required 
        # fields
        # need to check whether the fields of `cube.CubePars` are consistent
        # across branches
        _lrange = pars['model_dimension']['lambda_range']
        _Nlam = (_lrange[1]-_lrange[0])/pars['model_dimension']['lambda_res']
        _Nlam = np.ceil(_Nlam)
        pars['shape'] = np.array([_Nlam, 
                         pars['model_dimension']['Nx'], 
                         pars['model_dimension']['Ny']], dtype=int)
        pars['pix_scale'] = pars['model_dimension']['scale']
        pars['bandpasses'] = {
            'lambda_blue': _lrange[0],
            'lambda_red': _lrange[1],
            'dlambda': pars['model_dimension']['lambda_res'],
            'unit': pars['model_dimension']['lambda_unit'],
        }
        if pars['obs_conf'].get('BANDPASS', None) is not None:
            pars['bandpasses']['file'] = pars['obs_conf']['BANDPASS']
        else:
            pars['bandpasses']['throughput'] = 1.0

        # Call parent class initialization
        super(GrismPars, self).__init__(pars)

        self.obs_index = pars['obs_conf']['OBSINDEX']
        # update the obs_conf with the theoretical model cube parameters
        self.pars['obs_conf']['MDNAXIS1'] = pars['shape'][0]# Nlam
        self.pars['obs_conf']['MDNAXIS2'] = pars['shape'][1]# Nx
        self.pars['obs_conf']['MDNAXIS3'] = pars['shape'][2]# Ny
        self.pars['obs_conf']['MDSCALE'] = pars['pix_scale']

        return

    # quick approach to the obs_conf 
    @property
    def conf(self):
        return self.pars['obs_conf']

    # 2d ndarray version of the bandpass
    @property
    def bp_array(self):
        self._bp_array = np.zeros(self.lambdas.shape)
        for i,_bp in enumerate(self.bandpasses):
            self._bp_array[i][0] = _bp(self.lambdas[i][0])
            self._bp_array[i][1] = _bp(self.lambdas[i][1])
        return self._bp_array

class GrismModelCube(cube.DataCube):
    ''' Subclass DataCube for Grism modeling.

    '''
    @property
    def bp_array(self):
        return self.pars.bp_array

    def observe(self, force_noise_free=True):
        ''' get the simulated observation image using the C++ routine

        Arguments needed for C++ routine
            - int index: index of the `disperser_helper` to call
            - ndarray theory_data: theoretical 3d model data cube array
            - ndarray image: the simulated image array
            - ndarray lambdas: the 2D wavelength grid
            - ndarray bandpasses: the 2D bandpass grid
        '''
        # get simulated image
        _img = np.zeros([self.pars.conf['NAXIS2'], self.pars.conf['NAXIS1']],
                       dtype=np.float64, order='C')
        m.get_image(self.pars.obs_index, self.data, _img, 
                    self.lambdas, self.bp_array)
        img = gs.Image(_img, dtype=np.float64, scale=self.pars.conf['PIXSCALE'])
        # set psf
        self.set_psf()
        psf = self.get_psf()
        if psf is not None:
            _gal = gs.InterpolatedImage(img, scale=self.pars.conf['PIXSCALE'])
            gal = gs.Convolve([_gal, psf])
            img = gal.drawImage(
                nx=self.pars.conf['NAXIS1'], ny=self.pars.conf['NAXIS2'], 
                scale=self.pars.conf['PIXSCALE'])
        # set noise
        if force_noise_free:
            return img.array, None
        else:
            self.set_noise()
            noise = self.get_noise()
            img_wN = img.copy()
            img_wN.addNoise(noise)
            noise_img = img_wN - img
            if self.pars.conf.get('ADDNOISE', True):
                return img_wN.array, noise_img.array
            else:
                return img.array, noise_img.array

    def set_data(self, datacube):
        ''' update the data contained in the GrismModelCube

        This can be used for a quick method to get observed image
        '''
        _ds = datacube.shape
        assert _ds == self._data.shape, f'data shape {_ds}'+\
            f' is inconsistent with {self._data.shape}!'
        self._data = datacube
        return

    def stack(self, axis=0):
        return np.sum(self._data, axis=axis)

    def set_psf(self, **kwargs):
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
        _type = self.pars.conf.get('PSFTYPE', 'none').lower()
        if _type != 'none':
            if _type == 'airy':
                lam = kwargs.get('lam', 1000) # nm
                scale_unit = kwargs.get('scale_unit', gs.arcsec)
                self.pars['psf'] = gs.Airy(
                    lam=lam, diam=self.pars.conf['DIAMETER']/100,
                    scale_unit=scale_unit)
            elif _type == 'airy_mean':
                scale_unit = kwargs.get('scale_unit', gs.arcsec)
                self.pars['psf'] = gs.Airy(self.pars.conf['PSFFWHM']/1.029, 
                    scale_unit=scale_unit)
                #lam = kwargs.get("lam_mean", 1000) # nm
                #self.pars['psf'] = gs.Airy(
                #    lam=lam, diam=self.pars.conf['DIAMETER']/100,
                #    scale_unit=scale_unit)
            elif _type == 'moffat':
                beta = self.pars.conf.get('PSFBETA', 2.5)
                fwhm = self.pars.conf.get('PSFFWHM', 0.5)
                self.pars['psf'] = gs.Moffat(beta=beta, fwhm=fwhm)
            else:
                raise ValueError(f'{_type} has not been implemented yet!')
        else:
            self.pars['psf'] = None
        return

    def set_noise(self):
        ''' Generate image noise based on parameter settings

        Outputs:
        ========
        noise: GalSim Noise object
        '''
        random_seed = self.pars.conf.get('RANDSEED', int(time()))
        rng = gs.BaseDeviate(random_seed+1)

        _type = self.pars.conf.get('NOISETYP', 'ccd').lower()
        if _type == 'ccd':
            sky_level = self.pars.conf.get('SKYLEVEL', 0.65*1.2)
            read_noise = self.pars.conf.get('RDNOISE', 8.5)
            gain = self.pars.conf.get('GAIN', 1.0)
            exp_time = self.pars.conf.get('EXPTIME', 1.0)
            noise = gs.CCDNoise(rng=rng, gain=gain, 
                                read_noise=read_noise, 
                                sky_level=sky_level*exp_time/gain)
        elif _type == 'gauss':
            sigma = self.pars.conf.get('NOISESIG', 1.0)
            noise = gs.GaussianNoise(rng=rng, sigma=sigma)
        elif _type == 'poisson':
            sky_level = self.pars.conf.get('SKYLEVEL', 0.65*1.2)
            noise = gs.PoissonNoise(rng=rng, sky_level=sky_level)
        else:
            raise ValueError(f'{_type} not implemented yet!')
        self.pars['noise'] = noise
        return

    def get_noise(self):
        if 'noise' in self.pars:
            return self.pars['noise']
        else:
            return None

class GrismModelCube_lite(cube.DataVector):
    ''' Subclass DataCube for Grism modeling.

    Note: cube.DataCube interface

    methods:
    ========
        - __init__(data, pars=None, weights=None, masks=None,
                 pix_scale=None, bandpasses=None)
                the `data` is gonna to be returned by setup_model method is likelihood class
        - _check_shape_params()
        - get_sed(line_index=None, line_pars_update=None)
        - stack()
        - _set_maps(maps, map_type)
        - copy()
        - get_inv_cov_list()
        - write(outfile)

        - [NEW] observe(self, observe_params) # in the long run, observe_params gonna to be very close to roman grism pipeline convention. 
    attributes:
    ===========
        - 

    To fit into the cube.DataCube interface, what you need:

    '''

    def __init__(self, pars, gal=None, sed=None):
        ''' Initialize a GrismDataCube object

        Note that the interface is not finalize and is submit to changes

        pars: Pars
            parameters that hold meta parameter information needed
        datacube: np.ndarray, (Nspec, Nx, Ny)
            3D array of theoretical data cube
        imap: np.ndarray, 2D
            the intensity profile of the galaxy
        sed: galsim.SED object
            the SED of the galaxy
        '''

        # build CubePars
        _cubepars_dict = {}
        _cubepars_dict['pix_scale'] = pars.meta['model_dimension']['scale']
        # unity bandpass
        _cubepars_dict['bandpasses'] = {
            'lambda_blue': pars.meta['model_dimension']['lambda_range'][0],
            'lambda_red': pars.meta['model_dimension']['lambda_range'][1],
            'dlambda': pars.meta['model_dimension']['lambda_res'],
            'throughput': 1.0,
            'unit': pars.meta['model_dimension']['lambda_unit'],
        }
        cubepars = cube.CubePars(_cubepars_dict)
        # init parent DataCube class, with 3D datacube and CubePars
        #super(GrismModelCube, self).__init__(datacube, pars=cubepars)
        # also gs.Chromatic object for broadband image
        self.gal = gal
        self.sed = sed
        self.set_obs_method = False
        return

    def set_obs_methods(self, config_list):
        
        #if not self.set_obs_method:
        self.config_list = config_list
        _disperse_helper_index_map = []
        _grism_count = 0
        _bandpasses = []
        for config in config_list:
            _bp = gs.Bandpass(config.get('bandpass'), 
                wave_type=config.get('wave_type', 'nm'))
            _bandpasses.append(_bp)
            _bp_list = _bp(np.array(self.lambdas))
            
            if config.get('type')!='grism':
                _disperse_helper_index_map.append(-1)
            else:
                # set the grism disperse helper
                # add model information
                config['model_Nx'] = self.Nx
                config['model_Ny'] = self.Ny
                config['model_Nlam'] = self.Nspec
                config['model_scale'] = self.pix_scale
                #_helper = m.DisperseHelper(config, np.array(self.lambdas), _bp_list)
                #if rank==0:
                #if True:
                #    m.add_disperser(config, np.array(self.lambdas), _bp_list)
                _disperse_helper_index_map.append(_grism_count)
                _grism_count += 1
        self.disperse_helper_index_map = _disperse_helper_index_map
        self.bandpass_list = _bandpasses
        self.set_obs_method = True
        return 

    def observe(self, index, force_noise_free=True):
        ''' Simulate observed image given `observe_parmas` which specifies all
        the information you need.

        `config` is returned by `GrismDataVector` object.
        In the long run, `observe_params` gonna to be very close to Anahita's 
        Roman grism simulation convention, but in the short-term, it will just
        be a dictionary.
        '''
        _type = self.config_list[index].get('type', 'none').lower()
        if _type == 'grism':
            return self._get_grism_data(index, 
                force_noise_free=force_noise_free)
        elif _type == 'photometry':
            return self._get_photometry_data(index, 
                force_noise_free=force_noise_free)
        else:
            print(f'{_type} not supported by GrismModelCube!')
            exit(-1)

    def _get_photometry_data(self, index, force_noise_free = True):
        #print(f'WARNING: have not test normalization yet')
        #start_time = time()*1000
        # self._data is in units of [photons / (s cm2)]
        # No, get chomatic directly, from GrismModelCube
        gal_chromatic = self.gal * self.sed.spectrum
        config = self.config_list[index]
        # convolve with PSF
        psf = self._build_PSF_model(config, 
              lam=self.bandpass_list[index].calculateEffectiveWavelength())
        if psf is not None:
            gal_chromatic = gs.Convolve([gal_chromatic, psf])
        photometry_img = gal_chromatic.drawImage(
                                nx=config['Nx'], ny=config['Ny'], 
                                scale=config['pix_scale'], method='auto',
                                area=np.pi*(config['diameter']/2.0)**2, 
                                exptime=config['exp_time'],
                                gain=config['gain'],
                                bandpass=self.bandpass_list[index])
        # apply noise
        #print("----- photometry image | %.2f ms -----" % (time()*1000 - start_time))
        if force_noise_free:
            return photometry_img.array, None
        else:
            noise = self._getNoise(config)
            photometry_img_withNoise = photometry_img.copy()
            photometry_img_withNoise.addNoise(noise)
            noise_img = photometry_img_withNoise - photometry_img
            assert (photometry_img_withNoise.array is not None), \
                    "Null photometry data"
            assert (photometry_img.array is not None), "Null photometry data"
            if config.get('apply_to_data', True):
                #print("[ImageGenerator][debug]: add noise")
                return photometry_img_withNoise.array, noise_img.array
            else:
                #print("[ImageGenerator][debug]: noise free")
                return photometry_img.array, noise_img.array

    def _get_grism_data(self, index, force_noise_free=True):
        # prepare datacube and output
        config = self.config_list[index]
        grism_img_array = np.ones([config['Ny'], config['Nx']], 
            dtype=np.float64, order='C')
        # call c++ routine to disperse
        #start_time = time()*1000
        m.get_dispersed_image(self.disperse_helper_index_map[index], grism_img_array)
        #self.disperse_helper[index].getDispersedImage(self._data, grism_img_array)
        # wrap with noise & psf
        grism_img = gs.Image(
            grism_img_array,
            dtype = np.float64, scale=config['pix_scale'], 
            )
        # convolve with achromatic psf, if required
        psf = self._build_PSF_model(config, lam_mean=np.mean(self.lambdas))
        if psf is not None:
            _gal = gs.InterpolatedImage(grism_img, scale=config['pix_scale'])
            grism_gal = gs.Convolve([_gal, psf])
            grism_img = grism_gal.drawImage(nx=config['Nx'], ny=config['Ny'], 
                                            scale=config['pix_scale'])
        #print("----- disperse image | %.2f ms -----" % (time()*1000 - start_time))
        # apply noise
        if force_noise_free:
            return grism_img.array, None
        else:
            noise = self._getNoise(config)
            grism_img_withNoise = grism_img.copy()
            grism_img_withNoise.addNoise(noise)
            noise_img = grism_img_withNoise - grism_img
            assert (grism_img_withNoise.array is not None), "Null grism data"
            assert (grism_img.array is not None), "Null grism data"
            #print("----- %s seconds -----" % (time() - start_time))
            if self.apply_to_data:
                #print("[GrismGenerator][debug]: add noise")
                return grism_img_withNoise.array, noise_img.array
            else:
                #print("[GrismGenerator][debug]: noise free")
                return grism_img.array, noise_img.array

    def set_data(self, gal, sed):
        ''' update the data contained in the GrismModelCube

        This can be used for a quick method to get observed image
        '''
        self.gal = gal 
        self.sed = sed

    def stack(self, axis=0):
        pass 

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

class GrismDataVector(cube.DataVector):
    '''
    Class that defines grism data set, which generally has multiple grism and 
    broadband image files.

    The data structure of this class is

    - self.header: dict; info about the data set overall
        keywords:
            - NEXTEN  : number of extensions
            - OBSNUM  : number of observations, should be NEXTENSION/2

    - self.data: list of np.ndarray; the data image

    - self.noise: list of np.ndarray; the noise image

    - self.data_header: list of dict; the info about each data observation
        keywords:
            - INSTNAME: instrument name
            - OBSTYPE : observation type
            - BANDPASS: bandpass filename
            - MDNAXIS1: number of pixels along lambda-axis
            - MDNAXIS2: number of pixels along x
            - MDNAXIS3: number of pixels along y
            - MDSCALE : 3D model pixel scale
            - CONFFILE: instrument configuration file
            - NAXIS   : number of axis
            - NAXIS1  : number of pixels along axis 1
            - NAXIS2  : number of pixels along axis 2
            - PIXSCALE: observed image pixel scale
            - CRPIX1  : x-coordinate of the reference pixel
            - CRPIX2  : y-coordinate of the reference pixel
            - CRVAL1  : first axis value at the reference pixel
            - CRVAL2  : second axis value at the reference pixel
            - CTYPE1  : the coordinate type for the first axis
            - CTYPE2  : the coordinate type for the second axis
            - CD1_1   : partial of the 1st axis w.r.t. x
            - CD1_2   : partial of the 1st axis w.r.t. y
            - CD2_1   : partial of the 2nd axis w.r.t. x
            - CD2_2   : partial of the 2nd axis w.r.t. y
            - DIAMETER: aperture diameter in cm
            - EXPTIME : exposure time in seconds
            - GAIN    : detector gain
            - NOISETYP: noise model type
            - SKYLEVEL: sky level
            - RDNOISE : read noise in e/s
            - ADDNOISE: flag for adding noise or not
            - PSFTYPE : PSF model type
            - PSFFWHM : FWHM of the PSF model
            - RSPEC   : spectral resolution
            - DISPANG : dispersion angle in rad
            - OFFSET  : offset of the dispersed frame center to the 
                        pointing center
    '''
    _default_header = {'NEXTEN': 6, 'OBSNUM': 3}
    _default_data_header = {
        "OBSINDEX": 0,
        "INSTNAME": "none",
        "OBSTYPE" : 0,
        "BANDPASS": "none",
        #"MDNAXIS1": 0,
        #"MDNAXIS2": 0,
        #"MDNAXIS3": 0,
        #"MDSCALE" : 1.0,
        #"MDCD1_1" : 1.0, # arbitrary unit
        #"MDCD1_2" : 0.0,
        #"MDCD2_1" : 0.0,
        #"MDCD2_2" : 1.0,
        "CONFFILE": "none",
        "NAXIS"   : 2,
        "NAXIS1"  : 0,
        "NAXIS2"  : 0,
        "CRPIX1"  : 0,
        "CRPIX2"  : 0,
        "CRVAL1"  : 0.0,
        "CRVAL2"  : 0.0,
        "CTYPE1"  : "RA-TAN",
        "CTYPE2"  : "DEC-TAN",
        "PIXSCALE": 1.0,
        "CD1_1"   : 1.0,
        "CD1_2"   : 0.0,
        "CD2_1"   : 0.0,
        "CD2_2"   : 1.0,
        "DIAMETER": 100.0,
        "EXPTIME" : 0.0,
        "GAIN"    : 1.0,
        "NOISETYP": "none",
        "SKYLEVEL": 0.0,
        "RDNOISE" : 0.0,
        "ADDNOISE": False,
        "PSFTYPE" : "none",
        "PSFFWHM" : 0.0,
        "RSPEC"   : 0.0,
        "DISPANG" : 0.0,
        "OFFSET"  : 0.0,
    }
    _required_keys = ["OBSTYPE", "INSTNAME", "BANDPASS", "NAXIS", "NAXIS1",
        "NAXIS2", "PIXSCALE", "DIAMETER", "EXPTIME", "GAIN", "NOISETYP", 
        "SKYLEVEL", "RDNOISE", "ADDNOISE", "PSFTYPE", "PSFFWHM", "RSPEC",
        "DISPANG", "OFFSET"]

    def __init__(self, 
        file=None,
        header = None, data_header = None, data = None, noise = None,
        ):
        ''' Initialize the `GrismDataVector` class object
        either from a fits file, or from a series of input parameters

        '''
        if file is not None:
            if header!=None or data!=None or data_header!=None or noise!=None:
                print("header, data, data_header and noise can't be set when file is set!")
                exit(-1)
            else:
                self.from_fits(file)
        else:
            self.header = header
            self.data = data
            self.data_header = data_header
            self.noise = noise
        self.Nobs = self.header['OBSNUM']

        return

    def get_inv_cov_list(self):
        inv_cov_list = [self.get_inv_cov(i) for i in range(len(self.noise))]
        return inv_cov_list

    def get_inv_cov(self, idx):
        return 1/self.noise[idx]**2

    def get_data(self, idx=0):
        ''' Return the idx image in the data set
        '''
        return self.data[idx]

    def get_noise(self, idx=0):
        ''' Return the idx noise image in the data set
        '''
        return self.noise[idx]

    def from_fits(self, file=None):
        ''' Read `GrismDataVector` from a fits file
        '''
        hdul = fits.open(file)
        self.header = hdul[0].header
        self.Nobs = self.header['OBSNUM']
        self.data = []
        self.data_header = []
        self.noise = []
        for i in range(self.Nobs):
            self.data.append(hdul[1+2*i].data)
            self.data_header.append(hdul[1+2*i].header)
            self.noise.append(hdul[2+2*i].data)
            
        hdul.close()

    def get_config(self, idx=0):
        ''' Get the configuration of the idx image

        The output of this function can be passed to `GrismModelCube:observe`
        method to generate realistic grism/photometry images.

        Note that so far this would only be a wrapper of data_header, but 
        in the future we can add more realistic attributes.
        '''
        _h = self.data_header[idx].copy()
        for k in _required_keys:
            if(not k in _h):
                _h[k] = _default_data_header[k]
        _h["OBSINDEX"] = idx
        return _h

    def get_config_list(self):
        config_list = [self.get_config(i) for i in range(len(self.noise))]
        return config_list

    def stack(self):
        i = 0
        while i < len(self.data):
            if self.data_header[i]["OBSTYPE"] == 0:
                return self.data[i]
        return None


class GrismSED(SED):
    '''
    This class describe the obs-frame SED template of a source galaxy, includ-
    ing emission lines and continuum components.
    This is mostly a wrapper of the galsim.SED class object
    
    Note that this class should only capture the intrinsic properties of
    the galaxy SED, like
        - redshift
        - continuum and emission line flux density
        - emission line width
        - [dust?]
    Other extrinsic properties, like
        - sky background
        - system bandpass
        - spectral resolution and dispersion
        - exposure time
        - collecting area
        - pixelization
        - noise
    will be implemented in mock observation module
    '''
    ### settings for the whole class
    # default parameters
    _default_pars = {
        'template': '../../data/Simulation/GSB2.spec',
        'wave_type': 'Ang',
        'flux_type': 'flambda',
        'z': 0.0,
        'spectral_range': (50, 50000), # nm
        # obs-frame continuum normalization (nm, erg/s/cm2/nm)
        'obs_cont_norm': (400, 0.),
        # spectral resolution at 1 micron, assuming dispersion per pixel
        #'resolution': 3000,
        # a dict of line names and obs-frame flux values (erg/s/cm2)
        'lines': {'Ha': 1e-15},
        # intrinsic linewidth in nm
        'line_sigma_int': {'Ha': 0.5,},
        #'line_hlr': (0.5, Unit('arcsec')),
        'thin': -1,
    }
    # units conversion
    _h = constants.h.to('erg s').value
    _c = constants.c.to('nm/s').value
    # build-in emission line species and info
    _valid_lines = {k:v.to('nm').value for k,v in LINE_LAMBDAS.items()}
    
    def __init__(self, pars):
        '''
        Initialize SED class object with parameters dictionary
        '''
        self.pars = GrismSED._default_pars.copy()
        self.updatePars(pars)
        _con = self._addContinuum()
        _emi = self._addEmissionLines()
        self.spectrum = _con + _emi
        if self.pars['thin'] > 0:
            self.spectrum = self.spectrum.thin(rel_err=self.pars['thin'])
        super(GrismSED, self).__init__(
            self.pars['spectral_range'][0], self.pars['spectral_range'][1], 
            3000, 'nm')

    def __call__(self, wave):
        if isinstance(wave, u.Quantity):
            wave = wave.to(self.spectrum.wave_type).value
        return self.spectrum(wave)
        
        
    def updatePars(self, pars):
        '''
        Update parameters
        '''
        for key, val in pars.items():
            self.pars[key] = val

    def _addContinuum(self):
        '''
        Build and return continuum GalSim SED object
        '''
        template = self.pars['template']
        if not os.path.isfile(template):
            raise OSError(f'Can not find template file {template}!')
        # build GalSim SED object out of template file
        _template = np.genfromtxt(template)
        _table = gs.LookupTable(x=_template[:,0], f=_template[:,1],)
        SED = gs.SED(_table, 
                         wave_type=self.pars['wave_type'], 
                         flux_type=self.pars['flux_type'],
                         redshift=self.pars['z'],
                         _blue_limit=self.pars['spectral_range'][0],
                         _red_limit=self.pars['spectral_range'][1])
        # normalize the SED object at observer frame
        # erg/s/cm2/nm -> photons/s/cm2/nm
        # TODO: add more flexible normalization parametrization
        norm = self.pars['obs_cont_norm'][1]*self.pars['obs_cont_norm'][0]/\
            (GrismSED._h*GrismSED._c)
        return SED.withFluxDensity(target_flux_density=norm, 
                                   wavelength=self.pars['obs_cont_norm'][0])
    
    def _addEmissionLines(self):
        '''
        Build and return Gaussian emission lines GalSim SED object
        '''
        # init LookupTable for rest-frame SED
        lam_grid = np.arange(self.pars['spectral_range'][0]/(1+self.pars['z']),
                             self.pars['spectral_range'][1]/(1+self.pars['z']), 
                             0.1)
        flux_grid = np.zeros(lam_grid.size)
        # Set emission lines: (specie, observer frame flux)
        all_lines = GrismSED._valid_lines.keys()
        norm = -1
        for line, flux in self.pars['lines'].items():
            # sanity check
            rest_lambda = np.atleast_1d(GrismSED._valid_lines[line])
            flux = np.atleast_1d(flux)
            line_sigma_int = np.atleast_1d(self.pars['line_sigma_int'][line])
            if rest_lambda is None:
                raise ValueError(f'{line} is not a valid emission line! '+\
                        f'For now, line must be one of {all_lines}')
            else:
                assert rest_lambda.size == flux.size, f'{line} has'+\
                f' {rest_lambda.size} lines but {flux.size} flux are provided!'
            # build rest-frame f_lambda SED [erg s-1 cm-2 nm-1]
            # then, redshift the SED. The line width will increase automatically
            for i,cen in enumerate(rest_lambda):
                _lw_sq = line_sigma_int[i]**2
                # erg/s/cm2 -> erg/s/cm2/nm
                _norm = flux[i]/np.sqrt(2*np.pi*_lw_sq)
                flux_grid += _norm * np.exp(-(lam_grid-cen)**2/(2*_lw_sq))
                # also, calculate normalization factor for obs-frame spectrum
                # convert flux units: erg/s/cm2/nm -> photons/s/cm2/nm
                # only one emission line needed
                if(norm<0):
                    norm_lam = cen*(1+self.pars['z'])
                    norm = flux[i]*norm_lam/(GrismSED._h*GrismSED._c)/\
                                np.sqrt(2*np.pi*_lw_sq*(1+self.pars['z'])**2)
            
        _table = gs.LookupTable(x=lam_grid, f=flux_grid,)
        SED = gs.SED(_table,
                         wave_type='nm',
                         flux_type='flambda',
                         redshift=self.pars['z'],)
        # normalize to observer-frame flux
        SED = SED.withFluxDensity(target_flux_density=norm, 
                                  wavelength=norm_lam)
        return SED

def initialize_observations(Nobs, pars_list, datavector=None):
    ''' Initialize the C++ simulate observation routines

    Inputs:
    =======
    - Nobs: int
        Number of observations
    - pars_list: list of `GrismPars` object
        The `GrismPars` object associated with each observation. C++ routines 
        will be initialized by the parameters therein.
    - datavector: `GrismDataVector` object
        Optional. Default is `None`. If set, the C++ routines will be
        initialized with the data vector (observed images) given in the data 
        vector. Otherwise, set with empty arrays.
    '''
    assert Nobs==len(pars_list), f'pars_list has {len(pars_list)} elements,'+\
                                 f' while should be {Nobs}!'
    if datavector is not None:
        assert Nobs == datavector.Nobs, f'datavector object should have '+\
        f'{Nobs} observations but has {datavector.Nobs}!'
    if(Nobs != m.get_Nobs()):
        if(m.get_Nobs() > 0):
            print(f'cleaning up {m.get_Nobs()} observations')
            m.clear_observation()
        for i,pars in enumerate(pars_list):
            if datavector is not None:
                _data = datavector.get_data(i)
                _noise = datavector.get_noise(i)
            else:
                _data = np.zeros([pars.conf['NAXIS2'], pars.conf['NAXIS1']])
                _noise = np.ones([pars.conf['NAXIS2'], pars.conf['NAXIS1']])
            m.add_observation(pars.conf, pars.lambdas, pars.bp_array,
                _data, _noise)
    assert (m.get_Nobs() == Nobs), "Inconsistent # of observations!"
    return

def main(args):

    testdir = utils.get_test_dir()
    testpath = pathlib.Path(os.path.join(testdir, 'test_data'))
    spec1dPath = testpath / pathlib.Path("spectrum_102021103.fits")
    spec3dPath = testpath / pathlib.Path("102021103_objcube.fits")
    catPath = testpath / pathlib.Path("MW_1-24_main_table.fits")
    emlinePath = testpath / pathlib.Path("MW_1-24_emline_table.fits")

    # Try initializing a datacube object with these paths.
    muse = MuseDataCube(
        cubefile=spec3dPath,
        specfile=spec1dPath,
        catfile=catPath,
        linefile=emlinePath
        )

    muse.set_line()

    outdir = os.path.join(utils.TEST_DIR, 'muse')
    utils.make_dir(outdir)

    import matplotlib.pyplot as plt
    # plt.subplots(4,4)
    for i in range(muse.Nspec):
        lam = muse.lambdas[i]
        # sci
        plt.subplot(muse.Nspec,3,3*i+1)
        plt.imshow(muse.data[i])
        plt.colorbar()
        plt.ylabel(f'({lam[0]:.1f},{lam[1]:.1f})')
        if i == 0:
            plt.title('sci')
        # wgt
        plt.subplot(muse.Nspec,3,3*i+2)
        plt.imshow(muse.weights[i])
        plt.colorbar()
        if i == 0:
            plt.title('wgt')
        # msk
        plt.subplot(muse.Nspec,3,3*i+3)
        plt.imshow(muse.masks[i])
        plt.colorbar()
        if i == 0:
            plt.title('msk')

    plt.gcf().set_size_inches(7,24)

    outfile = os.path.join(outdir, 'muse-datacube-truncated.png')
    plt.savefig(outfile, bbox_inches='tight', dpi=300)

    return 0
