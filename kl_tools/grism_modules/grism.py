import numpy as np
from astropy.io import fits
from astropy.table import Table, join, hstack
import os
import sys
import pickle
import fitsio
from copy import deepcopy
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
#from mpi4py import MPI

sys.path.insert(0, '../')
import utils as utils
import parameters
#from cube import CubePars
import emission
import kltools_grism_module_2 as m
from datavector import DataVector

try:
    import mpi4py
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
except:
    rank = 0
    size = 1
m.set_mpi_info(size, rank)

import ipdb

parser = ArgumentParser()

parser.add_argument('--show', action='store_true', default=False,
                    help='Set to show test plots')
parser.add_argument('--test', action='store_true', default=False,
                    help='Set to run tests')

### Grism-specific pars here

class GrismPars(parameters.CubePars):

    # update the required/optional/generated fields
    _req_fields = ['model_dimension', 'sed', 'intensity', 'velocity', 'priors']
    _opt_fields = ['obs_conf',]
    _gen_fields = ['shape', 'pix_scale', 'bandpasses']

    def __init__(self, pars, obs_conf=None):
        ''' Grism-related `Pars` obj
        Each `GrismPars` store the necessary parameters for one observation.

        Inputs:
        =======
        pars: dict obj
            Dictionary of meta pars for `GrismCube`. Note that this dict has
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
                See grism.GrismDataVector.data_header
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
        # TODO: extend blue and red limit if photometry data
        # assure the input dictionary obj has required fields & obs config
        self._check_pars(pars)
        _pars = deepcopy(pars) # avoid shallow-copy of a dict object
        if obs_conf is None and _pars.get('obs_conf', None) is None:
            raise ValueError('Observation configuration is not set!')
        if obs_conf is not None:
            print(f'GrismPars: Overwrite the observation configuration')
            utils.check_type(obs_conf, 'obs_conf', dict)
            _pars['obs_conf'] = deepcopy(obs_conf)
        self.is_dispersed = (_pars['obs_conf']['OBSTYPE'] == 2)
        # tweak the pars dict such that it fits into `cube.CubePars` required 
        # fields
        # need to check whether the fields of `cube.CubePars` are consistent
        # across branches
        _lrange = _pars['model_dimension']['lambda_range'][0]
        _Nlam = (_lrange[1]-_lrange[0])/_pars['model_dimension']['lambda_res']
        _Nlam = np.ceil(_Nlam)
        _pars['shape'] = np.array([_Nlam, 
                         _pars['model_dimension']['Ny'], 
                         _pars['model_dimension']['Nx']], dtype=int)
        _pars['pix_scale'] = _pars['model_dimension']['scale']
        _pars['bandpasses'] = {
            'lambda_blue': _lrange[0],
            'lambda_red': _lrange[1],
            'dlambda': _pars['model_dimension']['lambda_res'],
            'unit': _pars['model_dimension']['lambda_unit'],
        }
        if _pars['obs_conf'].get('BANDPASS', None) is not None:
            _pars['bandpasses']['file'] = _pars['obs_conf']['BANDPASS']
        else:
            _pars['bandpasses']['throughput'] = 1.0

        # Call parent class initialization
        super(GrismPars, self).__init__(_pars)

        self.obs_index = _pars['obs_conf']['OBSINDEX']
        # update the obs_conf with the theoretical model cube parameters
        self.pars['obs_conf']['MDNAXIS1'] = _pars['shape'][2]# Nx
        self.pars['obs_conf']['MDNAXIS2'] = _pars['shape'][1]# Ny
        self.pars['obs_conf']['MDNAXIS3'] = _pars['shape'][0]# Nlam
        self.pars['obs_conf']['MDSCALE'] = _pars['pix_scale']

        return

    # quick approach to the `obs_conf` observation configuration (1 obs) 
    @property
    def conf(self):
        return self.pars['obs_conf']

    # get the Nx2 band-pass array, which has the same dimension to `lambdas`
    @property
    def bp_array(self):
        self._bp_array = np.zeros(self.lambdas.shape)
        for i,_bp in enumerate(self._bandpasses):
            self._bp_array[i][0] = _bp(self.lambdas[i][0])
            self._bp_array[i][1] = _bp(self.lambdas[i][1])
        return self._bp_array

class GrismModelCube(DataVector):
    ''' The most updated `GrismModelCube` implementation
    This class wraps the theoretical cube after it is generated to produce 
    simulated images.
    It will use the latest C++ interface while process the theoretical cube 
    on-the-fly
    '''
    def __init__(self, Gpars):
        ''' Store the `GrismPars` object
        '''
        self.pars = Gpars
        return
        
    @property
    def bp_array(self):
        return self.pars.bp_array
    @property
    def conf(self):
        return self.pars.conf
    @property
    def lambdas(self):
        return self.pars.lambdas
    @property
    def effective_lambda(self):
        return np.sum(self.pars.lambdas[:,0]*self.pars.bp_array[:,0])/\
        np.sum(self.pars.bp_array[:,0])

    def observe(self, **kwargs):
        ''' Simulate observed image, can be either photometry or grism image
        kwargs include:
            - theory_cube: np.ndarray
                3D photon distribution for grism observation
            - force_noise_free: boolean
                Flag to skip noise evaluation. Set to True to completely ignore
            - datavector: DataVector derived object
                DataVector object for PSF from data, 
            - gal_phot: gs.GSObject derived class
                GalSim surface brightness object for photometry image
        '''
        #start_time = time()*1000

        ### Prepare PSF
        psf = self._build_PSF_model(self.conf, lam_mean=self.effective_lambda,
           datavector=kwargs.get("datavector", None))

        ### Get dispersed grism image
        if self.pars.is_dispersed:
            # prepare datacube and output, z-y-x indexing
            img_array = np.ones([self.conf['NAXIS2'], self.conf['NAXIS1']], 
                                dtype=np.float64, order='C')
            m.get_image(self.conf['OBSINDEX'], kwargs["theory_cube"], 
                img_array, self.lambdas, self.bp_array)
            img = gs.Image(img_array, dtype = np.float64, 
                scale=self.conf['PIXSCALE'],)
            # apply PSF to the dispersed grism image
            if psf is not None:
                try:
                    _gal = gs.InterpolatedImage(img,scale=self.conf['PIXSCALE'])
                    gal = gs.Convolve([_gal, psf])
                    img = gal.drawImage(nx=self.conf['NAXIS1'], 
                        ny=self.conf['NAXIS2'], scale=self.conf['PIXSCALE'])
                except gs.errors.GalSimValueError:
                    _ary_ = np.zeros([self.conf['NAXIS2'], self.conf['NAXIS1']])
                    _ary_[:,:] = np.inf
                    img = gs.Image(_ary_)
        ### Get photometry image
        else:
            gal = gs.Convolve([kwargs["gal_phot"], psf])
            try:
                img = gal.drawImage(nx=self.conf['NAXIS1'], 
                    ny=self.conf['NAXIS2'], scale=self.conf["PIXSCALE"])
            except gs.errors.GalSimFFTSizeError:
                _ary_ = np.zeros([self.conf['NAXIS2'], self.conf['NAXIS1']])
                _ary_[:,:] = np.inf
                img = gs.Image(_ary_)

        # fig, axes = plt.subplots(1,3)
        # axes[0].imshow(img.array)
        # axes[1].imshow(theory_cube.sum(axis=0))
        # axes[2].imshow(theory_cube.sum(axis=1))
        # plt.show()
        
        
        # if psf is not None:
        #     try:
        #         _gal = gs.InterpolatedImage(img, scale=self.conf['PIXSCALE'])
        #         gal = gs.Convolve([_gal, psf])
        #         img = gal.drawImage(nx=self.conf['NAXIS1'], 
        #             ny=self.conf['NAXIS2'], scale=self.conf['PIXSCALE'])
        #     except gs.errors.GalSimValueError:
        #         _ary_ = np.zeros([self.conf['NAXIS2'], self.conf['NAXIS1']])
        #         _ary_[:,:] = np.inf
        #         img = gal.Image(_ary_)
        
        #print("----- disperse image | %.2f ms -----" % (time()*1000 - start_time))

        ### apply noise
        if kwargs.get("force_noise_free", True):
            return img.array, None
        else:
            noise = self._getNoise(self.conf)
            img_withNoise = img.copy()
            img_withNoise.addNoise(noise)
            noise_img = img_withNoise - img
            assert (img_withNoise.array is not None), "Null data"
            assert (img.array is not None), "Null data"
            #print("----- %s seconds -----" % (time() - start_time))
            if self.conf['ADDNOISE']:
                #print("[GrismGenerator][debug]: add noise")
                return img_withNoise.array, noise_img.array
            else:
                #print("[GrismGenerator][debug]: noise free")
                return img.array, noise_img.array

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
        _type = config.get('PSFTYPE', 'none').lower()
        if _type != 'none':
            if _type == 'airy':
                lam = kwargs.get('lam', 1000) # nm
                scale_unit = kwargs.get('scale_unit', gs.arcsec)
                return gs.Airy(lam=lam, diam=config['DIAMETER']/100,
                                scale_unit=scale_unit)
            elif _type == 'airy_mean':
                scale_unit = kwargs.get('scale_unit', gs.arcsec)
                #return gs.Airy(config['psf_fwhm']/1.028993969962188, 
                #               scale_unit=scale_unit)
                lam = kwargs.get("lam_mean", 1000) # nm
                return gs.Airy(lam=lam, diam=config['DIAMETER']/100,
                                scale_unit=scale_unit)
            elif _type == 'airy_fwhm':
                loverd = config['PSFFWHM']/1.028993969962188
                scale_unit = kwargs.get('scale_unit', gs.arcsec)
                return gs.Airy(lam_over_diam=loverd, scale_unit=scale_unit)
            elif _type == 'moffat':
                beta = config.get('PSFBETA', 2.5)
                fwhm = config.get('PSFFWHM', 0.5)
                return gs.Moffat(beta=beta, fwhm=fwhm)
            elif _type == 'data':
                return kwargs["datavector"].get_PSF(self.conf["OBSINDEX"])
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
        random_seed = config.get('RANDSEED', int(time()))
        rng = gs.BaseDeviate(random_seed+1)

        _type = config.get('NOISETYP', 'ccd').lower()
        if _type == 'ccd':
            sky_level = config.get('SKYLEVEL', 0.65*1.2)
            read_noise = config.get('RDNOISE', 8.5)
            gain = config.get('GAIN', 1.0)
            exp_time = config.get('EXPTIME', 1.0)
            noise = gs.CCDNoise(rng=rng, gain=gain, 
                                read_noise=read_noise, 
                                sky_level=sky_level*exp_time/gain)
        elif _type == 'gauss':
            sigma = config.get('NOISESIG', 1.0)
            noise = gs.GaussianNoise(rng=rng, sigma=sigma)
        elif _type == 'poisson':
            sky_level = config.get('SKYLEVEL', 0.65*1.2)
            noise = gs.PoissonNoise(rng=rng, sky_level=sky_level)
        else:
            raise ValueError(f'{self.noise_type} not implemented yet!')
        return noise

class GrismDataVector(DataVector):
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
            - OBSTYPE : observation type; 0-photometry; 1-fiber, 2-grism
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
        "BANDPASS": None,
        #"MDNAXIS1": 0,
        #"MDNAXIS2": 0,
        #"MDNAXIS3": 0,
        #"MDSCALE" : 1.0,
        #"MDCD1_1" : 1.0, # arbitrary unit
        #"MDCD1_2" : 0.0,
        #"MDCD2_1" : 0.0,
        #"MDCD2_2" : 1.0,
        "CONFFILE": None,
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
        # default diameter-exptime-gain-bandpass setting -> flux scaling = 1
        "DIAMETER": np.sqrt(1/np.pi)*2,
        "EXPTIME" : 1.0,
        "GAIN"    : 1.0,
        "NOISETYP": "none",
        # if NOISETYP is gauss, use NOISESIG to set noise std
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

    def get_PSF(self, idx=0):
        ''' Return the idx PSF model in the data set
        '''
        return self.PSF[idx]

    def get_mask(self, idx=0):
        ''' Return the idx mask in the data set
        '''
        return self.mask[idx]

    def from_fits(self, file=None):
        ''' Read `GrismDataVector` from a fits file
        '''
        print(f'Reading dataset from {file}...')
        hdul = fits.open(file)
        self.header = hdul[0].header
        self.Nobs = self.header['OBSNUM']
        print(f'Find {self.Nobs} observations')
        # collect image, noise, PSF, and mask (PSF and mask are optional)
        self.data = []
        self.data_header = []
        self.noise = []
        self.mask = []
        self.PSF = []
        for i in range(self.Nobs):
            # compulsory: image, image header, and noise
            self.data.append(hdul[self.header["IMG%d"%(i+1)]].data)
            self.data_header.append(hdul[self.header["IMG%d"%(i+1)]].header)
            self.noise.append(hdul[self.header["NOISE%d"%(i+1)]].data)
            # optional: PSF, mask
            if "PSF%d"%(i+1) in self.header:
                _hdu_ = hdul[self.header["PSF%d"%(i+1)]]
                _psf_img_ = gs.Image(_hdu_.data, scale=_hdu_.header["PIXELSCL"], copy=True)
                self.PSF.append(gs.InterpolatedImage(_psf_img_, flux=1))
            else:
                self.PSF.append(None)
            if "MASK%d"%(i+1) in self.header:
                self.mask.append(hdul[self.header["MASK%d"%(i+1)]].data)
            else:
                self.mask.append(None)
            # self.data.append(hdul[1+2*i].data)
            # self.data_header.append(hdul[1+2*i].header)
            # self.noise.append(hdul[2+2*i].data)
            
        hdul.close()

    def to_fits(self, file, overwrite=False):
        ''' Write the data vector obj to a fits file
        '''
        hdu_primary = fits.PrimaryHDU(header=fits.Header(self.header))
        hdu_list = []
        for i in range(self.Nobs):
            # Compulsory: data image and noise
            hdu_primary.header["IMG%d"%(i+1)] = "IMAGE%d"%(i+1)
            hdu_list.append(fits.ImageHDU(self.data[i], name="IMAGE%d"%(i+1)),
                header=fits.Header(self.data_header[i]))
            hdu_primary.header["NOISE%d"%(i+1)] = "NOISE%d"%(i+1)
            hdu_list.append(fits.ImageHDU(self.noise[i], name="NOISE%d"%(i+1)), 
                header=fits.Header(self.data_header[i]))
            # Optional: PSF and mask
            if self.PSF[i] is not None:
                hdu_primary.header["PSF%d"%(i+1)] = "PSF%d"%(i+1)
                _pixelscl_ = self.data_header[i]["PIXSCALE"]/4.
                _NAXIS_ = self.data_header[i]["NAXIS1"]*4.
                _psf_img_ = self.PSF[i].drawImage(nx=_NAXIS_, ny=_NAXIS_, scale=_pixelscl_)
                _psf_hdu_ = fits.ImageHDU(_psf_img_.array, name="PSF%d"%(i+1),
                    header=fits.Header(self.data_header[i]))
                _psf_hdu_.header["PIXELSCL"] = _pixelscl_
                _psf_hdu_.header["OVERSAMP"] = 4.
                _psf_hdu_.header["DET_SAMP"] = 4.
                _psf_hdu_.header["FOV"] = _pixelscl_ * _NAXIS_
                hdu_list.append(_psf_hdu_)
            if self.mask[i] is not None:
                hdu_primary.header["MASK%d"%(i+1)] = "MASK%d"%(i+1)
                hdu_list.append(fits.ImageHDU(self.mask[i],name="MASK%d"%(i+1)), header=fits.Header(self.data_header[i]))
        hdul = fits.HDUList(hdu_list)
        hdul.writeto(file, overwrite=overwrite)

    def get_config(self, idx=0):
        ''' Get the configuration of the idx image

        The output of this function can be passed to `GrismModelCube:observe`
        method to generate realistic grism/photometry images.

        Note that so far this would only be a wrapper of data_header, but 
        in the future we can add more realistic attributes.
        '''
        _h = deepcopy(self.data_header[idx])
        for k in self._required_keys:
            if(not k in _h):
                _h[k] = self._default_data_header[k]
        _h["OBSINDEX"] = idx
        return dict(_h)

    def get_config_list(self):
        config_list = [self.get_config(i) for i in range(self.Nobs)]
        return config_list

    def stack(self):
        i = 0
        while i < len(self.data):
            if self.data_header[i]["OBSTYPE"] == 0:
                return self.data[i]
        return None

# class GrismSED(emission.SED):
#     '''
#     This class describe the obs-frame SED template of a source galaxy, includ-
#     ing emission lines and continuum components.
#     This is mostly a wrapper of the galsim.SED class object
    
#     Note that this class should only capture the intrinsic properties of
#     the galaxy SED, like
#         - redshift
#         - continuum and emission line flux density
#         - emission line width
#         - [dust?]
#     Other extrinsic properties, like
#         - sky background
#         - system bandpass
#         - spectral resolution and dispersion
#         - exposure time
#         - collecting area
#         - pixelization
#         - noise
#     will be implemented in mock observation module
#     '''
#     ### settings for the whole class
#     # default parameters
#     _default_pars = {
#         'template': '../../data/Simulation/GSB2.spec',
#         'wave_type': 'Ang',
#         'flux_type': 'flambda',
#         'z': 0.0,
#         'lblue': 50, # nm
#         'lred': 50000, # nm
#         #'spectral_range': (50, 50000), # nm
#         # obs-frame continuum normalization (nm, erg/s/cm2/nm)
#         'obs_cont_norm_wave': 400, # obs-frame continuum reference wavelength
#         'obs_cont_norm_flux': 0., # obs-frame continuum norm flux erg/s/cm2/nm
#         'em_Ha_flux': 1e-15, # obs-frame flux
#         'em_Ha_sigma': 0.5, # obs-frame Gaussian sigma

#         # spectral resolution at 1 micron, assuming dispersion per pixel
#         #'resolution': 3000,
#         # a dict of line names and obs-frame flux values (erg/s/cm2)
#         'lines': {'Ha': 1e-15},
#         # intrinsic linewidth in nm
#         'line_sigma_int': {'Ha': 0.5,},
#         #'line_hlr': (0.5, Unit('arcsec')),
#         'thin': -1,
#     }
#     # units conversion
#     _h = constants.h.to('erg s').value
#     _c = constants.c.to('nm/s').value
#     # build-in emission line species and info
#     _valid_lines = {k:v.to('nm').value for k,v in emission.LINE_LAMBDAS.items()}
    
#     def __init__(self, pars):
#         '''
#         Initialize SED class object with parameters dictionary
#         '''
#         self.pars = GrismSED._default_pars.copy()
#         self.updatePars(pars)

#         _con = self._addContinuum()
#         _emi = self._addEmissionLines()
#         self.spectrum = _con + _emi
#         if self.pars['thin'] > 0:
#             self.spectrum = self.spectrum.thin(rel_err=self.pars['thin'])
#         super(GrismSED, self).__init__(
#             self.pars['spectral_range'][0], self.pars['spectral_range'][1], 
#             3000, 'nm')

#     def __call__(self, wave):
#         if isinstance(wave, u.Quantity):
#             wave = wave.to(self.spectrum.wave_type).value
#         return self.spectrum(wave)
        
        
#     def updatePars(self, pars):
#         '''
#         Update parameters
#         '''
#         for key, val in pars.items():
#             self.pars[key] = val

#     def _addContinuum(self):
#         '''
#         Build and return continuum GalSim SED object
#         '''
#         template = self.pars['template']
#         if not os.path.isfile(template):
#             raise OSError(f'Can not find template file {template}!')
#         # build GalSim SED object out of template file
#         _template = np.genfromtxt(template)
#         _table = gs.LookupTable(x=_template[:,0], f=_template[:,1],)
#         SED = gs.SED(_table, 
#                          wave_type=self.pars['wave_type'], 
#                          flux_type=self.pars['flux_type'],
#                          redshift=self.pars['z'],
#                          _blue_limit=self.pars['spectral_range'][0],
#                          _red_limit=self.pars['spectral_range'][1])
#         # normalize the SED object at observer frame
#         # erg/s/cm2/nm -> photons/s/cm2/nm
#         # TODO: add more flexible normalization parametrization
#         norm = self.pars['obs_cont_norm'][1]*self.pars['obs_cont_norm'][0]/\
#             (GrismSED._h*GrismSED._c)
#         return SED.withFluxDensity(target_flux_density=norm, 
#                                    wavelength=self.pars['obs_cont_norm'][0])
    
#     def _addEmissionLines(self):
#         '''
#         Build and return Gaussian emission lines GalSim SED object
#         '''
#         # init LookupTable for rest-frame SED
#         lam_grid = np.arange(self.pars['spectral_range'][0]/(1+self.pars['z']),
#                              self.pars['spectral_range'][1]/(1+self.pars['z']), 
#                              0.1)
#         flux_grid = np.zeros(lam_grid.size)
#         # Set emission lines: (specie, observer frame flux)
#         all_lines = GrismSED._valid_lines.keys()
#         norm = -1
#         for line, flux in self.pars['lines'].items():
#             # sanity check
#             rest_lambda = np.atleast_1d(GrismSED._valid_lines[line])
#             flux = np.atleast_1d(flux)
#             line_sigma_int = np.atleast_1d(self.pars['line_sigma_int'][line])
#             if rest_lambda is None:
#                 raise ValueError(f'{line} is not a valid emission line! '+\
#                         f'For now, line must be one of {all_lines}')
#             else:
#                 assert rest_lambda.size == flux.size, f'{line} has'+\
#                 f' {rest_lambda.size} lines but {flux.size} flux are provided!'
#             # build rest-frame f_lambda SED [erg s-1 cm-2 nm-1]
#             # then, redshift the SED. The line width will increase automatically
#             for i,cen in enumerate(rest_lambda):
#                 _lw_sq = line_sigma_int[i]**2
#                 # erg/s/cm2 -> erg/s/cm2/nm
#                 _norm = flux[i]/np.sqrt(2*np.pi*_lw_sq)
#                 flux_grid += _norm * np.exp(-(lam_grid-cen)**2/(2*_lw_sq))
#                 # also, calculate normalization factor for obs-frame spectrum
#                 # convert flux units: erg/s/cm2/nm -> photons/s/cm2/nm
#                 # only one emission line needed
#                 if(norm<0):
#                     norm_lam = cen*(1+self.pars['z'])
#                     norm = flux[i]*norm_lam/(GrismSED._h*GrismSED._c)/\
#                                 np.sqrt(2*np.pi*_lw_sq*(1+self.pars['z'])**2)
            
#         _table = gs.LookupTable(x=lam_grid, f=flux_grid,)
#         SED = gs.SED(_table,
#                          wave_type='nm',
#                          flux_type='flambda',
#                          redshift=self.pars['z'],)
#         # normalize to observer-frame flux
#         SED = SED.withFluxDensity(target_flux_density=norm, 
#                                   wavelength=norm_lam)
#         return SED



def initialize_observations(Nobs, pars_list, datavector=None, overwrite=False):
    ''' Initialize the C++ simulated observation routines
    This will inform the C++ library about the info of each observation and 
    the data vector. Both header info and data images array are saved in C++

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
    if((Nobs != m.get_Nobs()) or overwrite):
        if(m.get_Nobs() > 0):
            print(f'cleaning up {m.get_Nobs()} observations')
            m.clear_observation()
        for i in range(Nobs):
            pars = pars_list[i]
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
