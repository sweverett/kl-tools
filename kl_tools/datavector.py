from abc import abstractmethod
from astropy.io import fits
from astropy.table import Table, join, hstack
from copy import deepcopy

class DataVector(object):
    '''
    Light wrapper around things like cube.DataCube to allow for
    uniform interface for things like the intensity map rendering
    '''

    @abstractmethod
    def stack(self):
        '''
        Each datavector must have a method that defines how to stack it
        into a single (Nx,Ny) image for basis function fitting
        '''
        pass

class FiberDataVector(DataVector):
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
            - OBSTYPE : observation type; 0-photometry; 1-grism
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
        # if NOISETYP is gauss, use NOISESIG to set noise std
        "SKYLEVEL": 0.0,
        "RDNOISE" : 0.0,
        "ADDNOISE": False,
        "PSFTYPE" : "none",
        "PSFFWHM" : 0.0,
        "RSPEC"   : 0.0,
        "DISPANG" : 0.0,
        "OFFSET"  : 0.0,
        "FIBERDX": 0,
        "FIBERDY": 0,
        "FIBERRAD": 1.5,
        "FIBRBLUR": 3.4
    }
    _required_keys = ["OBSTYPE", "INSTNAME", "BANDPASS", "NAXIS", "NAXIS1",
        "NAXIS2", "PIXSCALE", "DIAMETER", "EXPTIME", "GAIN", "NOISETYP", 
        "SKYLEVEL", "RDNOISE", "ADDNOISE", "PSFTYPE", "PSFFWHM", "FIBERDX", "FIBERDY", "FIBERRAD", "FIBRBLUR"]

    def __init__(self, 
        file=None,
        header = None, data_header = None, data = None, noise = None,
        ):
        ''' Initialize the `FiberDataVector` class object
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

    def to_fits(self, file, overwrite=False):
        ''' Write the data vector obj to a fits file
        '''
        hdu_list = [fits.PrimaryHDU(header=fits.Header(self.header))]
        for i in range(self.Nobs):
            hdu_list.append(fits.ImageHDU(self.data[i], 
            header=fits.Header(self.data_header[i])))
            hdu_list.append(fits.ImageHDU(self.noise[i], 
            header=fits.Header(self.data_header[i])))
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