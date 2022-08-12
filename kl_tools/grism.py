import cube
import numpy as np
from astropy.io import fits
from astropy.table import Table, join, hstack
import os
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
from argparse import ArgumentParser

import utils
from emission import EmissionLine, LINE_LAMBDAS

import ipdb

parser = ArgumentParser()

parser.add_argument('--show', action='store_true', default=False,
                    help='Set to show test plots')
parser.add_argument('--test', action='store_true', default=False,
                    help='Set to run tests')

### Grism-specific pars here

class GrismPars(parameters.MetaPars):
    '''
    Class that defines structure for GrismModelCube meta parameters,
    e.g. dimension parameters for the underlying 3d theory model

    Add features for building 3d theory model cube
    '''

    _req_fields = ['', 
        'type', 'bandpass', 'Nx', 'Ny', 'pix_scale', ]
    _opt_fields = ['inst_name', 'diameter', 'exp_time', 'gain', 
        'noise', 'psf', 'grism_conf']

    def __init__(self, pars):
        '''
        pars: dict
            Dictionary of meta pars for the GrismCube
        '''
        Nobs = pars.get('number_of_observations', 0)
        assert Nobs > 0, 'number_of_observations must be positive!'
        flatten_pars = {
            'number_of_observations': Nobs
        }
        for field in cls._req_fields+cls._opt_fields:
            flatten_pars[field] = [
                pars['obs_%d'%(i+1)].get(field, None) for i in range(Nobs)
            ]
        super(GrismPars, self).__init__(flatten_pars)

        self._bandpasses = None
        bp = self.pars['bandpasses']
        try:
            utils.check_type(bp, 'bandpasses', list)
        except TypeError:
            try:
                utils.check_type(bp, 'bandpasses', dict)
            except:
                raise TypeError('GrismPars bandpass field must be a' +\
                                'list of galsim bandpasses/filenames or dict!')

        # ...

        return

    def build_bandpasses(self, remake=False):
        '''
        Build a bandpass list from pars if not provided directly
        '''
        bp = self.pars['bandpasses']

        if isinstance(bp, list):
            # we'll be lazy and just check the first entry
            if isinstance(bp[0], str):
                bandpasses = [gs.Bandpass(_bp, wave_type='nm') for _bp in bp]
            elif isinstance(bp[0], galsim.Bandpass):
                bandpasses = bp
            else:
                raise TypeError('bandpass list must be filled with ' +\
                                'galsim.Bandpass objects!')
        else:
            # already checked it is a list or dict
            bandpass_req = ['lambda_blue, lambda_red, dlambda']
            bandpass_opt = ['throughput', 'zp', 'unit']
            utils.check_fields(bp, bandpass_req, bandpass_opt)

            args = [
                pars['bandpasses'].pop('lambda_blue'),
                pars['bandpasses'].pop('lambda_red'),
                pars['bandpasses'].pop('dlambda')
                ]

            kwargs = pars['bandpasses']

            bandpasses = setup_simple_bandpasses(*args, **kwargs)

        self._bandpasses = bandpasses

        return bandpasses

class GrismModelCube(cube.DataCube):
    ''' Subclass DataCube for Grism modeling.

    Note: cube.DataCube interface

    methods:
    ========
        - __init__(data, pars=None, weights=None, masks=None,
                 pix_scale=None, bandpasses=None)
                the `data` is gonna to be returned by setup_model method is likelihood class
        - _check_shape_params()
        - slices()
        - _construct_slice_list()
        - from_fits(cubefile, dir=None, **kwargs)
        - get_sed(line_index=None, line_pars_update=None)
        - data()
        - slice(indx)
        - stack()
        - _set_maps(maps, map_type)
        - set_weights(weights)
        - set_masks(masks)
        - get_continuum()
        - copy()
        - get_inv_cov_list()
        - compute_aperture_spectrum(radius, offset=(0,0), plot_mask=False)
        - plot_aperture_spectrum(radius, offset=(0,0), size=None,
                               title=None, outfile=None, show=True,
                               close=True)
        - compute_pixel_spectrum(i, j)
        - _get_pixel_spectrum(i, j)
        - truncate(blue_cut, red_cut, lambda_unit=None, cut_type='edge',
                 trunc_type='in-place')
        - plot_slice(slice_index, plot_kwargs)
        - plot_pixel_spectrum(i, j, show=True, close=True, outfile=None)
        - write(outfile)

        - [NEW] observe(self, observe_params) # in the long run, observe_params gonna to be very close to roman grism pipeline convention. 
    attributes:
    ===========
        - 

    To fit into the cube.DataCube interface, what you need:

    '''

    def __init__(self, pars, datacube, imap, sed):
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
        _cubepars_dict['bandpasses'] = {
            'lambda_blue': pars.meta['model_dimension']['lambda_range'][0],
            'lambda_red': pars.meta['model_dimension']['lambda_range'][1],
            'dlambda': pars.meta['model_dimension']['lambda_res'],
            'throughput': 1.0,
            'unit': pars.meta['model_dimension']['lambda_unit'],
        }
        cubepars = cube.CubePars(_cubepars_dict)
        # init parent DataCube class, with 3D datacube and CubePars
        super(GrismModelCube, self).__init__(datacube, pars=cubepars)
        # also gs.Chromatic object for broadband image
        self.imap_img = imap_img
        self.sed = sed
        return

    def observe(self, config, add_noise = False, force_noise_free=True):
        ''' Simulate observed image given `observe_parmas` which specifies all
        the information you need.

        `config` is returned by `GrismDataVector` object.
        In the long run, `observe_params` gonna to be very close to Anahita's 
        Roman grism simulation convention, but in the short-term, it will just
        be a dictionary.
        '''
        _type = config.get('type', 'none').lower()
        if _type == 'grism':
            return self._get_grism_data(config, add_noise=add_noise)
        elif _type == 'photometry':
            return self._get_photometry_data(config, add_noise=add_noise)
        else:
            print(f'{_type} not supported by GrismModelCube!')
            exit(-1)

    def _get_photometry_data(self, config, 
        add_noise = False, force_noise_free = True):
        print(f'WARNING: have not test normalization yet')
        # self._data is in units of [photons / (s cm2)]
        # No, get chomatic directly, from GrismModelCube
        image = gs.Image(self.imap_img, copy=True)
        gal = gs.InterpolatedImage(image, scale=self.pars['pix_scale'])
        gal_chromatic = gal * self.sed

        # convolve with PSF
        if config.get('psf_type', 'none').lower()!='none':
            psf = self._build_PSF_model(
                config,
                lam=config['bandpass'].calculateEffectiveWavelength()
                )
            _gal_chromatic = gs.Convolve([gal_chromatic, psf])
        photometry_img = gal_chromatic.drawImage(
                                nx=config['Nx'], ny=config['Ny'], 
                                scale=config['pix_scale'], method='auto',
                                area=config['area'], 
                                exptime=config['exp_time'],
                                gain=config['gain'],
                                bandpass=config['bandpass'])
        # apply noise
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

    def _get_grism_data(self, config, 
            add_noise=False, force_noise_free=True):
        # prepare datacube, lambdas, bandpass, output and config dict
        bp = gs.Bandpass(config.get('bandpass'), 
                        wave_type=config.get('wave_type', 'nm'))
        bp_array = bp[self.lambdas]
        config['model_Nx'] = self.Nx
        config['model_Ny'] = self.Ny
        config['model_Nlam'] = self.Nspec
        config['model_scale'] = self.pix_scale
        grism_img_array = np.zeros([config['Ny'], config['Nx']], 
            dtype=np.float64, order='C')
        # call c++ routine to disperse
        m.cpp_stack(self._data, self.lambdas, bp_array, grism_img_array, config)
        # wrap with noise & psf
        grism_img = gs.Image(
            grism_img_array,
            dtype = np.float64, scale=config['pix_scale'], 
            )
        #print("----- wrap image | %s seconds -----" % (time() - start_time))
        # convolve with achromatic psf, if required
        psf = self._build_PSF_model(config)
        if psf is not None:
            _gal = gs.InterpolatedImage(grism_img, scale=config['pix_scale'])
            grism_gal = gs.Convolve([_gal, psf])
            grism_img = grism_gal.drawImage(nx=config['Nx'], ny=config['Ny'], 
                                            scale=config['pix_scale'])
        #print("----- convolv PSF | %s seconds -----" % (time() - start_time))
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

    def set_data(self, data):
        assert data.shape == self._data.shape, f'data shape {data.shape} is'+\
            f'inconsistent with {self._data.shape}!'
        self._data = data

    def stack(self, axis=0):
        return np.sum(self._data, axis=axis)

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
        if _type != 'None':
            if _type == 'airy':
                lam = kwargs.get('lam', 1000) # nm
                scale = kwargs.get('scale_unit', gs.arcsec)
                return gs.Airy(lam=lam, diam=self.diameter/100,
                                scale_unit=scale)
            elif _type == 'airy_mean':
                return gs.Airy(config['psf_fwhm']/1.028993969962188, 
                    scale_unit=scale)
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

    '''

    def __init__(self, 
        file=None,
        header = None, data = None, data_header = None, noise = None,
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
            ''' dict of meta params for the list of data
                as an example, {
                    'NEXTENSIONS': 3, etc
                }
            '''
            # list of np.ndarray
            self.data = data
            # list of dict for each data extension
            self.data_header = data_header
            '''
                as an example, {
                    'NAXIS_X_val': 123,
                    'NAXIS_Y_val': 234, 
                    'PIXSCALE': 0.1,
                    ''
                }
            '''
            # list of np.ndarray
            self.noise = noise


    def get_inv_cov_list(self):
        inv_cov_list = [self.get_inv_cov(i) for i in range(len(self.noise))]
        return inv_cov_list

    def get_inv_cov(self, idx):
        return 1/self.noise[idx]**2

    def get_data(self, idx=0):
        ''' Return the idx image in the data set
        '''
        return self.data[idx]

    def from_fits(self, file=None):
        ''' Read `GrismDataVector` from a fits file
        '''
        hdu = fitsio

    def get_config(self, idx=0):
        ''' Get the configuration of the idx image

        The output of this function can be passed to `GrismModelCube:observe`
        method to generate realistic grism/photometry images.

        Note that so far this would only be a wrapper of data_header, but 
        in the future we can add more realistic attributes.
        '''
        _h = self.data_header[idx]
        config = {
            'inst_name': _h['inst_name'],
            'type': _h['type'],
            'bandpass': _h['bandpass'],
            'Nx': _h['Nx'],
            'Ny': _h['Ny'],
            'pix_scale': _h['pix_scale'],
            'diameter': _h['diameter'],
            'exp_time': _h['exp_time'],
            'gain': _h['gain'],
            'noise_type': _h['noise_type'],
            'sky_level': _h['sky_level'],
            'read_noise': _h['read_noise'],
            'apply_to_data': _h['apply_to_data'],
            'psf_type': _h['psf_type'],
            'psf_fwhm': _h['psf_fwhm'],
            'R_spec': _h['R_spec'],
            'disp_ang': _h['disp_ang'],
            'offset': _h['offset'],
        }
        return config

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