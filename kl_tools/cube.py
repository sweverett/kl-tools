import numpy as np
from scipy.sparse import identity, dia_matrix
#import fitsio
from astropy.io import fits
import astropy.units as u
import galsim
import os
from time import time
import pickle
from copy import deepcopy
from astropy.table import Table
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from photutils.geometry import circular_overlap_grid as cog

import kl_tools.utils as utils
import kl_tools.parameters as parameters
from kl_tools.datavector import DataVector

import ipdb

parser = ArgumentParser()

parser.add_argument('--show', action='store_true', default=False,
                    help='Set to show test plots')
parser.add_argument('--test', action='store_true', default=False,
                    help='Set to run tests')


class DataCube(DataVector):
    '''
    Base class for an abstract data cube.
    Contains astronomical images of a source
    at various wavelength slices

    NOTE: All subclasses of DataVector must implement a stack() method
    '''

    def __init__(self, data, pars=None, weights=None, masks=None,
                 pix_scale=None, bandpasses=None):
        '''
        Initialize either a filled DataCube from an existing numpy
        array or an empty one from a given shape

        data: np.array
            A numpy array containing all image slice data.
            For now, assumed to be the shape format given below.
        pars: dict, CubePars
            A dictionary or CubePars that holds any additional metadata
            NOTE: should *only* be None if pix_scale & bandapsses are
            passed explicitly
        weights: int, list, np.ndarray
            The weight maps for each slice. See set_weights()
            for more info on acceptable types.
        masks: np.ndarray
            The mask maps for each slice. See set_masks()
            for more info on acceptable types.

        The following are not standard for current setups, but
        exist for compatibility w/ previous verions:

        bandpasses: list
            A list of galsim.Bandpass objects containing
            throughput function, lambda window, etc.
        pix_scale: float
            the pixel scale of the datacube slices
        '''

        # This is present to handle older versions of DataCube
        if pars is None:
            if (pix_scale is None) or (bandpasses is None):
                raise Exception('Both pix_scale and bandpasses must be ' +\
                                'passed if pars is not!')
            pars_dict = {
                'pix_scale': pix_scale,
                'bandpasses': bandpasses
            }
            pars = parameters.CubePars(pars_dict)
        else:
            if (pix_scale is not None) or (bandpasses is not None):
                raise Exception('Cannot pass pix_scale or bandpasses if ' +\
                                'pars is set!')
            try:
                utils.check_type(pars, 'pars', parameters.CubePars)
            except TypeError:
                try:
                    utils.check_type(pars, 'pars', dict)
                except:
                    raise TypeError('pars must be either CubePars ' +\
                                    'or a dict!')
                pars = parameters.CubePars(pars)

        self.pars = pars
        self.pix_scale = pars['pix_scale']

        if len(data.shape) != 3:
            # Handle the case of 1 slice
            assert len(data.shape) == 2
            data = data.reshape(1, data.shape[0], data.shape[1])

        self.shape = data.shape

        self.pars.set_shape(self.shape)

        self.Nspec = self.shape[0]
        self.Nx = self.shape[1]
        self.Ny = self.shape[2]

        self._data = data

        if self.shape[0] != len(self.bandpasses):
            raise ValueError('The length of the bandpasses must ' + \
                             'equal the length of the third data dimension!')

        if self.Nspec == 0:
            print('WARNING: there are no slices in passed datacube. ' +\
                  'Are you sure that is right?')

        # If weights/masks not passed, set non-informative defaults
        if weights is not None:
            self.set_weights(weights)
        else:
            self.weights = np.ones(self.shape)
        if masks is not None:
            self.set_masks(masks)
        else:
            self.masks = np.zeros(self.shape)

        self._continuum_template = None

        # only create slice list when requested, to save time in MCMC sampling
        self.slice_list = None

        return

    def _check_shape_params(self):
        Nzip = zip(['Nspec', 'Nx', 'Ny'], [self.Nspec, self.Nx, self.Ny])
        for name, val in Nzip:
            if val < 1:
                raise ValueError(f'{name} must be greater than 0!')

        if len(self.shape) != 3:
            raise ValueError('DataCube.shape must be len 3!')

        return

    @property
    def bandpasses(self):
        return self.pars.bandpasses

    @property
    def lambdas(self):
        return self.pars.lambdas
    @property
    def lambda_unit(self):
        return self.pars._lambda_unit

    @property
    def slices(self):
        if self.slice_list is None:
            self._construct_slice_list()

        return self.slice_list

    def _construct_slice_list(self):
        self.slice_list = SliceList()

        for i in range(self.Nspec):
            bp = self.bandpasses[i]
            weight = self.weights[i]
            mask = self.masks[i]
            self.slices.append(
                Slice(
                    self._data[i,:,:], bp, weight=weight, mask=mask
                    )
                )

        return

    @classmethod
    def from_fits(cls, cubefile, dir=None, **kwargs):
        '''
        Build a DataCube, but instantiated instead from a fitscube file
        and associated file containing bandpass list

        Assumes the datacube has a shape of (Nspec,Nx,Ny)

        cubefile: str
            Location of fits cube
        '''

        if dir is not None:
            cubefile = os.path.join(dir, cubefile)

        utils.check_file(cubefile)

        cubefile = cubefile

        #data = fitsio.read(cubefile)
        data = fits.open(cubefile)

        datacube = DataCube(data, **kwargs)

        datacube.pars['files'] = {
            'cubefile': cubefile
        }

        return datacube

    def get_sed(self, line_index=None, line_pars_update=None):
        '''
        Get emission line SED, or modify if needed

        line_index: int
            The index of the desired emission line
        line_pars_update: dict
            Any differences to the internally stored line_pars in the datacube
            emission line. Useful when sampling over SED parameters
        '''
        try:
            if line_index is None:
                if len(self.pars['emission_lines']) == 1:
                    line_index = 0
                else:
                    raise ValueError('Must pass a line_index if more than ' +\
                                     'one line are stored!')

            line = self.pars['emission_lines'][line_index]
            if line_pars_update is None:
                sed = line.sed
            else:
                line_pars = deepcopy(line.line_pars)
                line_pars.update(line_pars_update)
                sed_pars = line.sed_pars
                sed = line._build_sed(line_pars, sed_pars)

            return sed

        except KeyError:
            raise AttributeError('Emission lines never set for datacube!')

    def set_psf(self, psf):
        '''
        psf: galsim.GSObject
            A PSF model for the datacube

        NOTE: This assumes the psf is achromatic for now!
        '''

        if psf is not None:
            if not isinstance(psf, galsim.GSObject):
                raise TypeError('psf must be a galsim.GSObject!')

        self.pars['psf'] = psf

        return

    def get_psf(self, wavelength=None, wav_unit=None):
        '''
        Return the PSF of the datacube at the desired wavelength.
        In many cases this may be constant, in which case a wavelength
        does not have to be passed

        wavelength: float
            The wavelength of the PSF. Optional if PSF is achromatic
        wav_unit: astropy.units.Unit
            The unit of the passed wavelength. If not passed, will default
            to using the unit of the stored PSF in CubePars
        '''

        # TODO: Implement the rest!

        if 'psf' in self.pars:
            return self.pars['psf']
        else:
            # raise AttributeError('There is no PSF stored in datacube pars!')
            return None

    @property
    def data(self):
        return self._data

    @property
    def z(self, line_indx=0):
        '''
        TODO: This is a bit hacky, we may want a better way to store the object
        redshift & check that it is the same across all indices
        '''

        if 'emission_lines' in self.pars:
            Nlines = len(self.pars['emission_lines'])
            if line_indx >= Nlines:
                raise ValueError(f'{line_indx} is too large an index for ' +\
                                 'the emission line list of len {Nlines}')

            return self.pars['emission_lines'][line_indx].line_pars['z']

        else:
            return None

    def slice(self, indx):
        return self.slices[indx]

    def stack(self):
        return np.sum(self._data, axis=0)

    def _set_maps(self, maps, map_type):
        '''
        maps: float, list, np.ndarray
            Weight or mask maps to set
        map_type: str
            Name of map type

        Simple method for assigning weight or mask maps to datacube,
        without any knowledge of input datacube structure. Can assign
        uniform maps for all slices w/ a float, or pass a list of
        maps. Pre-assigned default is all 1's.
        '''

        valid_maps = ['weights', 'masks']
        if map_type not in valid_maps:
            raise ValueError(f'map_type must be in {valid_maps}!')

        # base array that will be filled
        stored_maps = np.ones(self.shape)

        if isinstance(maps, (int, float)):
            # set all maps to constant value
            stored_maps *= maps

        elif isinstance(maps, (list, np.ndarray)):
            if len(maps) != self.Nspec:
                if isinstance(maps, np.ndarray):
                    if (len(maps.shape) == 2) and (maps.shape == self.shape[1:]):
                        # set all maps to the passed map
                        # useful for say a global mask
                        for i in range(self.Nspec):
                            stored_maps[i] = maps
                    else:
                        raise ValueError('Passed {map_type} map has shape ' +\
                                         f'{maps.shape} but datacube shape ' +\
                                         f'is {self.shape}!')

                else:
                    raise ValueError(f'Passed maps list has len={len(maps)}' +\
                                     f' but Nspec={self.Nspec}!')
            else:
                for i, m in enumerate(maps):
                    if isinstance(m, (int, float)):
                        # set uniform slice map
                        stored_maps[i] *= m
                    else:
                        # assume slice is np.ndarray
                        if m.shape != self.shape[1:]:
                            raise ValueError(f'map shape of {m.shape} does not ' +\
                                             f'match slice shape {self.shape[1:]}!')
                        stored_maps[i] = m
        else:
            raise TypeError(f'{map_type} map must be a float, list, ' +\
                            'or np.ndarray!')

        setattr(self, map_type, stored_maps)

        return

    def set_weights(self, weights):
        '''
        weights: float, list, np.ndarray

        see _set_maps()
        '''

        self._set_maps(weights, 'weights')

        return

    def set_masks(self, masks):
        '''
        mask: float, list, np.ndarray

        see _set_maps()
        '''

        self._set_maps(masks, 'masks')

        return

    def get_continuum(self):
        if self._continuum_template is None:
            raise AttributeError('Need to have set calculate a template ' +\
                                 'for the continuum first')

        return self._continuum_template

    def set_continuum(self, continuum):
        '''
        Very basic version for the base class - should probably be
        overloaded for each data-specific subclass

        continuum: np.ndarray
            A 2D numpy array representing the spectral continuum template
            for a given emission line
        '''

        if continuum.shape != self.shape[1:]:
            raise Exception('Continuum template must have the same ' +\
                            'dimensions as the datacube slice!')

        self._continuum_template = continuum

        return

    def copy(self):
        return deepcopy(self)

    def get_inv_cov_list(self):
        '''
        Build inverse covariance matrices for slice images

        returns: List of (Nx*Ny, Nx*Ny) scipy sparse matrices
        '''

        Nspec = self.Nspec
        Npix = self.Nx * self.Ny

        weights = self.weights

        inv_cov_list = []
        for i in range(Nspec):
            inv_var = (weights[i]**2).reshape(Npix)
            inv_cov = dia_matrix((inv_var, 0), shape=(Npix,Npix))
            inv_cov_list.append(inv_cov)

        return inv_cov_list

    def compute_aperture_spectrum(self, radius, offset=(0,0), plot_mask=False):
        '''
        radius: aperture radius in pixels
        offset: aperture center offset tuple in pixels about slice center
        '''

        mask = np.zeros((self.Nx, self.Ny), dtype=np.dtype(bool))

        im_center = (self.Nx/2, self.Ny/2)
        center = np.array(im_center) + np.array(offset)

        aper_spec = np.zeros(self.Nspec)

        for x in range(self.Nx):
            for y in range(self.Ny):
                dist = np.sqrt((x-center[0])**2+(y-center[1])**2)
                if dist < radius:
                    aper_spec += self._get_pixel_spectrum(x,y)
                    mask[x,y] = True

        if plot_mask is True:
            plt.imshow(mask, origin='lower')

            cx, cy = center[0], center[1]
            circle = plt.Circle((cx,cy), radius, color='r', fill=False,
                                lw=3, label='Aperture')

            ax = plt.gca()
            ax.add_patch(circle)
            plt.legend()
            plt.show()

        return aper_spec

    def plot_aperture_spectrum(self, radius, offset=(0,0), size=None,
                               title=None, outfile=None, show=True,
                               close=True):

        aper_spec = self.compute_aperture_spectrum(radius, offset=offset)
        lambda_means = np.mean(self.lambdas, axis=1)

        plt.plot(lambda_means, aper_spec)
        plt.xlabel(f'Lambda ({self.lambda_unit})')
        plt.ylabel(f'Flux (ADUs)')

        if title is not None:
            plt.title(title)
        else:
            plt.title(f'Aperture spectrum for radius={radius} pixels; ' +\
                      f'offset={offset}')

        if size is not None:
            plt.gcf().set_size_inches(size)

        if outfile is not None:
            plt.savefig(outfile, bbox_inches='tight')

        if show is True:
            plt.show()
        elif close is True:
            plt.close()

        return

    def compute_pixel_spectrum(self, i, j):
        '''
        Compute the spectrum of the pixel (i,j) across
        all slices

        # TODO: Work out units!
        '''

        pix_spec = self._get_pixel_spectrum(i,j)

        # presumably some unit conversion...

        # ...

        return pix_spec

    def _get_pixel_spectrum(self, i, j):
        '''
        Return the raw spectrum of the pixel (i,j) across
        all slices
        '''

        return self._data[:,i,j]

    def truncate(self, blue_cut, red_cut, lambda_unit=None, cut_type='edge',
                 trunc_type='in-place'):
        '''
        Modify existing datacube to only hold slices between blue_cut
        and red_cut using either the lambda on a slice center or edge

        blue_cut: float
            Blue-end wavelength for datacube truncation
        red_cut: float
            Red-end wavelength for datacube truncation
        lambda_unit: astropy.Unit
            The unit of the passed wavelength limits
        cut_type: str
            The type of truncation applied ['edge' or 'center']
        trunc_type: str
            Select whether to apply the truncation w/ the DataCube constructor
            or to just return the (args, kwargs) needed to produce the
            truncation (args, kwargs). This is particularly useful for
            subclasses of DataCube
        '''

        for l in [blue_cut, red_cut]:
            if (not isinstance(l, float)) and (not isinstance(l, int)):
                raise ValueError('Truncation wavelengths must be ints or ' +\
                                 'floats!')

        if (blue_cut >= red_cut):
            raise ValueError('blue_cut must be less than red_cut!')

        if cut_type not in ['edge', 'center']:
            raise ValueError('cut_type can only be at the edge or center!')

        if trunc_type not in ['in-place', 'return-args']:
            raise ValueError('trunc_type can only be in_place or return-args!')

        # make sure we get a correct comparison between possible
        # unit differences
        lu = self.lambda_unit

        if lambda_unit is None:
            lambda_unit = self.lambda_unit
        blue_cut *= lambda_unit
        red_cut *= lambda_unit

        if cut_type == 'center':
            # truncate on slice center lambda value
            lambda_means = np.mean(self.lambdas, axis=1)

            cut = (lu*lambda_means >= blue_cut) & (lu*lambda_means <= red_cut)

        else:
            # truncate on slice lambda edge values
            lambda_blues = np.array([self.lambdas[i][0] for i in range(self.Nspec)])
            lambda_reds  = np.array([self.lambdas[i][1] for i in range(self.Nspec)])

            cut = (lu*lambda_blues >= blue_cut) & (lu*lambda_reds  <= red_cut)

        # NOTE: could either update attributes or return new DataCube
        # For base DataCube's, simplest to use constructor to build
        # a fresh one. But won't work for more complex subclasses
        # (like those built from fits files)
        trunc_data = self._data[cut,:,:]
        trunc_weights = self.weights[cut,:,:]
        trunc_masks = self.masks[cut,:,:]

        trunc_pars = self.pars.copy() # is a deep copy

        # Have to do it this way as lists cannot be indexed by np arrays
        trunc_pars['bandpasses'] = [self.bandpasses[i]
                                   for i in range(self.Nspec)
                                   if cut[i] == True]
        self.pars._bandpasses = None # Force CubePars to remake bandpass list

        # Reset any attributes set during initialization
        trunc_pars.reset()

        if trunc_type == 'in-place':
            self.__init__(
                trunc_data,
                pars=trunc_pars,
                weights=trunc_weights,
                masks=trunc_masks,
            )
        elif trunc_type == 'return-args':
            args = [trunc_data]
            kwargs = {
                'pars': trunc_pars,
                'weights': trunc_weights,
                'masks': trunc_masks
            }
            return (args, kwargs)

        return

    def set_data(self, data, weights=None, masks=None):
        '''
        This method overwrites the existing datacube slice data, while
        keeping all existing metadata

        data: np.ndarray
            The 3-dimensional numpy array to set as the new slice data
        weights: float, list, np.ndarray
            Pass if you want to overwrite the weight maps as well
        masks: float, list, np.ndarray
            Pass if you want to overwrite the mask maps as well

        see _set_maps() for details for weight & masks
        '''

        if data.shape != self.shape:
            raise ValueError('Passed data shape must match existing shape!')


        self._data = data

        if weights is not None:
            self.set_weights(weights)
        if masks is not None:
            self.set_masks(masks)

        return

    def plot_slice(self, slice_index, plot_kwargs):
        self.slices[slice_index].plot(**plot_kwargs)

        return

    def plot_pixel_spectrum(self, i, j, show=True, close=True, outfile=None):
        '''
        Plot the spectrum for pixel (i,j) across
        all slices

        # TODO: Work out units!
        '''

        pix_spec = self.compute_pixel_spectrum(i,j)

        lambda_means = np.mean(self.lambdas, axis=1)
        unit = self.lambda_unit

        plt.plot(lambda_means, pix_spec)
        plt.xlabel(f'Lambda ({unit})')
        plt.ylabel('Flux (ADU)')
        plt.title(f'Spectrum for pixel ({i}, {j})')

        if outfile is not None:
            plt.savefig(outfile, bbox_inches='tight')

        if show is True:
            plt.show()
        elif close is True:
            plt.close()

        return

    def write(self, outfile):
        '''
        TODO: Should update this now that there are
        weight & mask maps stored
        '''
        d = os.path.dirname(outfile)

        utils.make_dir(d)

        im_list = []
        for s in self.slices:
            im_list.append(s._data)

        galsim.fits.writeCube(im_list, outfile)

        return

class SliceList(list):
    '''
    A list of Slice objects
    '''
    pass

class Slice(object):
    '''
    Base class of an abstract DataCube slice,
    corresponding to a source observation in a given
    bandpass
    '''
    def __init__(self, data, bandpass, weight=None, mask=None):
        self._data = data
        self.bandpass = bandpass
        self.weight = weight
        self.mask = mask

        self.red_limit = bandpass.red_limit
        self.blue_limit = bandpass.blue_limit
        self.dlamda = self.red_limit - self.blue_limit
        self.lambda_unit = u.Unit(bandpass.wave_type)

        return

    @property
    def data(self):
        '''
        Seems silly now, but this might be useful later
        '''

        return self._data

    def plot(self, show=True, close=True, outfile=None, size=9, title=None,
             imshow_kwargs=None):

        if imshow_kwargs is None:
            im = plt.imshow(self._data)
        else:
            im = plt.imshow(self._data, **imshow_kwargs)

        plt.colorbar(im)

        if title is not None:
            plt.title(title)
        else:
            li, le = self.blue_limit, self.red_limit
            unit = self.lambda_unit
            plt.title(f'DataCube Slice; {li} {unit} < ' +\
                      f'lambda < {le} {unit}')

        plt.gcf().set_size_inches(size, size)

        if outfile is not None:
            plt.savefig(outfile, bbox_inches='tight')

        if show is True:
            plt.show()
        elif close is True:
            plt.close()

        return

class FiberModelCube(DataVector):
    ''' The most updated `FiberModelCube` implementation
    This class wraps the theoretical cube after it is generated to produce
    simulated images.
    '''
    def __init__(self, meta_pars, obs_conf=None):
        ''' Store the `FiberPars` object
        '''
        self.Fpars = parameters.FiberPars(meta_pars, obs_conf=obs_conf)
        # calculate the atmospheric PSF convolved fiber mask, if PSF model is
        # not sampled.
        if self.Fpars.is_dispersed:
            self.ATMPSF_conv_fiber_mask = self.get_PSF_convolved_fiber_mask()
            self.resolution_mat = self.get_resolution_matrix()
        else:
            self.ATMPSF_conv_fiber_mask = None
            self.resolution_mat = None
        return

    @property
    def bp_array(self):
        return self.Fpars.bp_array
    @property
    def conf(self):
        return self.Fpars.conf
    @property
    def lambdas(self): # (Nlam, 2)
        return self.Fpars.lambdas
    @property
    def wave(self):
        return self.Fpars.wave
    @property
    def lambda_eff(self):
        return self.Fpars.lambda_eff

    def get_fiber_mask(self):
        mNx, mNy = self.Fpars['shape'][2], self.Fpars['shape'][1]
        mscale = self.Fpars['pix_scale']
        if self.Fpars.is_dispersed:
            fiber_cen = [self.conf['FIBERDX'], self.conf['FIBERDY']] # dx, dy in arcsec
            fiber_rad = self.conf['FIBERRAD'] # radius in arcsec
            xmin, xmax = -mNx/2*mscale, mNx/2*mscale
            ymin, ymax = -mNy/2*mscale, mNy/2*mscale
            mask = cog(xmin-fiber_cen[0], xmax-fiber_cen[0],
                ymin-fiber_cen[1], ymax-fiber_cen[1],
                mNx, mNy, fiber_rad, 1, 2)
        else:
            mask = np.ones([mNy, mNx])
        return mask

    def get_fiber_1D_psf_kernel(self):
        if self.Fpars.is_dispersed:
            diameter_in_pixel = self.conf['FIBRBLUR']
            sigma = diameter_in_pixel / 4.
            x_in_pixel = np.arange(-5, 6)
            # assume Gaussian for now
            kernel = np.exp(-0.5*(x_in_pixel/sigma)**2)/((2*np.pi)**0.5*sigma)
        else:
            kernel = None
        return kernel
    def get_resolution_matrix(self):
        if self.Fpars.is_dispersed:
            diameter_in_pixel = self.conf['FIBRBLUR']
            sigma = diameter_in_pixel / 4.
            x_in_pixel = np.arange(-5, 6)
            # assume Gaussian for now
            kernel = np.exp(-0.5*(x_in_pixel/sigma)**2)/((2*np.pi)**0.5*sigma)
            # get the resolution matrix (sparse matrix)
            band = np.array([kernel]).repeat(self.Fpars['shape'][0], axis=0).T
            offset=np.arange(kernel.shape[0]//2, -(kernel.shape[0]//2)-1, -1)
            #print(band.shape, offset)
            #plt.imshow(band)
            Rmat = dia_matrix((band, offset),
                shape=(self.Fpars['shape'][0], self.Fpars['shape'][0]))
        else:
            Rmat = None
        return Rmat

    def get_PSF_convolved_fiber_mask(self):
        ''' get atm-PSF convolved fiber mask
        '''
        mNx, mNy = self.Fpars['shape'][2], self.Fpars['shape'][1]
        mscale = self.Fpars['pix_scale']
        psf = self._build_PSF_model(self.conf, lam_mean=self.lambda_eff)
        mask = galsim.InterpolatedImage(
            galsim.Image(array=self.get_fiber_mask()),
            scale=mscale)
        # convolve fiber mask with atmospheric PSF
        maskC = mask if psf is None else galsim.Convolve([mask, psf])
        ary = maskC.drawImage(nx=mNx,ny=mNy,scale=mscale).array

        return ary

    def observe(self, theory_cube, gal, sed, force_noise_free=True):
        ''' Simulate observed image
        '''
        # if photometry image
        if not self.Fpars.is_dispersed:
            if self.Fpars['run_options']['run_mode'] == 'ETC':
                # theory_cube, gal, and sed, in this case, are physical units
                gal_chro = gal * sed.obs_frame_sed
                psf = self._build_PSF_model(self.conf, lam_mean=self.lambda_eff)
                gal = gal_chro if psf is None else galsim.Convolve([gal_chro, psf])
                img = gal.drawImage(
                    bandpass = self.Fpars.throughput,
                    nx=self.conf['NAXIS1'], ny=self.conf['NAXIS2'],
                    scale=self.conf['PIXSCALE'], method='auto',
                    area=np.pi*(self.conf['DIAMETER']/2.)**2,
                    exptime=self.conf['EXPTIME'],gain=self.conf['GAIN'])
            elif self.Fpars['run_options']['run_mode'] == 'SNR':
                # theory_cube, gal, and sed, in this case, are arbitrary units
                # However, theory_cube and gal are already scaled by flux factor
                assert self.conf['NOISETYP'].lower() == 'gauss'
                img = gal.drawImage(
                    nx=self.conf['NAXIS1'], ny=self.conf['NAXIS2'],
                    scale=self.conf['PIXSCALE'], method='auto',
                    )
            if force_noise_free:
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
        # if fiber 1D spectrum
        else:
            # The `lambdas` should be in fiber ccd native resolution
            # Rigorous treatment would be:
            #   - convolve 3D model cube with atmosphere PSF per slice
            #   - apply fiber mask
            #   - sum slice-wise to get 1D spectrum
            # However, if the atmosphere PSF is symmetric about the origin,
            # e.g. PSF(x,y) = PSF(-x, -y), then it is equivalent to
            #  - convolve fiber mask with atmosphere PSF
            #  - apply PSF-convolved fiber mask
            #  - sum slice-wise to get 1D spectrum
            # But you want to ensure your grid is large enough to accommodate
            # convolution

            # theory_cube (should be) in units of [photons / (s cm2)]
            #print(f'theory cube shape: {theory_cube.shape}')
            #print(f'mask shape: {ary.shape}')
            spec_1D = (self.ATMPSF_conv_fiber_mask[np.newaxis,:,:]*theory_cube).sum(axis=(1,2))
            if self.Fpars['run_options']['run_mode'] == 'ETC':
                spec_1D = spec_1D*self.bp_array
                factor = np.pi*(self.conf['DIAMETER']/2.)**2*self.conf['EXPTIME']/self.conf['GAIN']
                spec_1D = spec_1D * factor
            # otherwise, the theory_cube is scaled by a factor for emission line

            # fiber PSF can result in degrade in spectra resolution
            if self.resolution_mat is not None:
                spec_1D = self.resolution_mat * spec_1D
            if force_noise_free:
                return spec_1D, None
            else:
                # place holder
                if self.Fpars['run_options']['run_mode'] == 'ETC':
                    skysb = galsim.LookupTable.from_file(self.conf["SKYMODEL"], f_log=True) # Ang v.s. 1e-17 erg s-1 cm-2 A-1 arcsec-2
                    fiber_area = np.pi*(self.conf["FIBERRAD"])**2
                    _wave = self.wave*10 # Angstrom
                    _dwave = (_wave[1]-_wave[0]) # Angstrom
                    _hnu = 1986445857.148928/_wave # 1e-17 erg
                    skyct = skysb(_wave)*fiber_area*_dwave/_hnu # s-1 cm-2
                    skyct *= self.bp_array*np.pi*(self.conf['DIAMETER']/2.)**2*self.conf['EXPTIME']/self.conf['GAIN']
                    # noise std, not including dark current
                    # eff read noise = sqrt(Npix along y) * read noise
                    noise_std = (skyct+spec_1D+self.conf['RDNOISE']**2)**0.5
                    noise = np.random.randn(spec_1D.shape[0])*noise_std
                else:
                    noise = np.random.randn(spec_1D.shape[0])*self.conf['NOISESIG']
                #print("----- %s seconds -----" % (time() - start_time))
                if self.conf['ADDNOISE']:
                    #print("[GrismGenerator][debug]: add noise")
                    return spec_1D+noise, noise
                else:
                    #print("[GrismGenerator][debug]: noise free")
                    return spec_1D, noise

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
                scale_unit = kwargs.get('scale_unit', galsim.arcsec)
                return galsim.Airy(lam=lam, diam=config['DIAMETER']/100,
                                scale_unit=scale_unit)
            elif _type == 'airy_mean':
                scale_unit = kwargs.get('scale_unit', galsim.arcsec)
                #return galsim.Airy(config['psf_fwhm']/1.028993969962188,
                #               scale_unit=scale_unit)
                lam = kwargs.get("lam_mean", 1000) # nm
                return galsim.Airy(lam=lam, diam=config['DIAMETER']/100,
                                scale_unit=scale_unit)
            elif _type == 'airy_fwhm':
                loverd = config['PSFFWHM']/1.028993969962188
                scale_unit = kwargs.get('scale_unit', galsim.arcsec)
                return galsim.Airy(lam_over_diam=loverd, scale_unit=scale_unit)
            elif _type == 'moffat':
                beta = config.get('PSFBETA', 2.5)
                fwhm = config.get('PSFFWHM', 0.5)
                return galsim.Moffat(beta=beta, fwhm=fwhm)
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
        random_seed = config.get('RANDSEED', int(time()*1000))
        rng = galsim.BaseDeviate(random_seed+1)

        _type = config.get('NOISETYP', 'ccd').lower()
        if _type == 'ccd':
            sky_level = config.get('SKYLEVEL', 0.65*1.2)
            read_noise = config.get('RDNOISE', 8.5)
            gain = config.get('GAIN', 1.0)
            exp_time = config.get('EXPTIME', 1.0)
            noise = galsim.CCDNoise(rng=rng, gain=gain,
                                read_noise=read_noise,
                                sky_level=sky_level*exp_time/gain)
        elif _type == 'gauss':
            sigma = config.get('NOISESIG', 1.0)
            noise = galsim.GaussianNoise(rng=rng, sigma=sigma)
        elif _type == 'poisson':
            sky_level = config.get('SKYLEVEL', 0.65*1.2)
            noise = galsim.PoissonNoise(rng=rng, sky_level=sky_level)
        else:
            raise ValueError(f'{self.noise_type} not implemented yet!')
        return noise

def get_datavector_types():
    return DATAVECTOR_TYPES

# NOTE: This is where you must register a new model
DATAVECTOR_TYPES = {
    'default': DataCube,
    'datacube': DataCube,
    }

def build_datavector(name, kwargs):
    '''
    name: str
        Name of datavector
    kwargs: dict
        Keyword args to pass to datavector constructor
    '''

    name = name.lower()

    if name in DATAVECTOR_TYPES.keys():
        # User-defined input construction
        datavector = DATAVECTOR_TYPES[name](**kwargs)
    else:
        raise ValueError(f'{name} is not a registered datavector!')

    return datavector

# Used for testing
def main(args):

    show = args.show

    outdir = os.path.join(utils.TEST_DIR, 'cube')
    utils.make_dir(outdir)

    li, le, dl = 500, 600, 1
    lambdas = np.arange(li, le, dl)

    throughput = '0.85'
    unit = 'nm'
    zp = 30
    bandpasses = []

    print('Building test bandpasses')
    for l1, l2 in zip(lambdas, lambdas+1):
        bandpasses.append(galsim.Bandpass(
            throughput, unit, blue_limit=l1, red_limit=l2, zeropoint=zp
            ))

    print('Testing bandpass helper func')
    bandpasses_alt = setup_simple_bandpasses(
        li, le, dl, throughput=throughput, unit=unit, zp=zp
        )
    assert bandpasses == bandpasses_alt

    Nspec = len(bandpasses)

    print('Building empty test data')
    shape = (Nspec, 100, 100)
    data = np.zeros(shape)
    pars = {
        'pix_scale': 1,
        'bandpasses': bandpasses
        }

    print('Building Slice object')
    n = 50 # slice num
    s = Slice(data[n,:,:], bandpasses[n])

    print('Testing slice plots')
    s.plot(show=False)

    print('Building SliceList object')
    sl = SliceList()
    sl.append(s)

    print('Building DataCube object from array')
    cube = DataCube(data=data, pars=pars)

    print('Build a bandpass list from a dict')
    dict_pars = {'pix_scale': 1}
    dict_pars['bandpasses'] = {
        'lambda_blue': li,
        'lambda_red': le,
        'dlambda': dl,
        'zp': 25.0
    }
    cube = DataCube(data=data, pars=dict_pars)

    print('Building DataCube with constant weight & mask')
    weights = 1. / 3
    masks = 0
    cube = DataCube(
        data=data, weights=weights, masks=masks, pars=pars
        )

    print('Building DataCube with weight & mask lists')
    weights = [i for i in range(Nspec)]
    masks = [0 for i in range(Nspec)]
    cube = DataCube(
        data=data, weights=weights, masks=masks, pars=pars
        )

    print('Building DataCube with weight & mask arrays')
    weights = 2 * np.ones(shape)
    masks = np.zeros(shape)
    masks[-1] = np.ones(shape[1:])
    cube = DataCube(
        data=data,  weights=weights, masks=masks, pars=pars
        )

    print('Testing DataCube truncation on slice centers')
    test_cube = cube.copy()
    lambda_range = le - li
    blue_cut = li + 0.25*lambda_range + 0.5
    red_cut  = li + 0.75*lambda_range - 0.5
    test_cube.truncate(blue_cut, red_cut, cut_type='center')
    nslices_cen = len(test_cube.slices)
    print(f'----Truncation resulted in {nslices_cen} slices')

    print('Testing DataCube truncation on slice edges')
    test_cube = cube.copy()
    test_cube.truncate(blue_cut, red_cut, cut_type='edge')
    nslices_edg = len(test_cube.slices)
    print(f'----Truncation resulted in {nslices_edg} slices')

    if nslices_edg != (nslices_cen-2):
        return 1

    print('Building DataCube from simulated fitscube file')
    mock_dir = os.path.join(utils.TEST_DIR,
                            'mocks',
                            'COSMOS')
    test_cubefile = os.path.join(
        mock_dir,'kl-mocks-COSMOS-001.fits'
        )
    bandpass_file = os.path.join(
        mock_dir, 'bandpass_list.pkl'
        )
    with open(bandpass_file, 'rb') as f:
        bandpasses = pickle.load(f)
    if os.path.exists(test_cubefile):
        print('Building from pickled bandpass list file in pars')
        pars['bandpasses'] = bandpasses
        fits_cube = DataCube.from_fits(test_cubefile, pars=pars)

        print('Building from bandpass list directly')
        fits_cube = DataCube.from_fits(
            test_cubefile, bandpasses=bandpasses, pix_scale=pars['pix_scale']
            )

        print('Making slice plot from DataCube')
        indx = fits_cube.Nspec // 2
        outfile = os.path.join(outdir, 'slice-plot.png')
        plot_kwargs = {
            'show': show,
            'outfile': outfile
        }
        fits_cube.plot_slice(indx, plot_kwargs)

        print('Making pixel spectrum plot from DataCube')
        box_size = fits_cube.slices[indx]._data.shape[0]
        i, j = box_size // 2, box_size // 2
        outfile = os.path.join(outdir, 'pixel-spec-plot.png')
        fits_cube.plot_pixel_spectrum(i, j, show=show, outfile=outfile)

        truth_file = os.path.join(mock_dir, 'truth.fits')
        if os.path.exists(truth_file):
            print('Loading truth catalog')
            truth = Table.read(truth_file)
            z = truth['zphot'][0]
            ha = 656.28 # nm
            ha_shift = (1+z) * ha

            print('Making pixel spectrum plot with true z')
            fits_cube.plot_pixel_spectrum(i, j, show=False, close=False)
            plt.axvline(
                ha_shift, lw=2, ls='--', c='k', label=f'(1+{z:.2})*H_alpha'
                )
            plt.legend()
            outfile = os.path.join(outdir, 'pix-spec-z.png')
            plt.savefig(outfile, bbox_inches='tight')
            if show is True:
                plt.show()

            print('Making aperture spectrum plot')
            radius = 4 # pixels
            offset = (0,0) # pixels
            fits_cube.plot_aperture_spectrum(radius, offset=offset,
                                             show=False, close=False)
            plt.axvline(
                ha_shift, lw=2, ls='--', c='k', label=f'(1+{z:.2})*H_alpha'
                )
            plt.legend()
            outfile = os.path.join(outdir, 'apt-spec-plot.png')
            plt.savefig(outfile, bbox_inches='tight')
            if show is True:
                plt.show()

    else:
        print('Files missing - skipping tests')

    print('Done!')

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
