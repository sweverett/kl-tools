import numpy as np
from scipy.sparse import identity, dia_matrix
import fitsio
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

import utils
import parameters
from datavector import DataVector

import ipdb

parser = ArgumentParser()

parser.add_argument('--show', action='store_true', default=False,
                    help='Set to show test plots')
parser.add_argument('--test', action='store_true', default=False,
                    help='Set to run tests')

class CubePars(parameters.MetaPars):
    '''
    Class that defines structure for DataCube meta parameters,
    e.g. image & emission line meta data
    '''

    _req_fields = ['pix_scale', 'bandpasses']
    _opt_fields = ['psf', 'emission_lines', 'files', 'shape', 'meta', 'truth']
    _gen_fields = ['wavelengths'] # not an allowed input field, but generated

    def __init__(self, pars):
        '''
        pars: dict
            Dictionary of meta pars for the DataCube
        '''

        super(CubePars, self).__init__(pars)

        self._bandpasses = None
        bp = self.pars['bandpasses']
        try:
            utils.check_type(bp, 'bandpasses', list)
        except TypeError:
            try:
                utils.check_type(bp, 'bandpasses', dict)
            except:
                raise TypeError('CubePars bandpass field must be a' +\
                                'list of galsim bandpasses or a dict!')

        self.build_bandpasses()

        self._lambdas = None
        self._lambda_unit = None
        self.build_wavelength_list()

        return

    def build_bandpasses(self, remake=False):
        '''
        Build a bandpass list from pars if not provided directly
        '''

        # sometimes it is already set in the parameter dict
        if (remake is False) & (self._bandpasses is not None):
            return

        bp = self.pars['bandpasses']

        if isinstance(bp, list):
            # we'll be lazy and just check the first entry
            if not isinstance(bp[0], galsim.Bandpass):
                raise TypeError('bandpass list must be filled with ' +\
                                'galsim.Bandpass objects!')
            bandpasses = bp
        else:
            # make this a separate dict
            bp_dict = deepcopy(bp)

            # already checked it is a list or dict
            bandpass_req = ['lambda_blue', 'lambda_red', 'dlambda']
            bandpass_opt = ['throughput', 'zp', 'unit', 'file']
            utils.check_fields(bp_dict, bandpass_req, bandpass_opt)

            args = [
                bp_dict.pop('lambda_blue'),
                bp_dict.pop('lambda_red'),
                bp_dict.pop('dlambda')
                ]

            kwargs = bp_dict

            bandpasses = setup_simple_bandpasses(*args, **kwargs)

        self._bandpasses = bandpasses
        self['bandpasses'] = bandpasses

        return

    def build_wavelength_list(self):
        '''
        Build a list of slice wavelength bounds (lambda_blue, lambda_red)
        given a set bandpass list
        '''

        if self._bandpasses is None:
            self.build_bandpasses()

        # Not necessarily needed, but could help ease of access
        self._lambda_unit = u.Unit(self._bandpasses[0].wave_type)
        self._lambdas = [] # Tuples of bandpass bounds in unit of bandpass
        for bp in self._bandpasses:
            # galsim bandpass limits are always stored in nm
            li = (bp.blue_limit * u.nm).to(self._lambda_unit).value
            le = (bp.red_limit * u.nm).to(self._lambda_unit).value
            self._lambdas.append((li, le))

            # Make sure units are consistent
            # (could generalize, but not necessary)
            assert bp.wave_type == self._lambda_unit
        self._lambdas = np.array(self._lambdas)

        self['wavelengths'] = self._lambdas

        return

    def reset(self):
        '''
        Reset any attributes or fields that were set during instantiation,
        not as a direct field input. Useful for datacube truncations
        '''

        self._bandpasses = None
        self._lambdas = None
        self._lambda_unit = None

        # remove any generated fields
        for field in self._gen_fields:
            del self[field]

        self.__init__(self.pars)

        return

    def set_shape(self, shape):
        '''
        Store the datacube shape in CubePars

        shape: (Nspec, Nx, Ny) tuple
        '''

        if len(shape) != 3:
            raise ValueError('shape must be a tuple of len 3!')

        # this will build a lambda list from the bandpasses, if not yet created
        lambdas = self.lambdas

        if shape[0] != len(lambdas):
            raise ValueError('The first dimension of shape must be the ' +\
                             'length as the bandpass & wavelength list!')

        self['shape'] = shape

        return

    @property
    def bandpasses(self):
        if self._bandpasses is None:
            self.build_bandpasses()

        return self._bandpasses

    @property
    def lambdas(self):
        if self._lambdas is None:
            self.build_wavelength_list()

        return self._lambdas

    def copy(self):
        return self.__copy__()

    def __copy__(self):
        return CubePars(deepcopy(self.pars))

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
            pars = CubePars(pars_dict)
        else:
            if (pix_scale is not None) or (bandpasses is not None):
                raise Exception('Cannot pass pix_scale or bandpasses if ' +\
                                'pars is set!')
            try:
                utils.check_type(pars, 'pars', CubePars)
            except TypeError:
                try:
                    utils.check_type(pars, 'pars', dict)
                except:
                    raise TypeError('pars must be either CubePars ' +\
                                    'or a dict!')
                pars = CubePars(pars)

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

        data = fitsio.read(cubefile)

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

def setup_simple_bandpasses(lambda_blue, lambda_red, dlambda,
                            throughput=1., zp=30., unit='nm', file=None):
    '''
    Setup list of bandpasses needed to instantiate a DataCube
    given the simple case of constant spectral resolution, throughput,
    and image zeropoints for all slices

    Useful for quick setup of tests and simulated datavectors

    lambda_blue: float
        Blue-end of datacube wavelength range
    lambda_red: float
        Rd-end of datacube wavelength range
    dlambda: float
        Constant wavelength range per slice
    throughput: float
        Throughput of filter of data slices
    unit: str
        The wavelength unit
    zeropoint: float
        Image zeropoint for all data slices
    '''

    li, lf = lambda_blue, lambda_red
    lambdas = [(l, l+dlambda) for l in np.arange(li, lf, dlambda)]

    bandpasses = []
    if file is None:
        for l1, l2 in lambdas:
            bandpasses.append(galsim.Bandpass(
                throughput, unit, blue_limit=l1, red_limit=l2, zeropoint=zp
                ))
    else:
        # build bandpass from file
        assert os.path.exists(file), f'Bandpass file {file} does not exist!'
        _bandpass = galsim.Bandpass(file, wave_type = unit)
        for l1, l2 in lambdas:
            bandpasses.append(_bandpass.truncate(blue_limit=l1, red_limit=l2))

    return bandpasses

class FiberPars(CubePars):

    # update the required/optional/generated fields
    _req_fields = ['model_dimension', 'sed', 'intensity', 'velocity', 'priors']
    _opt_fields = ['obs_conf',]
    _gen_fields = ['shape', 'pix_scale', 'bandpasses']

    def __init__(self, pars, obs_conf=None):
        ''' Fiber-related `Pars` obj
        Each `FiberPars` store the necessary parameters for one observation.

        Inputs:
        =======
        pars: dict obj
            Dictionary of meta pars for `FiberCube`. Note that this dict has
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
        # assure the input dictionary obj has required fields & obs config
        self._check_pars(pars)
        _pars = deepcopy(pars) # avoid shallow-copy of a dict object
        if obs_conf is None and _pars.get('obs_conf', None) is None:
            raise ValueError('Observation configuration is not set!')
        if obs_conf is not None:
            print(f'FiberPars: Overwrite the observation configuration')
            utils.check_type(obs_conf, 'obs_conf', dict)
            _pars['obs_conf'] = deepcopy(obs_conf)
        self.is_dispersed = (_pars['obs_conf']['OBSTYPE'] == 1)

        # set up bandpass, model cube shape, and model cube pixel scale
        _from_file_ = (_pars['obs_conf'].get('BANDPASS', None) is not None)
        if self.is_dispersed or (_from_file_==False):
            # focus on a narrow wavelength range
            _lrange = _pars['model_dimension']['lambda_range']
            _lblue, _lred = _lrange[0], _lrange[1]
            _dlam = _pars['model_dimension']['lambda_res']
            _Nlam = np.ceil( (_lred-_lblue)/_dlam ).astype(int)
        else:
            # need the full bandpass wavelength range to include all flux
            _bp_file_ = np.genfromtxt(_pars['obs_conf'].get('BANDPASS'))
            _lblue, _lred = _bp_file_[2,0], _bp_file_[-3,0]
            _Nlam = _bp_file_.shape[0]
            _dlam = (_lred-_lblue)/_Nlam

        _pars['shape'] = np.array([_Nlam, 
                         _pars['model_dimension']['Ny'], 
                         _pars['model_dimension']['Nx']], dtype=int)
        _pars['pix_scale'] = _pars['model_dimension']['scale']
        _pars['bandpasses'] = {
            'lambda_blue': _lblue, 'lambda_red': _lred,'dlambda': _dlam,
                'unit': _pars['model_dimension']['lambda_unit'],
        }
        if _from_file_:
            _pars['bandpasses']['file'] = _pars['obs_conf']['BANDPASS']
        else:
            _pars['bandpasses']['throughput'] = 1.0
        super(FiberPars, self).__init__(_pars)
        self.concatenated_bandpass = merge_bandpasses(self.bandpasses)
        self._bp_array = np.zeros(self.lambdas.shape)
        for i,_bp in enumerate(self.bandpasses):
            self._bp_array[i][0] = _bp(self.lambdas[i][0])
            self._bp_array[i][1] = _bp(self.lambdas[i][1])
        self.lambda_eff = np.sum(self.lambdas[:,0]*self.bp_array[:,0])/np.sum(self.bp_array[:,0])
        # update the obs_conf with the theoretical model cube parameters
        self.obs_index = _pars['obs_conf']['OBSINDEX']
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
        # self._bp_array = np.zeros(self.lambdas.shape)
        # for i,_bp in enumerate(self.bandpasses):
        #     self._bp_array[i][0] = _bp(self.lambdas[i][0])
        #     self._bp_array[i][1] = _bp(self.lambdas[i][1])
        return self._bp_array

class FiberModelCube(DataVector):
    ''' The most updated `FiberModelCube` implementation
    This class wraps the theoretical cube after it is generated to produce 
    simulated images.
    '''
    def __init__(self, Fpars):
        ''' Store the `FiberPars` object
        '''
        self.pars = Fpars
        # calculate the atmospheric PSF convolved fiber mask, if PSF model is
        # not sampled.
        if self.pars.is_dispersed:
            self.ATMPSF_conv_fiber_mask = self.get_PSF_convolved_fiber_mask()
            self.resolution_mat = self.get_resolution_matrix()
        else:
            self.ATMPSF_conv_fiber_mask = None
            self.resolution_mat = None
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
    def lambda_eff(self):
        return self.pars.lambda_eff

    def get_fiber_mask(self):
        mNx, mNy = self.pars['shape'][2], self.pars['shape'][1]
        mscale = self.pars['pix_scale']
        if self.pars.is_dispersed:
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
        if self.pars.is_dispersed:
            diameter_in_pixel = self.conf['FIBRBLUR']
            sigma = diameter_in_pixel / 4.
            x_in_pixel = np.arange(-5, 6)
            # assume Gaussian for now
            kernel = np.exp(-0.5*(x_in_pixel/sigma)**2)/((2*np.pi)**0.5*sigma)
        else:
            kernel = None
        return kernel
    def get_resolution_matrix(self):
        if self.pars.is_dispersed:
            diameter_in_pixel = self.conf['FIBRBLUR']
            sigma = diameter_in_pixel / 4.
            x_in_pixel = np.arange(-5, 6)
            # assume Gaussian for now
            kernel = np.exp(-0.5*(x_in_pixel/sigma)**2)/((2*np.pi)**0.5*sigma)
            # get the resolution matrix (sparse matrix)
            band = np.array([kernel]).repeat(self.pars['shape'][0], axis=0).T
            offset=np.arange(kernel.shape[0]//2, -(kernel.shape[0]//2)-1, -1)
            #print(band.shape, offset)
            #plt.imshow(band)
            Rmat = dia_matrix((band, offset), 
                shape=(self.pars['shape'][0], self.pars['shape'][0]))
        else:
            Rmat = None
        return Rmat

    def get_PSF_convolved_fiber_mask(self):
        ''' get atm-PSF convolved fiber mask
        '''
        mNx, mNy = self.pars['shape'][2], self.pars['shape'][1]
        mscale = self.pars['pix_scale']
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
        if not self.pars.is_dispersed:
            gal_chro = gal * sed.spectrum
            psf = self._build_PSF_model(self.conf, lam_mean=self.lambda_eff)
            gal = gal_chro if psf is None else galsim.Convolve([gal_chro, psf])
            img = gal.drawImage(
                bandpass = self.pars.concatenated_bandpass, 
                nx=self.conf['NAXIS1'], ny=self.conf['NAXIS2'], 
                scale=self.conf['PIXSCALE'], method='auto', 
                area=np.pi*(self.conf['DIAMETER']/2.)**2,
                exptime=self.conf['EXPTIME'],gain=self.conf['GAIN'])
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
            spec_1D = spec_1D*(self.bp_array[:,0]+self.bp_array[:,1])/2.
            factor = np.pi*(self.conf['DIAMETER']/2.)**2*self.conf['EXPTIME']/self.conf['GAIN']
            spec_1D = spec_1D * factor
            # fiber PSF can result in degrade in spectra resolution
            #kernel = self.get_fiber_1D_psf_kernel()
            #spec_1D = np.convolve(spec_1D, kernel, mode='same')
            if self.resolution_mat is not None:
                spec_1D = self.resolution_mat * spec_1D
            if force_noise_free:
                return spec_1D, None
            else:
                # place holder
                noise = spec_1D**(1/4)/2
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
        random_seed = config.get('RANDSEED', int(time()))
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

def merge_bandpasses(bandpasses):
    blim, rlim = bandpasses[0].blue_limit, bandpasses[-1].red_limit
    wtype, zp = bandpasses[0].wave_type, bandpasses[0].zeropoint
    waves = [(bp.blue_limit+bp.red_limit)/2. for bp in bandpasses]
    trans = [bp(w) for bp,w in zip(bandpasses, waves)]
    table = galsim.LookupTable(waves, trans)
    bandpass = galsim.Bandpass(table, wave_type=wtype, zeropoint=zp)
    return bandpass

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
