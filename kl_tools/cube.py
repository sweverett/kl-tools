from abc import abstractmethod
import numpy as np
import fitsio
from astropy.io import fits
import astropy.units as u
import galsim
import os
import pickle
from copy import deepcopy
from astropy.table import Table
import matplotlib.pyplot as plt
from argparse import ArgumentParser

# from . import utils
import utils

import pudb

parser = ArgumentParser()

parser.add_argument('--show', action='store_true', default=False,
                    help='Set to show test plots')
parser.add_argument('--test', action='store_true', default=False,
                    help='Set to run tests')

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

class DataCube(DataVector):
    '''
    Base class for an abstract data cube.
    Contains astronomical images of a source
    at various wavelength slices
    '''

    def __init__(self, data=None, shape=None, bandpasses=None,
                 pix_scale=None, weights=None, masks=None, pars=None):
        '''
        Initialize either a filled DataCube from an existing numpy
        array or an empty one from a given shape

        data: np.array
            A numpy array containing all image slice data.
            For now, assumed to be the shape format given below.
        shape: tuple
            A 3-tuple in the format of (Nspec, Nx, Ny)
            where (Nx, Ny) are the shapes of the image slices
            and Nspec is the Number of spectral slices.
        bandpasses: list
            A list of galsim.Bandpass objects containing
            throughput function, lambda window, etc.
        pix_scale: float
            the pixel scale of the datacube slices
        weights: int, list, np.ndarray
            The weight maps for each slice. See set_weights()
            for more info on acceptable types.
        masks: np.ndarray
            The mask maps for each slice. See set_masks()
            for more info on acceptable types.
        pars: dict
            A dictionary that holds any additional metadata
        '''

        # partitioned as it is useful for datacube truncation
        # as well
        self._setup_datacube(
            data=data,
            shape=shape,
            bandpasses=bandpasses,
            pix_scale=pix_scale,
            weights=weights,
            masks=masks,
            pars=pars
            )

        return

    def _setup_datacube(self, data=None, shape=None, bandpasses=None,
                       pix_scale=None, weights=None, masks=None, pars=None):
        '''
        This part is separate from the constructor as it is useful
        for datacube truncation as well

        see constructor for arg defs
        '''

        if data is None:
            if shape is None:
                raise ValueError('Must instantiate a DataCube with either ' + \
                                 'a data array or a shape tuple!')

            self.Nspec = shape[0]
            self.Nx = shape[1]
            self.Ny = shape[2]
            self.shape = shape

            self._check_shape_params()
            self._data = np.zeros(self.shape)

        else:
            if bandpasses is None:
                raise ValueError('Must pass bandpasses if data is not None!')

            if len(data.shape) != 3:
                # Handle the case of 1 slice
                assert len(data.shape) == 2
                data = data.reshape(1, data.shape[0], data.shape[1])

            self.shape = data.shape

            self.Nspec = self.shape[0]
            self.Nx = self.shape[1]
            self.Ny = self.shape[2]

            self._data = data

            if self.shape[0] != len(bandpasses):
                raise ValueError('The length of the bandpasses must ' + \
                                 'equal the length of the third data dimension!')

        if self.Nspec == 0:
            print('WARNING: there are no slices in passed datacube. ' +\
                  'Are you sure that is right?')

        # a bit awkward, but this allows flexible setup for other params
        if bandpasses is None:
            raise ValueError('Must pass a list of bandpasses!')
        self.bandpasses = bandpasses

        d = {'pix_scale': (pix_scale, (int, float)), 'pars': (pars, dict)}
        for name, (val, t) in d.items():
            if val is not None:
                if not isinstance(val, t):
                    raise TypeError(f'{name} must be a {t}!')

        if pix_scale is None:
            if pars is None:
                raise AttributeError('Must pass one of pix_scale and pars!')
            if 'pix_scale' not in pars.keys():
                raise AttributeError('Must pass pix_scale explicitly ' +\
                                     'or in pars!')
            pix_scale = pars['pix_scale']
        else:
            if pars is None:
                pars = {'pix_scale': pix_scale}
            else:
                if 'pix_scale' in pars.keys():
                    if pix_scale != pars['pix_scale']:
                        raise ValueError('pix_scale does not match value in pars!')
                else:
                    pars['pix_scale'] = pix_scale

        self.pix_scale = pix_scale
        self.pars = pars

        # Not necessarily needed, but could help ease of access
        self.lambda_unit = u.Unit(self.bandpasses[0].wave_type)
        self.lambdas = [] # Tuples of bandpass bounds in unit of bandpass
        for bp in bandpasses:
            # galsim bandpass limits are always stored in nm
            li = (bp.blue_limit * u.nm).to(self.lambda_unit).value
            le = (bp.red_limit * u.nm).to(self.lambda_unit).value
            self.lambdas.append((li, le))

            # Make sure units are consistent
            # (could generalize, but not necessary)
            assert bp.wave_type == self.lambda_unit

        # If weights/masks not passed, set non-informative defaults
        if weights is not None:
            self.set_weights(weights)
        else:
            self.weights = np.ones(self.shape)
        if masks is not None:
            self.set_masks(masks)
        else:
            self.masks = np.zeros(self.shape)

        self._construct_slice_list()

        return

    def _check_shape_params(self):
        Nzip = zip(['Nspec', 'Nx', 'Ny'], [self.Nspec, self.Nx, self.Ny])
        for name, val in Nzip:
            if val < 1:
                raise ValueError(f'{name} must be greater than 0!')

        if len(self.shape) != 3:
            raise ValueError('DataCube.shape must be len 3!')

        return

    def _construct_slice_list(self):
        self.slices = SliceList()

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

    @property
    def data(self):
        return self._data

    def slice(self, indx):
        return self.slices[indx].data

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

    def copy(self):
        return deepcopy(self)

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

    def truncate(self, blue_cut, red_cut, lambda_unit=None, trunc_type='edge'):
        '''
        Modify existing datacube to only hold slices between blue_cut
        and red_cut using either the lambda on a slice center or edge
        '''

        for l in [blue_cut, red_cut]:
            if (not isinstance(l, float)) and (not isinstance(l, int)):
                raise ValueError('Truncation wavelengths must be ints or floats!')

        if (blue_cut >= red_cut):
            raise ValueError('blue_cut must be less than red_cut!')

        if trunc_type not in ['edge', 'center']:
            raise ValueError('trunc_type can only be at the edge or center!')

        # make sure we get a correct comparison between possible
        # unit differences
        lu = self.lambda_unit

        if lambda_unit is None:
            lambda_unit = self.lambda_unit
        blue_cut *= lambda_unit
        red_cut *= lambda_unit

        if trunc_type == 'center':
            # truncate on slice center lambda value
            lambda_means = np.mean(self.lambdas, axis=1)

            cut = (lu*lambda_means >= blue_cut) & (lambda_means <= red_cut)

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

        # Have to do it this way as lists cannot be indexed by np arrays
        # trunc_bandpasses = self.bandpasses[cut]
        trunc_bandpasses = [self.bandpasses[i]
                            for i in range(self.Nspec)
                            if cut[i] == True]


        # use main part of base constructor to reset affected attributes,
        # regardless of potential subclass
        self._setup_datacube(
            data=trunc_data,
            bandpasses=trunc_bandpasses,
            weights=trunc_weights,
            masks=trunc_masks,
            pars=self.pars
            )

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

class FitsDataCube(DataCube):
    '''
    Same as Datacube, but instantiated from a fitscube file
    and associated file containing bandpass list

    Assumes the datacube has a shape of (Nspec,Nx,Ny)

    cubefile: str
        Location of fits cube
    bandpasses: str, list
        Either a filename of a bandpass list or the list
    '''

    def __init__(self, cubefile, bandpasses, dir=None, **kwargs):
        if dir is not None:
            cubefile = os.path.join(dir, cubefile)

        utils.check_file(cubefile)

        self.cubefile = cubefile

        data = fitsio.read(cubefile)

        if isinstance(bandpasses, str):
            bandpass_file = bandpasses
            if '.pkl' in bandpass_file:
                with open(bandpass_file, 'rb') as f:
                    bandpasses = pickle.load(f)
            else:
                raise Exception('For now, only pickled lists of ' +\
                                'galsim.Bandpass objects are accepted')
        else:
            if not isinstance(bandpasses, list):
                raise Exception('For now, must pass bandpasses as either filename or list!')

        super(FitsDataCube, self).__init__(
            data=data, bandpasses=bandpasses, **kwargs
            )

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
                            throughput=1., zp=30., unit='nm'):
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
    for l1, l2 in lambdas:
        bandpasses.append(galsim.Bandpass(
            throughput, unit, blue_limit=l1, red_limit=l2, zeropoint=zp
            ))
    bandpasses = [galsim.Bandpass(
        throughput, unit, blue_limit=l1, red_limit=l2, zeropoint=zp
        ) for l1,l2 in lambdas]

    return bandpasses

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
    pars = {'pix_scale': 1}

    print('Building Slice object')
    n = 50 # slice num
    s = Slice(data[n,:,:], bandpasses[n])

    print('Testing slice plots')
    s.plot(show=False)

    print('Building SliceList object')
    sl = SliceList()
    sl.append(s)

    print('Building DataCube object from array')
    cube = DataCube(data=data, bandpasses=bandpasses, pars=pars)

    print('Building DataCube with constant weight & mask')
    weights = 1. / 3
    masks = 0
    cube = DataCube(
        data=data, bandpasses=bandpasses, weights=weights, masks=masks,
        pars=pars
        )

    print('Building DataCube with weight & mask lists')
    weights = [i for i in range(Nspec)]
    masks = [0 for i in range(Nspec)]
    cube = DataCube(
        data=data, bandpasses=bandpasses, weights=weights, masks=masks,
        pars=pars
        )

    print('Building DataCube with weight & mask arrays')
    weights = 2 * np.ones(shape)
    masks = np.zeros(shape)
    masks[-1] = np.ones(shape[1:])
    cube = DataCube(
        data=data, bandpasses=bandpasses, weights=weights, masks=masks,
        pars=pars
        )

    print('Testing DataCube truncation on slice centers')
    test_cube = cube.copy()
    lambda_range = le - li
    blue_cut = li + 0.25*lambda_range + 0.5
    red_cut  = li + 0.75*lambda_range - 0.5
    test_cube.truncate(blue_cut, red_cut, trunc_type='center')
    nslices_cen = len(test_cube.slices)
    print(f'----Truncation resulted in {nslices_cen} slices')

    print('Testing DataCube truncation on slice edges')
    test_cube = cube.copy()
    test_cube.truncate(blue_cut, red_cut, trunc_type='edge')
    nslices_edg = len(test_cube.slices)
    print(f'----Truncation resulted in {nslices_edg} slices')

    if nslices_edg != (nslices_cen-2):
        return 1

    print('Building DataCube from simulated fitscube file')
    mock_dir = os.path.join(utils.TEST_DIR,
                            'mocks',
                            'COSMOS')
    test_cubefile = os.path.join(mock_dir,
                                 'kl-mocks-COSMOS-001.fits')
    bandpass_file = os.path.join(mock_dir,
                                 'bandpass_list.pkl')
    if (os.path.exists(test_cubefile)) and (os.path.exists(bandpass_file)):
        print('Building from pickled bandpass list file')
        fits_cube = FitsDataCube(
            test_cubefile, bandpass_file, pars=pars
            )

        print('Building from bandpass list directly')
        with open(bandpass_file, 'rb') as f:
            bandpasses = pickle.load(f)
        fits_cube = FitsDataCube(
            test_cubefile, bandpasses, pars=pars
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
