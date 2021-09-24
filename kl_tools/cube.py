import numpy as np
from astropy.io import fits
import galsim
import os
import pickle
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

class DataCube(object):
#     '''
#     Base class for an abstract data cube.
#     Contains astronomical images of a source
#     at various wavelength slices
#     '''

    def __init__(self, data=None, shape=None, bandpasses=None):
        '''
        Initialize either a filled DataCube from an existing numpy
        array or an empty one from a given shape

        data: A numpy array containing all image slice data.
               For now, assumed to be the shape format given below.
        shape: A 3-tuple in the format of (Nspec, Nx, Ny)
               where (Nx, Ny) are the shapes of the image slices
               and Nspec is the Number of spectral slices.
        bandpasses: A list of galsim.Bandpass objects containing
               throughput function, lambda window, etc.
        '''

        if data is None:
            if shape is None:
                raise ValueError('Must instantiate a DataCube with either ' + \
                                 'a data array or a shape tuple!')

            self.Nx = shape[0]
            self.Ny = shape[1]
            self.Nspec = shape[2]
            self.shape = shape

            self._check_shape_params()
            self._data = np.zeros(self.shape)

        else:
            assert len(data.shape) == 3

            if bandpasses is None:
                raise ValueError('Must pass bandpasses if data is not None!')

            self._data = data
            self.shape = data.shape

            if data.shape[2] != len(bandpasses):
                raise ValueError('The length of the bandpasses must ' + \
                                 'equal the length of the third data dimension!')
            self.Nx = self.shape[0]
            self.Ny = self.shape[1]
            self.Nspec = self.shape[2]

        # a bit awkward, but this allows flexible setup for other params
        if bandpasses is None:
            raise ValueError('Must pass a list of bandpasses!')
        self.bandpasses = bandpasses

        # Not necessarily needed, but could help ease of access
        self.lambda_unit = self.bandpasses[0].wave_type
        self.lambdas = [] # Tuples of bandpass bounds in unit of bandpass
        for bp in bandpasses:
            li = bp.blue_limit
            le = bp.red_limit
            self.lambdas.append((li, le))

            # Make sure units are consistent
            # (could generalize, but not necessary)
            assert bp.wave_type == self.lambda_unit

        self._construct_slice_list()

        return

    def _check_shape_params(self):
        Nzip = zip(['Nx', 'Ny', 'Nspec'], [self.Nx, self.Ny, self.Nspec])
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
            self.slices.append(Slice(self._data[:,:,i], bp))

        return

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

        # aper_spec = np.sum(self._data[mask, :])

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

        return self._data[i,j,:]

    def truncate(self, blue_cut, red_cut, trunc_type='edge'):
        '''
        Return a truncated DataCube to slices between blue_cut and
        red_cut using either the lambda on a slice center or edge
        '''

        for l in [blue_cut, red_cut]:
            if (not isinstance(l, float)) and (not isinstance(l, int)):
                raise ValueError('Truncation wavelengths must be ints or floats!')

        if (blue_cut >= red_cut):
            raise ValueError('blue_cut must be less than red_cut!')

        if trunc_type not in ['edge', 'center']:
            raise ValueError('trunc_type can only be at the edge or center!')

        if trunc_type == 'center':
            # truncate on slice center lambda value
            lambda_means = np.mean(self.lambdas, axis=1)

            cut = (lambda_means >= blue_cut) & (lambda_means <= red_cut)

        else:
            # truncate on slice lambda edge values
            lambda_blues = np.array([self.lambdas[i][0] for i in range(self.Nspec)])
            lambda_reds  = np.array([self.lambdas[i][1] for i in range(self.Nspec)])

            cut = (lambda_blues >= blue_cut) & (lambda_reds  <= red_cut)

        # could either update attributes or return new DataCube
        # for now, just return a new one
        trunc_data = self._data[:,:,cut]

        # Have to do it this way as lists cannot be indexed by np arrays
        # trunc_bandpasses = self.bandpasses[cut]
        trunc_bandpasses = [self.bandpasses[i]
                            for i in range(self.Nspec)
                            if cut[i] == True]

        return DataCube(data=trunc_data, bandpasses=trunc_bandpasses)

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

class FitsDataCube(DataCube):
    '''
    Same as Datacube, but instantiated from a fitscube file
    and associated file containing bandpass list

    We assume the same structure as galsim.fits.writeCube()

    cubefile: location of fits cube
    bandpasses: either a filename of a bandpass list or the list
    '''

    def __init__(self, cubefile, bandpasses, dir=None):
        if dir is not None:
            cubefile = os.path.join(dir, cubefile)

        self.cubefile = cubefile

        fits_cube = galsim.fits.readCube(cubefile)
        Nimages = len(fits_cube)
        im_shape = fits_cube[0].array.shape
        data = np.zeros((im_shape[0], im_shape[1], Nimages))

        for i, im in enumerate(fits_cube):
            data[:,:,i] = im.array

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

        super(FitsDataCube, self).__init__(data=data, bandpasses=bandpasses)

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
    def __init__(self, data, bandpass):
        self._data = data
        self.bandpass = bandpass

        self.red_limit = bandpass.red_limit
        self.blue_limit = bandpass.blue_limit
        self.dlamda = self.red_limit - self.blue_limit
        self.lambda_unit = bandpass.wave_type

        return

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

    Nspec = len(bandpasses)

    print('Building empty test data')
    shape = (100, 100, Nspec)
    data = np.zeros(shape)

    print('Building Slice object')
    n = 50 # slice num
    s = Slice(data[:,:,n], bandpasses[n])

    print('Testing slice plots')
    s.plot(show=False)

    print('Building SliceList object')
    sl = SliceList()
    sl.append(s)

    print('Building DataCube object from array')
    cube = DataCube(data=data, bandpasses=bandpasses)

    print('Testing DataCube truncation on slice centers')
    lambda_range = le - li
    blue_cut = li + 0.25*lambda_range + 0.5
    red_cut  = li + 0.75*lambda_range - 0.5
    truncated = cube.truncate(blue_cut, red_cut, trunc_type='center')
    nslices_cen = len(truncated.slices)
    print(f'----Truncation resulted in {nslices_cen} slices')

    print('Testing DataCube truncation on slice edges')
    truncated = cube.truncate(blue_cut, red_cut, trunc_type='edge')
    nslices_edg = len(truncated.slices)
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
        fits_cube = FitsDataCube(test_cubefile, bandpass_file)

        print('Building from bandpass list directly')
        with open(bandpass_file, 'rb') as f:
            bandpasses = pickle.load(f)
        fits_cube = FitsDataCube(test_cubefile, bandpasses)

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
