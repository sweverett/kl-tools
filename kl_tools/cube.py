import numpy as np
from astropy.io import fits
import galsim
import os
import pickle
import utils
import pudb

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
        shape: A 3-tuple in the format of (Nx, Ny, Nspec)
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

        self.slices = self._construct_slice_list()

        return

    def _check_shape_params(self):
        Nzip = zip(['Nx', 'Ny', 'Nspec'], [self.Nx, self.Ny, self.Nspec])
        for name, val in Nzip:
            if val < 1:
                raise ValueError(f'{name} must be greater than 0!')

        if len(self.shape != 3):
            raise ValueError('DataCube.shape must be len 3!')

        return

    def _construct_slice_list(self):
        self.slices = SliceList()

        for i in range(self.Nspec):
            bp = self.bandpasses[i]
            self.slices.append(Slice(self._data[:,:,i], bp))

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

        # fits_cube = fits.open(cubefile)
        # pudb.set_trace()
        # Nimages = len(fits_cube)
        # im_shape = fits_cube[0].data.shape
        # data = np.zeros((im_shape[0], im_shape[1], Nimages))

        # for i, im in enumerate(fits_cube):
        #     data[:,:,i] = im
        fits_cube = galsim.fits.readCube(cubefile)
        Nimages = len(fits_cube)
        im_shape = fits_cube[0].array.shape
        data = np.zeros((im_shape[0], im_shape[1], Nimages))

        # pudb.set_trace()
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

# class DataCube(object):
#     '''
#     Base class for an abstract data cube.
#     Contains astronomical images of a source
#     at various wavelength slices
#     '''

#     def __init__(self, shape=None, Nx=None, Ny=None, Nspec=None)
#         '''
#         shape: a 3-tuple in the format of (Nx, Ny, Nspec)
#                where (Nx, Ny) are the shapes of the image slices
#                and Nspec is the Number of spectral slices.
#         '''

#         if shape is None:
#             for N in [Nx, Ny, Nspec]:
#                 if N is None:
#                     raise ValueError('If DataCube.shape is not set, ' + \
#                                      'then must pass Nx, Ny, and Nspec!')

#             self.Nx = Nx
#             self.Ny = Ny
#             self.Nspec = Nspec
#             self.shape = (Nx, Ny, Nspec)
#         else:
#             for N in [Nx, Ny, Nspec]:
#                 if N is not None:
#                     raise ValueError('Cannot pass a DataCube.shape ' + \
#                                      'as well as individual shape params!')


#         self.Nx = shape[0]
#         self.Ny = shape[1]
#         self.Nspec = shape[2]
#         self.shape = shape

#         self._check_shape_params()

#         self.slices = self._construct_slice_list()

#         return

#     def _check_shape_params(self):
#         Nzip = zip(['Nx', 'Ny', 'Nspec'], [self.Nx, self.Ny, self.Nspec])
#         for name, val in Nzip:
#             if val < 1:
#                 raise ValueError(f'{name} must be greater than 0!')

#         if len(self.shape != 3):
#             raise ValueError('DataCube.shape must be len 3!')

#         return

#     def _construct_slice_list(self):
#         slice_list = []

#         for i in range(self.Nspec):
#             slice_list.append(Slice(shape=))

#         # The slice list is useful for abstract representation,
#         # but for computations we want to work on a numpy array
#         self._slice_array = np.zeros(self.shape)

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

# Used for testing
def main():

    li, le, dl = 500, 600, 1
    lambdas = np.arange(li, le, dl)

    throughput = '0.85'
    unit = 'nm'
    zp = 30
    bandpasses = []

    print('Building bandpasses')
    for l1, l2 in zip(lambdas, lambdas+1):
        bandpasses.append(galsim.Bandpass(
            throughput, unit, blue_limit=l1, red_limit=l2, zeropoint=zp
            ))

    Nspec = len(bandpasses)

    shape = (100, 100, Nspec)
    data = np.zeros(shape)

    print('Building Slice object')
    n = 50 # slice num
    s = Slice(data[:,:,n], bandpasses[n])

    sl = SliceList()
    sl.append(s)

    print('Building DataCube object from array')
    cube = DataCube(data=data, bandpasses=bandpasses)

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
    else:
        print('Files missing - skipping test')

    print('Done!')

    return 0

if __name__ == '__main__':
    print('Starting tests')
    rc = main()

    if rc == 0:
        print('All tests ran succesfully')
    else:
        print(f'Tests failed with return code of {rc}')
