import numpy as np
import pudb

class DataCube(object):
#     '''
#     Base class for an abstract data cube.
#     Contains astronomical images of a source
#     at various wavelength slices
#     '''

    def __init__(self, data=None, shape=None, lambda_array=None):
        '''
        Initialize either a filled DataCube from an existing numpy
        array or an empty one from a given shape

        data: A numpy array containing all image slice data.
               For now, assumed to be the shape format given below.
        shape: A 3-tuple in the format of (Nx, Ny, Nspec)
               where (Nx, Ny) are the shapes of the image slices
               and Nspec is the Number of spectral slices.
        lambda_data: A list or numpy array of lambda of each data
               slice. Assumed to be the center of the spectral window.
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


            self._data = data
            self.shape = data.shape

            if data.shape[2] != len(lambda_array):
                raise ValueError('The length of the lambda_array must ' + \
                                 'equal the length of the third data dimension!')
            self.Nx = self.shape[0]
            self.Ny = self.shape[1]
            self.Nspec = self.shape[2]

        self.lambda_array = lambda_array

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
            lam = self.lambda_array[i]
            self.slices.append(Slice(self._data[:,:,i], lam))

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
    corresponding to a source observation at
    give wavelength
    '''
    def __init__(self, data, wavelength):
        self._data = data
        self.wavelength = wavelength

        return

# Used for testing
def main():

    li, le, dl = 500, 600, 1
    lambda_array = np.arange(li, le+dl, dl)

    shape = (100, 100, len(lambda_array))
    data = np.zeros(shape)

    n = 50 # slice num
    wave = 500 # nm
    s = Slice(data[:,:,n], wave)

    sl = SliceList()
    sl.append(s)

    cube = DataCube(data=data, lambda_array=lambda_array)

    print('Done!')

    return 0

if __name__ == '__main__':
    rc = main()

    if rc == 0:
        print('All tests ran succesfully')
    else:
        print(f'Tests failed with return code of {rc}')
