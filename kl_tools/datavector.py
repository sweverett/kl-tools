from abc import abstractmethod

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
