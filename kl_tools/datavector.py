from abc import abstractmethod
from cube import DataCube

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
