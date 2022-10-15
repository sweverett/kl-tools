import numpy as np
import os, sys
import yaml
import matplotlib.colors as colors
import pdb, pudb

class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child
    https://stackoverflow.com/questions/4716533/how-to-attach-debugger-to-a-python-subproccess
    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin

        return

class MidpointNormalize(colors.Normalize):
    '''
    Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)

    e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
    '''

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

        return

    def __call__(self, value, clip=None):
        # Ignoring masked values and edge cases
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))

def read_yaml(yaml_file):
    with open(yaml_file, 'r') as stream:
        return yaml.safe_load(stream)

def build_map_grid(Nx, Ny):
    '''
    We define the grid positions as the center of pixels

    For a given dimension: if # of pixels is even, then
    image center is on pixel corners. Else, a pixel center
    '''

    # max distance in given direction
    # even pixel counts requires offset by 0.5 pixels
    Rx = (Nx // 2) - 0.5 * ((Nx-1) % 2)
    Ry = (Ny // 2) - 0.5 * ((Ny-1) % 2)

    x = np.arange(-Rx, Rx+1, 1)
    y = np.arange(-Ry, Ry+1, 1)

    assert len(x) == Nx
    assert len(y) == Ny

    X, Y = np.meshgrid(x, y, indexing='ij')

    return X, Y

def check_file(filename):
    '''
    Check if file exists; err if not
    '''

    if not os.path.exists(filename):
        raise OSError(f'{filename} does not exist!')

    return

def check_type(var, name, desired_type):
    '''
    Checks that the passed variable (with given name)
    is of the desired type
    '''

    if not isinstance(var, desired_type):
        raise TypeError(f'{name} must be a {desired_type}!')

    return

def check_types(var_dict):
    '''
    Check that the passed variables match the desired type.
    Convenience wrapper around check_type() for multiple variables

    var_dict: dict
        A dictionary in the format of name: (var, desired_type)
    '''

    for name, tup in var_dict.items():
        var, desired_type = tup
        check_type(var, name, desired_type)

    return

def check_req_fields(config, req, name=None):
    for field in req:
        if not field in config:
            raise ValueError(f'{name}config must have field {field}')

    return

def check_fields(config, req, opt, name=None):
    '''
    req: list of required field names
    opt: list of optional field names
    name: name of config type, for extra print info
    '''
    assert isinstance(config, dict)

    if name is None:
        name = ''
    else:
        name = name + ' '

    if req is None:
        req = []
    if opt is None:
        opt = []

    # ensure all req fields are present
    check_req_fields(config, req, name=name)

    # now check for fields not in either
    for field in config:
        if (not field in req) and (not field in opt):
            raise ValueError(f'{field} not a valid field for {name} config!')

    return

def make_dir(d):
    '''
    Makes dir if it does not already exist
    '''

    if not os.path.exists(d):
        os.mkdir(d)

    return

def get_base_dir():
    '''
    base dir is parent repo dir
    '''
    module_dir = get_module_dir()
    return os.path.dirname(module_dir)

def get_module_dir():
    return os.path.dirname(__file__)

def get_test_dir():
    base_dir = get_base_dir()
    test_dir = os.path.join(base_dir, 'tests')
    make_dir(test_dir) # will only create if it does not exist
    return test_dir

def set_cache_dir():
    basedir = get_base_dir()
    cachedir = os.path.join(basedir, '.cache/')
    make_dir(cachedir) # will only create if it does not exist
    return

BASE_DIR = get_base_dir()
MODULE_DIR = get_module_dir()
TEST_DIR = get_test_dir()
CACHE_DIR = set_cache_dir()
