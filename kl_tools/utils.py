import numpy as np
import os, sys
import yaml
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

    X, Y = np.meshgrid(x, y)

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
    return os.path.join(base_dir, 'tests')

BASE_DIR = get_base_dir()
MODULE_DIR = get_module_dir()
TEST_DIR = get_test_dir()
