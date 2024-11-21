# TODO: Adapt code to use util functions from terminus instead of this file!

from pathlib import Path
import numpy as np
import os, sys
import yaml
import matplotlib.pyplot as plt
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
    def __init__(self, vmin, vmax, midpoint=0, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        try:
            normalized_min = max(0, 1 / 2 * (1 - abs((self.midpoint - self.vmin) / (self.midpoint - self.vmax))))
        except ZeroDivisionError:
            normalized_min = self.midpoint-1

        try:
            normalized_max = min(1, 1 / 2 * (1 + abs((self.vmax - self.midpoint) / (self.midpoint - self.vmin))))
        except ZeroDivisionError:
            normalized_max = self.midpoint+1

        normalized_mid = 0.5
        x, y = [self.vmin, self.midpoint, self.vmax], [normalized_min, normalized_mid, normalized_max]
        return np.ma.masked_array(np.interp(value, x, y))

def read_yaml(yaml_file):
    with open(yaml_file, 'r') as stream:
        return yaml.safe_load(stream)

def build_map_grid(Nx, Ny, indexing='ij'):
    '''
    We define the grid positions as the center of pixels

    For a given dimension: if # of pixels is even, then
    image center is on pixel corners. Else, a pixel center

    Parameters:
    Nx, Ny: int
        Number of pixels in the x and y directions
    indexing: str
        If passing 2D numpy arrays for x & y, set the indexing to be either
        'ij' or 'xy'. Consistent with numpy.meshgrid `indexing` arg:
            - 'ij' if using matrix indexing; [i,j]=(x,y)
            - 'xy' if using cartesian indexing; [i,j]=(y,x) 
        If the equivalent 1D arrays are such that len(x) = M and len(y) = N,
        then the 2D arrays should be of shape (N,M) if indexing='xy' and
        (M,N) if indexing='ij'. Default is 'ij'. For more info, see:
        https://numpy.org/doc/stable/reference/generated/numpy.meshgrid.html
    '''

    # max distance in given direction
    # even pixel counts requires offset by 0.5 pixels
    Rx = (Nx // 2) - 0.5 * ((Nx-1) % 2)
    Ry = (Ny // 2) - 0.5 * ((Ny-1) % 2)

    x = np.arange(-Rx, Rx+1, 1)
    y = np.arange(-Ry, Ry+1, 1)

    assert len(x) == Nx
    assert len(y) == Ny

    X, Y = np.meshgrid(x, y, indexing=indexing)

    # more for clarity than any intrinsic check
    if indexing == 'ij':
        assert X.shape == (Nx, Ny)
    if indexing == 'xy':
        assert X.shape == (Ny, Nx)

    return X, Y

def get_image_center(Nx, Ny):
    '''
    Get the image center in array coordinates. For even pixel counts, the center
    will be on a pixel corner, not the center of a pixel. Pixel centers are
    defied as the center of the pixel.
    '''

    x0 = Nx // 2 + 0.5 * ((Nx+1) % 2)
    y0 = Ny // 2 + 0.5 * ((Ny+1) % 2)

    return x0, y0

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

def make_dir(d, recursive=True):
    '''
    Makes dir if it does not already exist

    recursive: bool
        If True, will make all parent dirs if they do not exist
    '''

    if not os.path.exists(d):
        if recursive is True:
            os.makedirs(d)
        else:
            os.mkdir(d)

    return

def add_colorbar(im, aspect=10, pad_fraction=0.5, **kwargs):
    '''
    Add a vertical color bar to an image plot

    kwargs: dict
        The kwargs passed to colorbar()
    '''

    from mpl_toolkits import axes_grid1
    import matplotlib.pyplot as plt

    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)

    current_ax = plt.gca()
    cax = divider.append_axes('right', size=width, pad=pad)
    plt.sca(current_ax)

    return im.axes.figure.colorbar(im, cax=cax, **kwargs)

def plot(show, save, out_file=None):
    '''
    helper function to streamline plotting options
    '''

    if save is True:
        if out_file is None:
            raise ValueError('Must provide out_file if save is True')
        plt.savefig(out_file)
    if show is True:
        plt.show()
    else:
        plt.close()

    return

def get_base_dir() -> Path:
    '''
    base dir is parent repo dir
    '''
    module_dir = get_module_dir()
    return module_dir.parent

def get_module_dir():
    return Path(__file__).parent

def get_test_dir():
    base_dir = get_base_dir()
    return base_dir / 'tests'

def get_script_dir():
    base_dir = get_base_dir()
    return base_dir / 'scripts'

BASE_DIR = get_base_dir()
MODULE_DIR = get_module_dir()
TEST_DIR = get_test_dir()
SCRIPT_DIR = get_script_dir()
