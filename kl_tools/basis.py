import numpy as np
import os
import time
from abc import abstractmethod
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from argparse import ArgumentParser
import galsim as gs
from galsim.angle import Angle, radians
import astropy.constants as const
import astropy.units as units
from scipy.special import eval_hermitenorm
from scipy.special import factorial

import utils
import likelihood

import pudb

parser = ArgumentParser()

parser.add_argument('--show', action='store_true', default=False,
                    help='Set to show test plots')
parser.add_argument('--test', action='store_true', default=False,
                    help='Set to run tests')

class Basis(object):
    '''
    Base Basis class
    '''

    def __init__(self, name, N, nx, ny):
        '''
        name: str
            Name of basis
        N: int
            Number of basis functions to use
        nx: int
            Size of image on x-axis
        ny: int
            Size of image on y-axis
        '''

        self.name = name
        self.N = N
        self.im_nx = nx
        self.im_ny = ny

        self._initialize()

        return

    def _initialize(self):
        pass

    def get_basis_func(self, n):
        '''
        Return the function call that will evaluate the nth basis
        function at (x,y) positions
        '''

        if (n < 0) or (n >= self.N):
            raise ValueError(f'Basis functions range from 0 to {self.N-1}!')

        return self._get_basis_func(n)

    def render_im(self, coefficients):
        '''
        Render image given basis coefficients

        coefficients: list, np.array
            A list or array of basis coefficients
        '''

        if len(coefficients) != self.N:
            raise ValueError('The len of the passed coefficients ' +\
                             f'does not equal {self.N}!')

        nx, ny = self.im_nx, self.im_ny
        X, Y = utils.build_map_grid(nx, ny)

        im = np.zeros((nx, ny))

        for n in range(self.N):
            func, func_args = self._get_basis_func(n)
            args = [X, Y, *func_args]
            im += coefficients[n] * func(*args)

        return im

    @abstractmethod
    def _get_basis_func(self, n):
        pass

class ShapeletBasis(Basis):
    def __init__(self, nx, ny, beta=None, Nmax=None):
        '''
        nx: int
            Size of the image x-axis
        ny: int
            Size of the image y-axis
        beta: float
            Scale factor used to define basis functions. If
            None, then set automatically given (nx, ny)
        Nmax: int
            Nmaxumber of basis functions to use. If None, then
            set automatically given (nx, ny)
        '''

        pars = {'nx':nx, 'ny':ny}
        if Nmax is not None:
            pars['Nmax'] = Nmax
        for name, n in pars.items():
            if not isinstance(n, int):
                raise TypeError(f'{name} must be an int!')

        self.im_nx = nx
        self.im_ny = ny

        self._set_scale_params()

        if beta is not None:
            if not isinstance(beta, (int, float)):
                raise TypeError('beta must be an int or float!')
            self.beta = beta
        else:
            self._set_beta()

        if Nmax is not None:
            self.Nmax = Nmax
        else:
            self._set_Nmax()

        # Compute number of independent basis vectors given Nmax
        N = (self.Nmax+1) * (self.Nmax+2) // 2

        super(ShapeletBasis, self).__init__('shapelet', N, nx, ny)

        self._setup_nxny_grid()

        return

    def _set_scale_params(self):
        '''
        Set scale parameters given image size. Used for default
        choices of beta and N
        '''

        self.theta_min = 1 # pixel
        self.theta_max = int(
            np.ceil(np.max([self.im_nx, self.im_ny]))
            )

        return

    def _set_beta(self):
        '''
        Set beta automatically given image size
        '''

        self.beta = np.sqrt(self.theta_min * self.theta_max)

        return

    def _set_Nmax(self):
        '''
        Set Nmax automatically given image size
        '''

        self.Nmax = int(
            np.ceil((self.theta_max / self.theta_min)) - 1
            )

        return

    def _setup_nxny_grid(self):
        '''
        We define here the mapping between the nth basis
        function and the corresponding (nx, ny pairs)

        We do this by creating the triangular matrix
        corresponding to Nmax, ordering the positions
        such that each sub triangular matrix is consistent
        with one another
        '''

        nmax = self.Nmax

        self.ngrid = -1 * np.ones((nmax+1, nmax+1))

        x, y = 0, 0
        height = 0
        for i in range(self.N):
            self.ngrid[x, y] = i

            if self._is_triangular(i+1):
                x += height + 1
                y -= height
                height += 1
            else:
                x -= 1
                y += 1

        return

    @staticmethod
    def _is_triangular(n):
        '''
        Checks if n is a triangular number
        '''

        # this is equivalent to asking if m is square
        m = 8*n+1

        if np.sqrt(m) % 1 == 0:
            return True
        else:
            return False

    def _get_basis_func(self, n):
        '''
        Return function call along with args
        needed to evaluate nth basis function at a
        given image position
        '''

        Nx, Ny = self.n_to_NxNy(n)

        args = [self.beta, Nx, Ny]

        return (self._eval_basis_function, args)

    def n_to_NxNy(self, n):
        '''
        Return the (Nx, Ny) pair corresponding to the nth
        basis function

        We define this relationship by traveling through the
        triangular (nx, ny) grid in a zig-zag starting with x.
        Thus the ordering of each sub triangular matrix will be
        consistent.
        '''

        Nx, Ny  = np.where(self.ngrid == n)

        return int(Nx), int(Ny)

    def __call__(self, x, y, Nx, Ny):
        '''
        Evaluate the basis at positions (x,y) for order (nx, ny)
        '''

        if (Nx + Ny) > self.Nmax:
            raise ValueError(f'Nx+Ny cannot be greater than Nmax={self.Nmax}!')

        if x.shape != y.shape:
            raise ValueError('x and y must have the same shape!')

        return _eval_basis_function(x, y, self.beta, Nx, Ny)

    # TODO: Can use numba here if it is helpful
    @staticmethod
    def _eval_basis_function(x, y, beta, Nx, Ny):
        '''
        Returns a single Cartesian Shapelet basis function of order
        Nx, Ny evaluated at the points (x,y).

        Adapted from code by Eric Huff in Trillian
        '''

        bfactor = (1. / np.sqrt(beta))
        x_norm = 1. / np.sqrt(2**Nx * np.sqrt(np.pi) * factorial(Nx)) * bfactor
        y_norm = 1. / np.sqrt(2**Ny * np.sqrt(np.pi) * factorial(Ny)) * bfactor

        exp_x = np.exp(-(x/beta)**2 / 2.)
        exp_y = np.exp(-(y/beta)**2 / 2.)
        phi_x = x_norm * eval_hermitenorm(Nx, x/beta) * exp_x
        phi_y = y_norm * eval_hermitenorm(Ny, y/beta) * exp_y

        return phi_x * phi_y

    def plot_basis_funcs(self, outfile=None, show=True, close=True,
                         size=(9,9)):

        N = self.N
        nmax = self.Nmax
        sqrt = int(np.ceil(np.sqrt(N)))

        X, Y = utils.build_map_grid(self.im_nx, self.im_ny)

        fig, axes = plt.subplots(nrows=nmax+1, ncols=nmax+1, figsize=size,
                                 sharex=True, sharey=True)
        for i in range(N):
            nx, ny = self.n_to_NxNy(i)
            ax = axes[nx, ny]
            func, fargs = self.get_basis_func(i)
            args = [X, Y, *fargs]
            image = func(*args)
            im = ax.imshow(image, origin='lower')

            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            plt.colorbar(im, cax=cax)
            nx, ny = fargs[1], fargs[2]
            ax.set_title(f'({nx},{ny})')

        x = np.arange(nmax+1)
        xx, yy = np.meshgrid(x, x)
        empty = np.where((xx+yy) > nmax)

        for x,y in zip(empty[0], empty[1]):
            fig.delaxes(axes[x,y])

        plt.tight_layout()
        plt.subplots_adjust(hspace=0.15, wspace=0.5)

        if outfile is not None:
            plt.savefig(outfile, bbox_inches='tight', dpi=300)

        if show is True:
            plt.show()

        if close is True:
            plt.close()

        return

def get_basis_types():
    return BASIS_TYPES

# NOTE: This is where you must register a new model
BASIS_TYPES = {
    'default': ShapeletBasis,
    'shapelets': ShapeletBasis,
    }

def build_basis(name, kwargs):
    '''
    name: str
        Name of basis
    kwargs: dict
        Keyword args to pass to basis constructor
    '''

    name = name.lower()

    if name in BASIS_TYPES.keys():
        # User-defined input construction
        basis = BASIS_TYPES[name](**kwargs)
    else:
        raise ValueError(f'{name} is not a registered basis!')

    return basis

def main(args):
    '''
    For now, just used for testing the classes
    '''

    show = args.show

    outdir = os.path.join(utils.TEST_DIR, 'basis')
    utils.make_dir(outdir)

    print('Creating a ShapeletBasis')
    nx, ny = 30,30
    nmax = 4 # To limit the plot size
    shapelets = ShapeletBasis(nx, ny, Nmax=nmax)

    outfile = os.path.join(outdir, 'shapelet-basis-funcs.png')
    print(f'Saving plot of shapelet basis functions to {outfile}')
    shapelets.plot_basis_funcs(outfile=outfile, show=show)

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
