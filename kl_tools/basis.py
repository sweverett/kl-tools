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
from scipy.special import eval_hermitenorm, genlaguerre, factorial, gamma

import utils
import likelihood
from transformation import transform_coords

import pudb

parser = ArgumentParser()

parser.add_argument('--show', action='store_true', default=False,
                    help='Set to show test plots')
parser.add_argument('--test', action='store_true', default=False,
                    help='Set to run tests')

# TODO: In retrospect, it probably makes more sense to have theta_pars
#       passed at construction time, so a basis is always defined only
#       for a specific sample, as they are being constructed for each
#       sample anyway

class Basis(object):
    '''
    Base Basis class
    '''

    def __init__(self, name, nx, ny, pix_scale, plane, beta):
        '''
        name: str
            Name of basis
        nx: int
            Size of image on x-axis
        ny: int
            Size of image on y-axis
        pix_scale: float
            The pixel scale of the fitted image
        plane: str
            The image plane where the basis is defined;
            e.g. disk, obs, etc.
        beta: float
            Scale factor used to define basis functions. If
            None, then set automatically given (nx, ny)
        '''

        pars = {'nx':nx, 'ny':ny}
        for key, val in pars.items():
            if not isinstance(val, int):
                raise TypeError('{key} must be an int!')
        self.im_nx = nx
        self.im_ny = ny

        pars = {'name':name, 'plane':plane}
        for key, val in pars.items():
            if not isinstance(val, str):
                raise TypeError('{key} must be a str!')
        self.name = name
        self.plane = plane

        if not isinstance(pix_scale, (int,float)):
            raise TypeError('pix_scale must be an int or float!')
        self.pix_scale = pix_scale

        self._initialize()

        if beta is not None:
            if not isinstance(beta, (int, float)):
                raise TypeError('beta must be an int or float!')
            self.beta = beta
        else:
            self._set_default_beta()

        # NOTE: Can't set the number of basis functions used yet,
        # depends on the implementation
        self.N = None

        # Need to know if functions are complex
        self.is_complex = False

        return

    def _initialize(self):
        self._set_scale_params()

        # ...

        return

    def _set_scale_params(self):
        '''
        Set scale parameters given image size. Used for default
        choices of beta and N

        nx: int
            The number of pixels along the x-axis
        ny: int
            The number of pixels along the y-ayis

        TODO: Figure out if pixscale should be here
        '''

        self.theta_min = 1 # pixel
        self.theta_max = int(
            np.ceil(np.max([self.im_nx, self.im_ny]))
            )

        # self.theta_min = self.pix_scale # 1 pixel
        # self.theta_max = self.pix_scale * np.ceil(
        #     np.max([self.im_nx, self.im_ny])
        #     )

        return

    def _set_N(self, N):
        if not isinstance(N, int):
            raise TypeError('N must be an int!')
        if N <= 0:
            raise ValueError('N must be positive!')

        self.N = N

        return

    def _set_default_beta(self):
        '''
        Set beta automatically given image size
        '''

        self.beta = np.sqrt(self.theta_min * self.theta_max)

        return

    def get_basis_func(self, n):
        '''
        Return the function call that will evaluate the nth basis
        function at (x,y) positions
        '''

        if (n < 0) or (n >= self.N):
            raise ValueError(f'Basis functions range from 0 to {self.N-1}!')

        return self._get_basis_func(n)

    def render_im(self, theta_pars, coefficients, im_shape=None):
        '''
        Render image given transformation parameters andbasis coefficients

        theta_pars: dict
            A dict that holds the sampled transformation parameters
        coefficients: list, np.array
            A list or array of basis coefficients
        im_shape: tuple
            A (nx,ny) tuple for image bounds. If none, use the bounds
            during fitting
        '''

        if len(coefficients) != self.N:
            raise ValueError('The len of the passed coefficients ' +\
                             f'does not equal {self.N}!')

        if im_shape is None:
            nx, ny = self.im_nx, self.im_ny
        else:
            if len(im_shape) != 2:
                raise ValueError('im_shape must be a 2-tuple!')
            nx, ny = im_shape[0], im_shape[1]

        Xobs, Yobs = utils.build_map_grid(nx, ny)

        X, Y = transform_coords(
            Xobs, Yobs, 'obs', self.plane, theta_pars
            )

        if self.is_complex is True:
            im = np.zeros((nx, ny), dtype=np.complex128)
        else:
            im = np.zeros((nx, ny))

        for n in range(self.N):
            func, func_args = self._get_basis_func(n)
            args = [X, Y, *func_args]
            im += coefficients[n] * func(*args)

        if self.is_complex is True:
            im = im.real

        return im

    @abstractmethod
    def _get_basis_func(self, n):
        pass

# def guess_beta(self, datacube, bmin=None, bmax=None, n=None, pix_scale=None):
#     '''
#     Determines a good initial guess for beta given the stacked datacube.
#     Will override any currently set beta value (usually the default)

#     datacube: DataCube
#         The datacube whose stacked image we fit beta to
#     bmin: float
#         The minimum beta value to consider
#     bmax: float
#         The maximum beta value to consider
#     n: int
#         The number of beta values to scan
#     pix_scale: float
#         The pixel scale of the datacube images (if not saved to the dc)
#     '''

#     if datacube.pix_scale is None:
#         if pix_scale is None:
#             raise ValueError('pix_scale must be set either in the datacube ' +
#                              'if not explicitly!')
#         else:
#             if not isinstance(pix_scale, (int,float)):
#                 raise TypeError('pix_scale must be an int or float!')
#             if pix_scale <=0:
#                 raise ValueError('pix_scale must be positive!')
#     else:
#         if pix_scale is not None:
#             if datacubepix_scale != pix_scale:
#                 raise ValueError('Passed pix_scale not consistent with ' +
#                                  'datacube pix_scale!')
#         pix_scale = datacube.pix_scale

#     # if bmin is None:
#     #     bmin = 

#     bounds = {'bmin':bmin, 'bmax':bmax}
#     for name, val in bounds.items():
#         if val is not None:
#             if val <= 0:
#                 raise ValueError(f'{name} must be positive!')
#             if val >= np.min(datacube.Nx, datacube.Ny):
#                 raise ValueError(f'{name} must be smaller than the ' +
#                              'datacube image size!')

#     if bmin <= bmax:
#         raise ValueError('bmin must be less than bmax!')


#     betas = np.linspace(bmin, bmax, n)

#     # get stacked datacube image to fit to
#     image = datacube.stack()

#     self.beta = beta

#     return

class PolarBasis(Basis):
    '''
    Adds some functionality useful for all polar basis function classes

    As it does not implement _get_basis_func(), it is still an abstract
    class
    '''

    def _setup_lm_grid(self):
        '''
        Setup the correspondence between a (l,m) pair and the associated
        basis function n
        '''

        if (not hasattr(self, 'Nmax')) or (self.Nmax is None):
            raise AttributeError('Nmax must be set before setting up ' +\
                                 'the lm grid!')

        lm_grid = {}

        k = 0
        for l in range(0, self.Nmax+1):
            for m in range(-l, l+1):
                lm_grid[(l,m)] = k
                k += 1

        self.lm_grid = lm_grid
        self.lm_grid_inv = {val:key for key, val in lm_grid.items()}

        return

    def lm_to_n(self, l, m):
        '''
        Convert between (l, m) and the corresponding basis function N
        '''

        if not hasattr(self, 'lm_grid'):
            raise AttributeError('Must set up lm grid first!')

        return self.lm_grid[(l,m)]

    def n_to_lm(self, n):
        '''
        Convert between the Nth basis function and the corresponding (l, m)
        '''

        if not hasattr(self, 'lm_grid_inv'):
            raise AttributeError('Must set up lm grid first!')

        return self.lm_grid_inv[n]

class SersicletBasis(PolarBasis):
    '''
    See https://arxiv.org/abs/1106.6045
    '''

    def __init__(self, nx, ny, pix_scale, plane, index, beta=None,
                 b=None, Nmax=None):
        '''
        nx: int
            Size of the image x-axis
        ny: int
            Size of the image y-axis
        pix_scale: float
            The pixel scale of the datacube images
        plane: str
            The image plane where the basis is defined;
            e.g. disk, obs, etc.
        index: float
            Sersic index
        beta: float
            Scale factor used to define basis functions. If
            None, then set automatically given (nx, ny)
        b: float
            Sersic param
        Nmax: int
            Max radial order of sersiclets. If None, then
            set automatically given (nx, ny)
        '''

        super(SersicletBasis, self).__init__(
            'sersiclet', nx, ny, pix_scale, plane, beta
            )

        if Nmax is not None:
            self.Nmax = Nmax
        else:
            self._set_default_Nmax()

        # Compute number of independent basis vectors given Nmax
        N = (1+self.Nmax)**2
        self._set_N(N)

        for name, val in {'index':index, 'b':b}.items():
            if not isinstance(val, (int, float)):
                if val is None:
                    continue
                else:
                    raise TypeError(f'{name} must be an int or float!')

        if index <= 0:
            raise ValueError('The sersic index must be positive!')
        self.index = index

        if b is None:
            self._set_default_b()
        else:
            self.b = b

        self._setup_lm_grid()

        self.is_complex = True

        return

    def _set_default_Nmax(self):
        '''
        Set Nmax automatically given image size
        '''

        self.Nmax = int(
            np.ceil((self.theta_max / self.theta_min)) - 1
            )

        return

    def _set_default_b(self):
        '''
        Use series approx for b
        '''

        n = self.index

        assert n > 0.36

        self.b = 2.*n - 1./3 + 4./(405*n) + 46./(25515 * n**2) +\
                 131./(1148175 * n**3) - 2194697./(30690717750 * n**4)

        return

    def _get_basis_func(self, n):
        '''
        Return function call along with args
        needed to evaluate nth basis function at a
        given image position
        '''

        l, m = self.n_to_lm(n)

        args = [self.beta, l, m, self.index, self.b]

        return (self._eval_basis_function, args)

    @staticmethod
    def _eval_basis_function(x, y, beta, l, m, n, b):
        '''
        Evaluate sersiclet at all (x, y) values using
        def from https://arxiv.org/abs/1106.6045

        r: float, np.array
            The array of r values to evaluate at
        phi: float, np.array
            The array of phi values to evaluate at
        beta: float
            The scale factor of the disclets
        l: int
            The principle quantum number
        m: int
            The "magnetic" quantum number
        n: float
            The sersic index
        b: float
            Needed for general sersiclet
        '''

        r = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)

        k = 2*n - 1
        u = b * (r / beta)**(1/n)

        norm_inner = ( (beta**2 * n) / b**(2*n) ) * ( gamma(l + 2*n) / factorial(l) )
        norm = 1. / np.sqrt(norm_inner)

        lag = genlaguerre(l, k)(u)

        exp = np.exp( -(b/2.) * (r/beta)**(1./n) )

        rad = lag * exp

        ang = np.exp(-1j * m * phi)

        return norm * rad * ang

class ExpShapeletBasis(PolarBasis):
    '''
    See https://arxiv.org/abs/0809.3465,
        https://arxiv.org/abs/1106.6045,
        https://arxiv.org/abs/1903.05837
    '''

    def __init__(self, nx, ny, pix_scale, plane, beta=None,
                 Nmax=None):
        '''
        nx: int
            Size of the image x-axis
        ny: int
            Size of the image y-axis
        pix_scale: float
            The pixel scale of the datacube images
        plane: str
            The image plane where the basis is defined;
            e.g. disk, obs, etc.
        beta: float
            Scale factor used to define basis functions. If
            None, then set automatically given (nx, ny)
        Nmax: int
            Max Nx+Ny order. If None, then
            set automatically given (nx, ny)
        '''

        super(ExpShapeletBasis, self).__init__(
            'exp_shapelets', nx, ny, pix_scale, plane, beta
            )

        if Nmax is not None:
            self.Nmax = Nmax
        else:
            self._set_default_Nmax()

        # Compute number of independent basis vectors given Nmax
        N = (1+self.Nmax)**2
        self._set_N(N)

        self._setup_lm_grid()

        self.is_complex = True

        return

    def _set_default_Nmax(self):
        '''
        Set Nmax automatically given image size
        '''

        self.Nmax = int(
            np.ceil((self.theta_max / self.theta_min)) - 1
            )

        return

    @staticmethod
    def _factorial_ratio(num, den):
        '''
        evaluate the ratio of num! / den!
        '''

        for v in [num, den]:
            if not isinstance(v, int):
                raise TypeError('Factorial inputs must be ints!')

        # make it so the numerator is always greater
        if num < den:
            num, den = den, num
            flip = True
        else:
            flip = False

        diff = num - den

        product = 1

        for i in range(den+1, num+1):
            product *= i

        if flip is True:
            return 1. / product
        else:
            return product

    def _get_basis_func(self, n):
        '''
        Return function call along with args
        needed to evaluate nth basis function at a
        given image position
        '''

        l, m = self.n_to_lm(n)

        args = [l, m, self.beta]

        return (self._eval_basis_function, args)

    @staticmethod
    def _eval_basis_function(x, y, l, m, beta):
        '''
        Returns a single polar exp shapelet basis function of order
        (l,m) evaluated at the points (x,y).

        We use the def from https://arxiv.org/abs/1903.05837

        r: float, np.array
            The array of r values to evaluate at
        phi: float, np.array
            The array of phi values to evaluate at
        beta: float
            The scale factor of the exponential shapelets
        l: int
            The principle quantum number
        m: int
            The "magnetic" quantum number
        '''

        if l < 0:
            raise ValueError('l must be non-negative!')
        if abs(m) > l:
            raise ValueError('|m| cannot be greater than l!')

        r = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)

        norm1 = (-1)**l
        norm2 = np.sqrt(2. / (np.pi * beta * (2*l+1)**3))
        # norm3 = np.sqrt(factorial_ratio(n-abs(m), n+abs(m)))
        norm3 = np.sqrt(factorial(l-abs(m)) / factorial(l+abs(m)))

        norm = norm1 * norm2 * norm3

        rad1 = ((2.*r) / (beta * (2*l+1)))**abs(m)

        lag = genlaguerre(l-abs(m), 2*abs(m))
        rad2 = lag( (2.*r) / (beta*(2*l+1)) )

        exp = np.exp( -r / (beta * (2*l+1)) )

        rad = rad1 * rad2 * exp

        ang = np.exp(-1j * m * phi)

        return norm * rad * ang

class ShapeletBasis(Basis):
    def __init__(self, nx, ny, pix_scale, plane, beta=None, Nmax=None):
        '''
        nx: int
            Size of the image x-axis
        ny: int
            Size of the image y-axis
        pix_scale: float
            The pixel scale of the datacube images
        plane: str
            The image plane where the basis is defined;
            e.g. disk, obs, etc.
        beta: float
            Scale factor used to define basis functions. If
            None, then set automatically given (nx, ny)
        Nmax: int
            Maximum N=Nx+Ny to use for shapelets. If None, then
            set automatically given (nx, ny)
        '''

        super(ShapeletBasis, self).__init__(
            'shapelet', nx, ny, pix_scale, plane, beta
            )

        if Nmax is not None:
            self.Nmax = Nmax
        else:
            self._set_default_Nmax()

        # Compute number of independent basis vectors given Nmax
        N = (self.Nmax+1) * (self.Nmax+2) // 2

        self._set_N(N)

        self._setup_nxny_grid()

        return

    def _set_default_Nmax(self):
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

    def NxNy_to_n(self, nx, ny):
        if (nx+ny) > self.Nmax:
            raise ValueError(f'Nx+Ny cannot be greater than nmax={self.Nmax}!')
        return int(self.ngrid[nx, ny])

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
    'sersiclets': SersicletBasis,
    # 'disclets': DiscletBasis,
    'exp_shapelets': ExpShapeletBasis,
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

    print('Creating a SersicletBasis')

    nx, ny = 30,30
    pix_scale = 0.1
    nmax = 4 # To limit the plot size
    shapelets = ShapeletBasis(nx, ny, pix_scale, 'disk', Nmax=nmax)

    outfile = os.path.join(outdir, 'shapelet-basis-funcs.png')
    print(f'Saving plot of shapelet basis functions to {outfile}')
    shapelets.plot_basis_funcs(outfile=outfile, show=show)

    print('Creating a ShapeletBasis')
    index = 1
    sersiclets = SersicletBasis(nx, ny, pix_scale, 'obs', index, b=1, Nmax=nmax)

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
