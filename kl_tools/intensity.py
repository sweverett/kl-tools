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

'''
This file contains a mix of IntensityMap classes for explicit definitions
(e.g. an inclined exponential) and IntensityMapFitter + Basis classes for
fitting a a chosen set of arbitrary basis functions to the stacked datacube
image
'''

parser = ArgumentParser()

parser.add_argument('--show', action='store_true', default=False,
                    help='Set to show test plots')
parser.add_argument('--test', action='store_true', default=False,
                    help='Set to run tests')

class IntensityMap(object):
    '''
    If needed, could have this be the return class of
    IntensityMapFitter.fit()

    name: str
        Name of intensity map type
    nx: int
        Size of image on x-axis
    ny: int
        Size of image on y-axis
    '''

    def __init__(self, name, nx, ny):

        if not isinstance(name, str):
            raise TypeError('IntensityMap name must be a str!')
        self.name = name

        for n in [nx, ny]:
            if not isinstance(n, int):
                raise TypeError('IntensityMap image size params must be ints!')
        self.nx = nx
        self.ny = ny

        self.image = None

        return

    @abstractmethod
    def render(self, theta_pars, pars):
        '''
        theta_pars: dict
            A dict of the sampled mcmc params for both the velocity
            map and the tranformation matrices
        pars: dict
            A dictionary of any additional parameters needed
            to render the intensity map
        redo: bool
            Set to remake rendered image regardless of whether
            it is already internally stored

        return: np.ndarray
            The rendered intensity map
        '''
        pass

    @abstractmethod
    def _render(self, theta_pars, pars):
        pass

class InclinedExponential(IntensityMap):
    '''
    This class is mostly for testing purposes. Can give the
    true flux, hlr for a simulated datacube to ensure perfect
    intensity map modeling for validation tests.

    We explicitly use an exponential over a general InclinedSersic
    as it is far more efficient to render, and is only used for
    testing anyway
    '''

    def __init__(self, nx, ny, flux=None, hlr=None):
        '''
        nx: int
            Size of image on x-axis
        ny: int
            Size of image on y-axis
        flux: float
            Object flux
        hlr: float
            Object half-light radius (in pixels)
        '''

        super(InclinedExponential, self).__init__('inclined_exp', nx, ny)

        pars = {'flux': flux, 'hlr': hlr}
        for name, val in pars.items():
            if val is None:
                pars[name] = 1.
            else:
                if not isinstance(val, (float, int)):
                    raise TypeError(f'{name} must be a float or int!')

        self.flux = pars['flux']
        self.hlr = pars['hlr']

        return

    def render(self, theta_pars, pars, redo=False):
        '''
        theta_pars: dict
            A dict of the sampled mcmc params for both the velocity
            map and the tranformation matrices
        pars: dict
            A dictionary of any additional parameters needed
            to render the intensity map
        redo: bool
            Set to remake rendered image regardless of whether
            it is already internally stored

        return: np.ndarray
            The rendered intensity map
        '''

        # only render if it has not been computed yet, or if
        # explicitly asked
        if self.image is not None:
            if redo is False:
                return image

        # A = theta_pars['A']
        inc = Angle(np.arcsin(theta_pars['sini']), radians)

        gal = gs.InclinedExponential(
            inc, flux=self.flux, half_light_radius=self.hlr
        )

        rot_angle = Angle(theta_pars['theta_int'], radians)
        gal = gal.rotate(rot_angle)

        # TODO: still don't understand why this sometimes randomly fails
        try:
            g1 = theta_pars['g1']
            g2 = theta_pars['g2']
            gal = gal.shear(g1=g1, g2=g2)
        except Exception as e:
            print('imap generation failed!')
            print(f'Shear values used: g=({g1}, {g2})')
            raise e

        # TODO: could generalize in future, but for now assume
        #       a constant PSF for exposures
        if 'psf' in pars:
            psf = pars['psf']
            gal = gs.Convolve([gal, psf])

        self.image = gal.drawImage(nx=self.nx, ny=self.ny).array

        return self.image

class BasisIntensityMap(IntensityMap):
    '''
    This is a catch-all class for Intensity Maps made by
    fitting arbitrary basis functions to the stacked
    datacube image

    TODO: Need a better name!
    '''

    def __init__(self, basis_type):
        # TODO: Add something here that does the fitting
        # with IntensityMapFitter ...
        return

    # def render(self, theta_pars, pars):
    #     return image

    # def _render(self, theta_pars, pars):
    #     pass

def get_intensity_types():
    return INTENSITY_TYPES

# NOTE: This is where you must register a new model
INTENSITY_TYPES = {
    'default': InclinedExponential,
    'inclined_exp': InclinedExponential,
    }

def build_intensity_map(name, kwargs):
    '''
    name: str
        Name of intensity map type
    kwargs: dict
        Keyword args to pass to intensity constructor
    '''

    name = name.lower()

    if name in INTENSITY_TYPES.keys():
        # User-defined input construction
        intensity = INTENSITY_TYPES[name](**kwargs)
    else:
        raise ValueError(f'{name} is not a registered intensity!')

    return intensity

class IntensityMapFitter(object):
    '''
    This base class represents an intensity map defined
    by some set of basis functions {phi_i}.
    '''
    def __init__(self, basis_type, nx, ny, basis_kwargs=None):
        '''
        basis_type: str
            The name of the basis_type type used
        nx: int
            The number of pixels in the x-axis
        ny: int
            The number of pixels in the y-ayis
        basis_kwargs: dict
            Keyword args needed to build given basis type
        '''

        self.basis_type = basis_type
        self.nx = nx
        self.ny = ny

        self.grid = utils.build_map_grid(nx, ny)

        self._initialize_basis(basis_kwargs)

        return

    def _initialize_basis(self, basis_kwargs):
        if basis_kwargs is not None:
            basis_kwargs['nx'] = self.nx
            basis_kwargs['ny'] = self.ny
        else:
            basis_kwargs = {
                'nx': self.nx,
                'ny': self.ny,
                }

        self.basis = build_basis(self.basis_type, basis_kwargs)
        self.Nbasis = self.basis.N
        self.mle_coefficients = np.zeros(self.Nbasis)

        self._initialize_pseudo_inv()

        return

    # TODO: Add @njit when ready
    def _initialize_pseudo_inv(self):
        '''
        Setup Moore-Penrose pseudo inverse given basis

        data: np.ndarray
            The sum of datacube slices corresponding to the emission
            line. This is what we will fit the intensity map to
        basis: list
            A list of functions that evaluate the ith basis function
            at the given location(s)
        '''


        Ndata = self.nx * self.ny
        Nbasis = self.Nbasis

        # build image grid vectors
        # X, Y = utils.build_map_grid(nx, ny)
        X, Y = self.grid
        x = X.reshape(Ndata)
        y = Y.reshape(Ndata)

        # the design matrix for a given basis and datacube
        M = np.zeros((Ndata, Nbasis))
        for n in range(Nbasis):
            func, func_args = self.basis.get_basis_func(n)
            args = [x, y, *func_args]
            M[:,n] = func(*args)

        # now compute the pseudo-inverse:
        self.pseudo_inv = np.linalg.pinv(M)

        # for n, b in enumerate(basis):
        #     X[:, n] = b(data_vec)

        # for n in range(Ndata):
        #     for m in range(Ndata):
        #         self.pseudo_inv[n,m] = self.basis

        return

    def fit(self, datacube, cov=None):
        '''
        Fit MLE of the intensity map for a given set of datacube
        slices

        NOTE: This assumes that the datacube is truncated to be only be
              around the relevant emission line region, though not
              necessarily with the continuum emission subtracted

        datacube: cube.DataCube
            The datacube to find the MLE intensity map of
        cov: np.ndarray
            The covariance matrix of the datacube images.
            NOTE: If the covariance matrix is of the form sigma*I for a
                  constant sigma, it does not contribute to the MLE. Thus
                  you should only pass cov if it is non-trivial
        '''

        nx, ny = self.nx, self.ny
        if (datacube.Nx, datacube.Ny) != (nx, ny):
            raise ValueError('DataCube must have same dimensions ' +\
                             'as intensity map!')

        # X, Y = utils.build_map_grid(nx, ny)

        # We will fit to the sum of all slices
        data = np.sum(datacube._data, axis=2).reshape(nx*ny)

        # Find MLE basis coefficients
        mle_coeff = self._fit_mle_coeff(data, cov=cov)

        assert len(mle_coeff) == self.Nbasis
        self.mle_coefficients = mle_coeff

        # Now create MLE intensity map
        mle_im = self.basis.render_im(mle_coeff)

        assert mle_im.shape == (nx, ny)
        self.mle_im = mle_im

        return mle_im

    def _fit_mle_coeff(self, data, cov=None):
        '''
        data: np.array
            The (nx*ny) data vector
        '''

        if cov is None:
            # The solution is simply the Moore-Penrose pseudo inverse
            # acting on the data vector
            mle_coeff = self.pseudo_inv.dot(data)

        else:
            # If cov is diagonal but not constant sigma, then see
            # SPECTRO-PERFECTIONISM (bolton, schlegel et al. 2009)
            raise NotImplementedError(
                'The MLE for intensity maps with non-trivial ' +\
                'covariance matrices is not yet implemented!'
                )

        return mle_coeff

    def plot_mle_fit(self, datacube, show=True, close=True, outfile=None,
                     size=(9,9)):
        '''
        datacube: DataCube
            Datacube that MLE fit was done on
        '''

        # fit was done on stacked datacube
        data = np.sum(datacube._data, axis=2)
        mle = self.mle_im

        fig, axes = plt.subplots(
            nrows=2, ncols=2, sharex=True, sharey=True, figsize=size
            )

        image = [data, mle, data-mle, 100.*(data-mle)/mle]
        titles = ['Data', 'MLE', 'Residual', '% Residual']

        for i in range(len(image)):
            ax = axes[i//2, i%2]
            if '%' in titles[i]:
                vmin, vmax = -100., 100.
            else:
                vmin, vmax = None, None
            im = ax.imshow(
                image[i], origin='lower', vmin=vmin, vmax=vmax
                )
            ax.set_title(titles[i])
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            plt.colorbar(im, cax=cax)

        plt.suptitle(f'MLE comparison for {self.Nbasis} basis functions')
        plt.tight_layout()

        if outfile is not None:
            plt.savefig(outfile, bbox_inches='tight', dpi=300)

        if show is True:
            plt.show()

        if close is True:
            plt.close()

        return

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

    def compute_design_matrix(self, x, y):
        '''
        Compute the design matrix given the basis definition and
        image positions (x,y)

        x: np.ndarray
            x positions of image in pixel coords
        y: np.ndarray
            y positions of image in pixel coords
        '''
        pass

        # phi = np.zeros()

        # return phi

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

    outdir = os.path.join(utils.TEST_DIR, 'intensity')
    utils.make_dir(outdir)

    print('Creating IntensityMapFitter object')
    fitter = IntensityMapFitter

    print('Creating a ShapeletBasis')
    nx, ny = 30,30
    nmax = 4 # To limit the plot size
    shapelets = ShapeletBasis(nx, ny, Nmax=nmax)

    outfile = os.path.join(outdir, 'shapelet-basis-funcs.png')
    print(f'Saving plot of shapelet basis functions to {outfile}')
    shapelets.plot_basis_funcs(outfile=outfile, show=show)

    print('Creating IntensityMapFitter w/ shapelet basis')
    nmax = 20
    basis_kwargs = {'Nmax': nmax}
    fitter = IntensityMapFitter(
        'shapelets', nx, ny, basis_kwargs=basis_kwargs
        )

    print('Setting up test datacube and true Halpha image')
    true_pars, pars = likelihood.setup_test_pars(nx, ny)

    li, le, dl = 655.8, 656.8, 0.1
    lambdas = [(l, l+dl) for l in np.arange(li, le, dl)]

    Nspec = len(lambdas)
    shape = (nx, ny, Nspec)
    datacube, sed, vmap, true_im = likelihood.setup_likelihood_test(
        true_pars, pars, shape, lambdas
        )

    outfile = os.path.join(outdir, 'datacube.fits')
    print(f'Saving test datacube to {outfile}')
    datacube.write(outfile)

    outfile = os.path.join(outdir, 'datacube-slices.png')
    print(f'Saving example datacube slice images to {outfile}')
    # if Nspec < 10:
    sqrt = int(np.ceil(np.sqrt(Nspec)))
    slice_indices = range(Nspec)

    k = 1
    for i in slice_indices:
        plt.subplot(sqrt, sqrt, k)
        plt.imshow(datacube.slices[i]._data, origin='lower')
        plt.colorbar()
        l, r = lambdas[i]
        plt.title(f'lambda=({l:.1f}, {r:.1f})')
        k += 1
    plt.gcf().set_size_inches(12,12)
    plt.tight_layout()
    plt.savefig(outfile, bbox_inches='tight', dpi=300)
    if show is True:
        plt.show()
    else:
        plt.close()

    print('Fitting simulated datacube with shapelet basis')
    start = time.time()
    mle_im = fitter.fit(datacube)
    t = time.time() - start
    print(f'Total fit time took {1000*t:.2f} ms for {fitter.Nbasis} basis funcs')

    outfile = os.path.join(outdir, 'compare-mle-to-data.png')
    print(f'Plotting MLE fit compared to stacked data to {outfile}')
    fitter.plot_mle_fit(datacube, outfile=outfile, show=show)

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
