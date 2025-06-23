'''
A module that defines various basis functions to use in modeling the intensity 
maps of galaxy images.

NOTE: All basis function classes are defined in (x,y) coodinates in the units of
the passed beta parameters, and are assumed to be centered at (0,0). They are 
*not* defined in standard numpy (row,col) format.
'''

import numpy as np
from pathlib import Path
from abc import abstractmethod
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import galsim as gs
from scipy.special import eval_hermitenorm, genlaguerre, factorial, gamma

import kl_tools.utils as utils
from kl_tools.transformation import transform_coords

class Basis(object):
    '''
    Base Basis class

    name: str
        The name of the basis
    beta: float; default None
        Scale factor used to define basis functions, which effectively sets
        the units of the domain of the basis functions as well. If None, then
        set automatically given an sampled image of size (ncol,nrow)=(nx,ny) 
        whenever possible. NOTE: beta can always be changed after 
        initialization to satisfy different requirements.
    offset: tuple[float, float]; default None
        The offset of the basis from (0,0), in units of beta. If None, then
        the basis is centered at (0,0). NOTE: In most cases, you probably just
        want to apply an offset by using a transformation plane that is before 
        the offset later, such as `cen` or `disk`. Occasionally useful for
        testing purposes, but not recommended for general use.
    psf: galsim.GSObject
        A PSF model to convolve the basis by, if desired
    '''

    # TODO: we should add the capability of caching PSF-convolved basis
    # functions, so that we don't have to convolve them every time. This
    # requires certain assumptions, so something like the following:
    # 1) Only used in render_im() method, which is able to ensure that the
    #    position grid, PSF, pixel scale, etc. are constant within the method 
    #    scope
    # 2) Create a hash table based off of a (n, nx, ny, pixel_scale) tuple
    # 3) If the hash table already has the convolved basis function, then
    #    return it, otherwise convolve & cache

    # each subclass must override this, if necessary
    is_complex = False

    def __init__(self, name, beta=None, offset=None, psf=None):

        if not isinstance(name, str):
            raise TypeError('{key} must be a str!')
        self.name = name

        if beta is not None:
            if not isinstance(beta, (int, float)):
                raise TypeError('beta must be an int or float!')
        self.beta = beta

        if offset is not None:
            if not isinstance(offset, tuple):
                raise TypeError('offset must be a tuple!')
            if len(offset) != 2:
                raise ValueError('offset must be a 2-tuple!')
            for val in offset:
                if not isinstance(val, (int, float)):
                    raise TypeError('offset values must be ints or floats!')
        self.offset = offset

        if psf is not None:
            if not isinstance(psf, gs.GSObject):
                raise TypeError('psf must be a galsim.GSObject!')
        self.psf = psf

        # NOTE: Can't set the number of basis functions used yet,
        # depends on the implementation
        self.N = None

        # make sure any subclass didn't muck this up
        if not isinstance(self.is_complex, bool):
            raise TypeError('is_complex must be a bool!')

        return

    def _get_scale_params(self, nx, ny, pixel_scale):
        '''
        Set the minimum & maximum scale parameters given image size. Used for 
        default choices of beta and N

        nx: int
            The number of pixels along the x-axis
        ny: int
            The number of pixels along the y-ayis
        pixel_scale: float
            The pixel scale of the image

        return: tuple
            The scale parameters (theta_min, theta_max)
        '''

        # NOTE: This is the most basic implementation, but it should be
        # overridden by the child class if there is better information
        theta_min = pixel_scale
        theta_max = int(np.ceil(np.max([pixel_scale*nx, pixel_scale*ny])))

        return (theta_min, theta_max)

    def _set_N(self, N):
        if not isinstance(N, int):
            raise TypeError('N must be an int!')
        if N <= 0:
            raise ValueError('N must be positive!')

        self.N = N

        return

    def _get_default_beta(self, nx, ny, pixel_scale):
        '''
        Set beta automatically given image size
        '''

        theta_min, theta_max = self._get_scale_params(nx, ny, pixel_scale)
        default_beta = np.sqrt(theta_min * theta_max)

        return default_beta

    def set_default_beta(self, nx, ny, pixel_scale):
        '''
        Set the basis scale radius beta given some image parameters
        '''

        self.beta = self._get_default_beta(nx, ny, pixel_scale)

        return

    def get_basis_func(self, n, x, y, nx=None, ny=None, pixel_scale=None):
        '''
        Return the function call that will evaluate the nth basis
        function at (x,y) positions. Will account for pixel area noralization
        if the pixel_scale is passed

        n: int
            The indexed basis function to evaluate
        x: numpy.ndarray
            X positions to evaluate basis at, in units of beta
        y: numpy.ndarray
            Y positions to evaluate basis at, in units of beta
        nx: int; default None
            The number of pixels along the x-axis; only needed for PSF
            convolutions
        ny: int; default None
            The number of pixels along the y-axis; only needed for PSF
            convolutions
        pixel_scale: float; default None
            The pixel scale of the rendered imaged; only needed for PSF 
            convolutions
        '''

        if (n < 0) or (n >= self.N):
            raise ValueError(f'Basis functions range from 0 to {self.N-1}!')

        if (nx is None) or (ny is None) or (pixel_scale is None):
            if self.psf is not None:
                raise ValueError(
                    'nx, ny, and pixel_scale must be passed for basis funcs '
                    'with PSF convolutions!'
                    )

        # grab root basis function before PSF convolution
        func, func_args = self._get_basis_func(n)
        args = [x, y, *func_args]
        bfunc = func(*args)

        if self.psf is not None:
            bfunc = self.convolve_basis_func(bfunc, nx, ny, pixel_scale)

        return bfunc

    def render_im(
            self,
            coefficients,
            image_pars,
            plane='obs',
            transformation_pars=None,
            allow_default_beta=True
            ):
        '''
        Render image given transformation parameters and basis coefficients

        coefficients: list, np.array
            A list or array of basis coefficients
        image_pars: ImagePars
            An ImaegPars instance, including image shape and pixel scale
        plane: str; default 'obs'
            The image plane to render the image in. Must be one of the
            planes defined in the transformation module, e.g. obs, disk, etc.
        transformation_pars: dict; default None
            A dict that holds the sampled transformation parameters. If None,
            then cannot render the image in any plane other than `obs`. All
            scale parameters in the transformation_pars dict should be in
            units of beta, including x0 and y0 offsets, if applicable. All
            angles should be in radians.
        allow_default_beta: bool; default True
            If True, then the default beta value will be used if it is not
            set, given the image parameters. If False, then an error will be 
            raised
        '''

        nrow = image_pars.Nrow # numpy shape[0]
        ncol = image_pars.Ncol # numpy shape[1]
        nx = image_pars.Nx # numpy shape[1]
        ny = image_pars.Ny # numpy shape[0]
        pixel_scale = image_pars.pixel_scale

        if len(coefficients) != self.N:
            raise ValueError(
                f'The len of the passed coefficients does not equal {self.N}!'
            )

        if self.beta is None:
            beta_set = False # will get reset later
            if allow_default_beta is False:
                raise ValueError(
                    'beta must be set before rendering the image if ' +
                    'allow_default_beta is False!'
                    )
            self.beta = self._get_default_beta(nx, ny, pixel_scale)
        else:
            beta_set = True

        # set the pixel coordinate grid in the obs plane, and then
        # transform to the desired plane, if necessary
        Xobs, Yobs = utils.build_map_grid(nx, ny, indexing='xy')
        Xobs *= pixel_scale
        Yobs *= pixel_scale

        # NOTE/TODO: think about how to generalize this in the future!
        # adapt any offset parameters as well
        if transformation_pars is not None:
            for col in ['x0', 'y0']:
                if not col in transformation_pars.keys():
                    raise ValueError(
                        'transformation_pars must contain x0 and y0 keys!'
                        )
        else:
            transformation_pars = {}

        if plane == 'obs':
            X, Y = Xobs, Yobs
        else:
            X, Y = transform_coords(
                Xobs, Yobs, 'obs', plane, transformation_pars
                )

        if self.is_complex is True:
            im = np.zeros((nrow, ncol), dtype=np.complex128)
        else:
            im = np.zeros((nrow, ncol))

        for n in range(self.N):
            bfunc = self.get_basis_func(
                n, X, Y, nx=nx, ny=ny, pixel_scale=pixel_scale
                )
            im += coefficients[n] * bfunc

        if self.is_complex is True:
            im = im.real

        if beta_set is False:
            # reset beta to None for future image renderings
            self.beta = None

        return im

    def convolve_basis_func(self, bfunc, nx, ny, pixel_scale, method='auto'):
        '''
        Convolve a given basis vector/function by the stored PSF

        bfunc: np.ndarray
            A vector of a basis function evaluated on the pixel grid
        nx: int
            The number of pixels along the x-axis
        ny: int
            The number of pixels along the y-axis
        pixel_scale: float
            The pixel scale of the rendered image
        method: str; default 'auto'
            The method to use for convolution. If 'auto', then use the
            default method for galsim.drawImage(). For real PSF estimates with
            the pixel convolution already applied, use 'no_pixel'.
        '''

        if self.psf is None:
            return bfunc

        if self.is_complex:
            real = bfunc.real
            imag = bfunc.imag

            real_conv_b = self._convolve_basis_func(
                real, nx, ny, pixel_scale, method=method
                )
            imag_conv_b = self._convolve_basis_func(
                imag, nx, ny, pixel_scale, method=method
                )

            conv_b = real_conv_b + 1j*imag_conv_b

        else:
            conv_b = self._convolve_basis_func(
                bfunc, nx, ny, pixel_scale, method=method
                )

        return conv_b

    def _convolve_basis_func(self, bfunc, nx, ny, pixel_scale, method='auto'):
        '''
        Handle the conversion of a basis vector to a galsim interpolated image, 
        and then convolve by the psf

        bfunc: np.array (1D)
            A vector of a basis function evaluateon the pixel grid
        nx: int
            The number of pixels along the x-axis
        ny: int
            The number of pixels along the y-axis
        pixel_scale: float
            The pixel scale of the rendered image
        method: str; default 'auto'
            The method to use for convolution. If 'auto', then use the
            default method for galsim.drawImage(). For real PSF estimates with
            the pixel convolution already applied, use 'no_pixel'.dfk
        '''

        # the real/imag modes of some complex basis functions are fully zero 
        # across all samples; no need to bother convolving in this case
        if np.equal(bfunc, 0.0).all():
            return bfunc

        orig_shape = bfunc.shape

        if len(orig_shape) == 1:
            bfunc = bfunc.reshape(ny, nx, order='C')

        im = gs.Image(bfunc, scale=pixel_scale)

        im_gs = gs.InterpolatedImage(im)
        conv = gs.Convolve([self.psf, im_gs])
        conv_func = conv.drawImage(
            scale=pixel_scale, nx=nx, ny=ny, method=method
        ).array

        if len(orig_shape) == 1:
            conv_func = conv_func.reshape(nx*ny, order='C')

        return conv_func

    @abstractmethod
    def _get_basis_func(self, n):
        pass

class PolarBasis(Basis):
    '''
    Adds some functionality useful for all polar basis function classes.

    As it does not implement _get_basis_func(), it is still an abstract
    class.

    Parameters
    ----------
    name; str
        The name of the basis
    nmax: int; default None
        The maximum radial order of the basis functions. If None, then set
        automatically given image parameters during rendering.
    beta: float; default None
        Scale factor used to define basis functions, which effectively sets
        the units of the domain of the basis functions as well. If None, then
        set automatically given an sampled image of size (ncol,nrow)=(nx,ny) 
        whenever possible. NOTE: beta can always be changed after 
        initialization to satisfy different requirements.
    offset: tuple[float, float]; default None
        The offset of the basis from (0,0), in units of beta. If None, then
        the basis is centered at (0,0). NOTE: In most cases, you probably just
        want to apply an offset by using a transformation plane that is before 
        the offset later, such as `cen` or `disk`. Occasionally useful for
        testing purposes, but not recommended for general use.
    psf: galsim.GSObject
        A PSF model to convolve the basis by, if desired
    '''

    def __init__(self, *args, **kwargs):
        nmax = kwargs.pop('nmax', None)
        super().__init__(*args, **kwargs)

        # if None, can set automatically given image size once we know it during
        # image rendering
        if nmax is not None:
            if not isinstance(nmax, int):
                raise TypeError('nmax must be an int!')
            N = nmax**2
            self.nmax = nmax
            self._set_N(N)
            self._setup_lm_grid()
        else:
            self.nmax = None
            self.N = None

        return

    def _set_default_nmax(self, nx, ny, pixel_scale):
        '''
        Set nmax automatically given image parameters.

        nx: int
            The number of pixels along the x-axis
        ny: int
            The number of pixels along the y-axis
        pixel_scale: float
            The pixel scale of the rendered image
        '''

        theta_min, theta_max = self._get_scale_params(nx, ny, pixel_scale)

        self.nmax = int(
            np.ceil((theta_max / theta_min)) - 1
            )

        return

    def _setup_lm_grid(self):
        '''
        Setup the correspondence between a (l,m) pair and the associated
        basis function n
        '''

        if (not hasattr(self, 'nmax')) or (self.nmax is None):
            raise AttributeError(
                '`nmax` must be set before setting up the lm grid!'
                )

        lm_grid = {}

        k = 0
        for l in range(0, self.nmax):
            # l ranges from 0 to nmax-1
            for m in range(-l, l+1):
                # m ranges from -l to l
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

    def n_to_NxNy(self, n):
        '''
        Return the (Nx, Ny) pair corresponding to the nth
        basis function

        Used for mapping (l,m) pairs to plot grid
        '''

        l, m = self.n_to_lm(n)

        nx = l
        ny = l + m

        return nx, ny

    def render_im(
            self,
            coefficients,
            image_pars,
            plane='obs',
            transformation_pars=None,
            allow_default_beta=True
            ):
        '''
        See Basis.render_im() for more details. This method is overridden to
        handle the additional complexity of polar basis functions.
        '''

        if self.nmax is None:
            nmax_set = False
            # set nmax automatically given image parameters
            self._set_default_nmax(
                image_pars.Nx,
                image_pars.Ny,
                image_pars.pixel_scale
                )
        else:
            nmax_set = True

        basis_im = super().render_im(
            coefficients,
            image_pars,
            plane=plane,
            transformation_pars=transformation_pars,
            allow_default_beta=allow_default_beta
            )

        if nmax_set is False:
            # reset nmax to None for future renderings
            self.nmax = None

        return basis_im

    def plot_basis_funcs(
            self,
            image_pars,
            Nfuncs=None,
            outfile=None,
            show=True,
            size=(14,7)
            ):
        '''
        Plot the polar basis functions.
        
        image_pars: ImagePars
            An ImagePars instance, including image shape and pixel scale
        Nfuncs: int; default None
            The number of basis functions to plot. If None, then plot all basis 
            functions
        outfile: Path; default None
            The path to save the plot to. If None, then do not save the plot
        show: bool; default True
            Whether to show the plot or not
        size: tuple; default (14,7)
            The size of the plot
        '''

        if outfile is not None:
            if isinstance(outfile, str):
                outfile = Path(outfile)

        if Nfuncs is None:
            Nfuncs = self.N
        nmax = self.nmax
        sqrt = int(np.ceil(np.sqrt(Nfuncs)))

        X, Y = utils.build_map_grid(
            image_pars.Nx, image_pars.Ny, indexing='xy'
            )
        X *= image_pars.pixel_scale
        Y *= image_pars.pixel_scale

        for component in ['real', 'imag']:
            fig, axes = plt.subplots(
                nrows=nmax, ncols=2*(nmax-1)+1, figsize=size,
                sharex=True, sharey=True
                )
            for i in range(Nfuncs):
                l, m = self.n_to_lm(i)
                nx, ny = self.n_to_NxNy(i)
                ax = axes[nx, ny]
                image = self.get_basis_func(
                    i,
                    X,
                    Y,
                    nx=image_pars.Nx,
                    ny=image_pars.Ny,
                    pixel_scale=image_pars.pixel_scale
                    )

                try:
                    if component == 'real':
                        image = image.real
                    elif component == 'imag':
                        image = image.imag
                except:
                    pass

                im = ax.imshow(image, origin='lower')

                divider = make_axes_locatable(ax)
                cax = divider.append_axes('right', size='5%', pad=0.05)
                plt.colorbar(im, cax=cax)
                ax.set_title(f'({l},{m})')

            # remove empty axes
            for row in range(nmax):
                for col in range(2*(nmax-1)+1):
                    # if col > row and col < 2*(nmax-1)-row:
                    if col >= (2*row + 1):
                        fig.delaxes(axes[row, col])

            plt.tight_layout()
            plt.subplots_adjust(hspace=0.125, wspace=0.5)
            if self.psf is not None:
                pstr = ' with PSF'
            else:
                pstr = ''
            plt.suptitle(
                f'First {Nfuncs} {component} basis functions for {self.name}'
                f'{pstr}\nbeta={self.beta}, pixel_scale={image_pars.pixel_scale}',
                y=0.95
                )

            if outfile is not None:
                ext = outfile.suffix
                this_outfile = outfile.with_name(
                    outfile.name.replace(ext, f'-{component}{ext}')
                    )
                plt.savefig(this_outfile, bbox_inches='tight', dpi=300)

            if show is True:
                plt.show()
            else:
                plt.close()

        return

class SersicletBasis(PolarBasis):
    '''
    See https://arxiv.org/abs/1106.6045

    Parameters
    ----------
    index: float
        The Sersic index of the basis profiles.
    b: float; default None
        A normalizing sersic parameter. If None, then set automatically given 
        the index
    beta: float; default None
        Scale factor used to define basis functions, which effectively sets
        the units of the domain of the basis functions as well. If None, then
        set automatically given an sampled image of size (ncol,nrow)=(nx,ny) 
        whenever possible. NOTE: beta can always be changed after 
        initialization to satisfy different requirements.
    nmax: int; default None
        Max radial order of sersiclets. If None, then set automatically given 
        (nx, ny)
    offset: tuple[float, float]; default None
        The offset of the basis from (0,0), in units of beta. If None, then
        the basis is centered at (0,0). NOTE: In most cases, you probably just
        want to apply an offset by using a transformation plane that is before 
        the offset later, such as `cen` or `disk`. Occasionally useful for
        testing purposes, but not recommended for general use.
    psf: galsim.GSObject
        A PSF model to convolve the basis by, if desired
    '''

    is_complex = True

    def __init__(
            self,
            index,
            b=None,
            beta=None,
            nmax=None,
            offset=None,
            psf=None
            ):

        super(SersicletBasis, self).__init__(
            'sersiclet',
            nmax=nmax,
            beta=beta,
            offset=offset,
            psf=psf,
            )

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
            The array of r values to evaluate at, in units of beta
        phi: float, np.array
            The array of phi values to evaluate at, in radians
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

        norm_inner = ((beta**2 * n) / b**(2*n)) * (gamma(l + 2*n) / factorial(l))
        norm = 1. / np.sqrt(norm_inner)

        lag = genlaguerre(l, k)(u)

        exp = np.exp(-(b/2.) * (r/beta)**(1./n))

        rad = lag * exp

        ang = np.exp(-1j * m * phi)

        return norm * rad * ang


class ExpShapeletBasis(PolarBasis):
    '''
    See https://arxiv.org/abs/0809.3465,
        https://arxiv.org/abs/1106.6045,
        https://arxiv.org/abs/1903.05837

    Parameters
    ----------
    beta: float; default None
        Scale factor used to define basis functions, which effectively sets
        the units of the domain of the basis functions as well. If None, then
        set automatically given an sampled image of size (ncol,nrow)=(nx,ny) 
        whenever possible. NOTE: beta can always be changed after 
        initialization to satisfy different requirements.
    nmax: int; default None
        Max radial order of sersiclets. If None, then set automatically given 
        (nx, ny)
    offset: tuple[float, float]; default None
        The offset of the basis from (0,0), in units of beta. If None, then
        the basis is centered at (0,0). NOTE: In most cases, you probably just
        want to apply an offset by using a transformation plane that is before 
        the offset later, such as `cen` or `disk`. Occasionally useful for
        testing purposes, but not recommended for general use.
    psf: galsim.GSObject
        A PSF model to convolve the basis by, if desired
    '''

    is_complex = True

    def __init__(
            self, beta=None, nmax=None, offset=None, psf=None
            ):

        super().__init__(
            'exp_shapelets',
	        beta=beta,
	        nmax=nmax,
	        offset=offset,
	        psf=psf
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

        x: float, np.array
            The array of x values to evaluate at (converted to r, phi), in
            units of beta
        y: float, np.array
            The array of y values to evaluate at (converted to r, phi), in
            units of beta
        phi: float, np.array
            The array of phi values to evaluate at, in radians
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
    '''
    See https://arxiv.org/abs/astro-ph/0105178

    Parameters
    ----------
    beta: float; default None
        Scale factor used to define basis functions, which effectively sets
        the units of the domain of the basis functions as well. If None, then
        set automatically given an sampled image of size (ncol,nrow)=(nx,ny) 
        whenever possible. NOTE: beta can always be changed after 
        initialization to satisfy different requirements.
    nmax: int; default None
        Max radial order of sersiclets. If None, then set automatically given 
        (nx, ny)
    offset: tuple[float, float]; default None
        The offset of the basis from (0,0), in units of beta. If None, then
        the basis is centered at (0,0). NOTE: In most cases, you probably just
        want to apply an offset by using a transformation plane that is before 
        the offset later, such as `cen` or `disk`. Occasionally useful for
        testing purposes, but not recommended for general use.
    psf: galsim.GSObject
        A PSF model to convolve the basis by, if desired
    '''

    is_complex = False

    def __init__(self, beta=None, nmax=None, offset=None, psf=None):

        super(ShapeletBasis, self).__init__(
            'shapelet',
            beta=beta,
		    offset=offset,
		    psf=psf,
            )

        # compute number of independent basis vectors given nmax
        if nmax is not None:
            if not isinstance(nmax, int):
                raise TypeError('nmax must be an int!')
            if nmax < 0:
                raise ValueError('nmax must be non-negative!')
            N = (nmax+1) * (nmax+2) // 2 # triangular number
            self.nmax = nmax
            self._set_N(N)
            self._setup_nxny_grid()
        else:
            self.nmax = None
            self.N = None

        return

    def _get_default_nmax(self, nx, ny, pixel_scale):
        '''
        Determine a default nmax given image parameters

        Parameters
        ----------
        nx: int
            The number of pixels along the x-axis (*not* numpy shape[0]!)
        ny: int
            The number of pixels along the y-axis (*not* numpy shape[1]!)
        pixel_scale: float
            The pixel scale of the image, in the units of beta. Typically
            arcseconds per pixel.
        '''

        theta_min, theta_max = self._get_scale_params(nx, ny, pixel_scale)

        default_nmax = int(np.ceil((theta_max / theta_min)) - 1)

        return default_nmax

    def _setup_nxny_grid(self):
        '''
        We define here the mapping between the nth basis
        function and the corresponding (nx, ny pairs)

        We do this by creating the triangular matrix
        corresponding to nmax, ordering the positions
        such that each sub triangular matrix is consistent
        with one another
        '''

        nmax = self.nmax

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

        Parameters
        ----------
        n: int
            The number to check
        '''

        # this is equivalent to asking if m is square
        m = 8*n+1

        if np.sqrt(m) % 1 == 0:
            return True
        else:
            return False

    def _get_basis_func(self, n):
        '''
        Return function call along with args needed to evaluate nth basis 
        function at a given image position

        Parameters
        ----------
        n: int
            The indexed basis function to evaluate
        '''

        Nx, Ny = self.n_to_NxNy(n)

        args = [self.beta, Nx, Ny]

        return (self._eval_basis_function, args)

    def n_to_NxNy(self, n):
        '''
        Return the (Nx, Ny) pair corresponding to the nth basis function

        We define this relationship by traveling through the triangular (nx, 
        ny) grid in a zig-zag starting with x. Thus the ordering of each sub 
        triangular matrix will be consistent.
        '''

        Nx, Ny  = np.where(self.ngrid == n)

        return int(Nx[0]), int(Ny[0])

    def NxNy_to_n(self, Nx, Ny):
        '''
        Return the index of the basis function corresponding to the (Nx, Ny)
        pair

        Parameters
        ----------
        Nx: int
            The n-th eigenstate along the x-axis
        Ny: int
            Teh n-th eigenstate along the y-axis
        '''

        if (Nx+Ny) > self.nmax:
            raise ValueError(f'Nx+Ny cannot be greater than nmax={self.nmax}!')

        return int(self.ngrid[Nx, Ny])

    @staticmethod
    def _eval_basis_function(x, y, beta, Nx, Ny):
        '''
        Returns a single Cartesian Shapelet basis function of order
        Nx, Ny evaluated at the points (x,y).

        Parameters
        ----------
        x: float, np.array
            The array of x values to evaluate at, in units of beta
        y: float, np.array
            The array of y values to evaluate at, in units of beta
        beta: float
            The scale factor of the shapelets
        Nx: int
            The n-th eigenstate along the x-axis
        Ny: int
            The n-th eigenstate along the y-axis
        '''

        bfactor = (1. / np.sqrt(beta))
        x_norm = 1. / np.sqrt(2**Nx * np.sqrt(np.pi) * factorial(Nx)) * bfactor
        y_norm = 1. / np.sqrt(2**Ny * np.sqrt(np.pi) * factorial(Ny)) * bfactor

        exp_x = np.exp(-(x/beta)**2 / 2.)
        exp_y = np.exp(-(y/beta)**2 / 2.)
        phi_x = x_norm * eval_hermitenorm(Nx, x/beta) * exp_x
        phi_y = y_norm * eval_hermitenorm(Ny, y/beta) * exp_y

        return phi_x * phi_y

    def plot_basis_funcs(
            self,
            image_pars,
            Nfuncs=None,
            outfile=None,
            show=True,
            size=(9,9),
            hspace=0.15,
            wspace=0.5
            ):
        '''
        Plot the Shapelet basis functions.
        
        image_pars: ImagePars
            An ImagePars instance, including image shape and pixel scale
        Nfuncs: int; default None
            The number of basis functions to plot. If None, then plot all basis 
            functions
        outfile: Path; default None
            The path to save the plot to. If None, then do not save the plot
        show: bool; default True
            Whether to show the plot or not
        size: tuple; default (12,12)
            The size of the plot
        hspace: float; default 0.15
            The height space between subplots
        wspace: float; default 0.5
            The width space between subplots
        '''

        N = self.N
        nmax = self.nmax
        sqrt = int(np.ceil(np.sqrt(N)))

        X, Y = utils.build_map_grid(
            image_pars.Nx, image_pars.Ny, indexing='xy'
            )
        X *= image_pars.pixel_scale
        Y *= image_pars.pixel_scale


        if Nfuncs is None:
            Nfuncs = self.N

        fig, axes = plt.subplots(
            nrows=nmax+1, ncols=nmax+1, figsize=size, sharex=True, sharey=True
            )
        for i in range(Nfuncs):
            nx, ny = self.n_to_NxNy(i)
            ax = axes[nx, ny]
            image = self.get_basis_func(
                i,
                X,
                Y,
                image_pars.Nx,
                image_pars.Ny,
                pixel_scale=image_pars.pixel_scale
                )
            im = ax.imshow(image, origin='lower')

            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            plt.colorbar(im, cax=cax)
            ax.set_title(f'({nx},{ny})')

        x = np.arange(nmax+1)
        xx, yy = np.meshgrid(x, x)
        empty = np.where((xx+yy) > nmax)

        for x,y in zip(empty[0], empty[1]):
            fig.delaxes(axes[x,y])

        plt.tight_layout()
        plt.subplots_adjust(hspace=hspace, wspace=wspace)

        if self.psf is not None:
            pstr = ' with PSF'
        else:
            pstr = ''
        plt.suptitle(
            f'First {Nfuncs} basis functions for {self.name}'
            f'{pstr}\nbeta={self.beta} pixel_scaled={image_pars.pixel_scale}',
            y=0.1
            )

        if outfile is not None:
            plt.savefig(outfile, bbox_inches='tight', dpi=300)

        if show is True:
            plt.show()

        return

# TODO: Refactor this when ready
# def guess_beta(self, datacube, bmin=None, bmax=None, n=None, pixel_scale=None):
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
#     pixel_scale: float
#         The pixel scale of the datacube images (if not saved to the dc)
#     '''

#     if datacube.pixel_scale is None:
#         if pixel_scale is None:
#             raise ValueError('pixel_scale must be set either in the datacube ' +
#                              'if not explicitly!')
#         else:
#             if not isinstance(pixel_scale, (int,float)):
#                 raise TypeError('pixel_scale must be an int or float!')
#             if pixel_scale <=0:
#                 raise ValueError('pixel_scale must be positive!')
#     else:
#         if pixel_scale is not None:
#             if datacubepixel_scale != pixel_scale:
#                 raise ValueError('Passed pixel_scale not consistent with ' +
#                                  'datacube pixel_scale!')
#         pixel_scale = datacube.pixel_scale

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

def get_basis_types():
    return BASIS_TYPES

# NOTE: This is where you must register a new model
BASIS_TYPES = {
    'default': ShapeletBasis,
    'shapelets': ShapeletBasis,
    'sersiclets': SersicletBasis,
    # 'disclets': DiscletBasis, # one day maybe!
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