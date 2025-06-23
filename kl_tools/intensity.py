'''
This file contains a mix of IntensityMap classes for explicit definitions
(e.g. an inclined exponential) and IntensityMapFitter + Basis classes for
fitting a a chosen set of arbitrary basis functions to the 2D image
'''

import numpy as np
from abc import abstractmethod
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import galsim as gs
from galsim.angle import Angle, radians

import kl_tools.utils as utils
import kl_tools.basis as basis
import kl_tools.likelihood as likelihood
from kl_tools.transformation import transform_coords, SUPPORTED_PLANES

class IntensityMap(object):
    '''
    Class that handles the management and rendering of a 2D intensity image
    for a given set of model, transformation, and image parameters. Note that a 
    single imap instance can render the underlying imap model repeatedly given 
    different transformation and image parameters passed at rendering time.

    We make a distinction between the intensity coming from emission
    vs the continuum. Typically the image will depend on model
    parameters passed to render() while the continuum will be a static
    estimate passed here, but we allow for a callable function that takes
    as the image parameters so it can be computed on the fly.

    name: str
        Name of intensity map type
    continuum: np.ndarray | callable | Noen
        The continuum image to set, or a callable function
        that takes in ImagePars and returns a numpy array
        of the same shape as the image. If None, the continuum
        will be made to be a zero array of the same shape as the image
        set during the render() call
    '''

    def __init__(self, name, continuum=None):

        if not isinstance(name, str):
            raise TypeError('IntensityMap name must be a str!')
        self.name = name

        # while the "official" way to get an image is to call the
        # render() method, we also store the last rendered image
        # so that we can more efficiently handle static imaps to
        # avoid re-rendering them, as well as for some convenience
        # methods like plotting
        self._image = None
        self._continuum = None

        # the continuum image will be set to one of the following depending
        # on the input:
        # 1) np.ndarray: the continuum image is a static array and will error
        #    if you pass incompatible image parameters during render()
        # 2) callable: the continuum image is a function that takes in
        #    ImagePars and returns a numpy array of the same shape as the image
        # 3) None: the continuum image is not set, and will be made to be a zero
        #    array of the same shape as the image during the render() call
        self.set_continuum(continuum)

        # some intensity maps will not change per sample, but in general
        # they might
        self.is_static = False

        return

    def set_continuum(self, continuum):
        '''
        continuum: np.ndarray | callable | None
            The continuum image to set, or a callable function
            that takes in ImagePars and returns a numpy array
            of the same shape as the image. If None, the continuum
            will be made to be a zero array of the same shape as the image
            during the render() call
        '''

        if (continuum is None) or (isinstance(continuum, np.ndarray)):
            self._continuum_callable = False
        elif callable(continuum):
            # we assume the callable takes in ImagePars and returns
            # a numpy array of the same shape as the image
            self._continuum_callable = True
        else:
            raise TypeError('Continuum must be a numpy array or callable!')

        self.continuum = continuum

        return

    def render(
            self,
            image_pars,
            theta_pars,
            pars,
            weights=None,
            mask=None,
            image=None,
            redo=True,
            im_type='emission',
            raise_errors=True,
            ):
        '''
        Render an image of the emission line intensity

        image_pars: ImagePars
            The image parameters, including the shape & pixel scale
        theta_pars: dict
            A dict of the sampled mcmc params for both the velocity
            map and the tranformation matrices
        pars: dict
            A dictionary of any additional parameters needed
            to render the intensity map
        weights: np.ndarray; default None
            A 2D array of weights to apply to the image pixels, if needed
        mask: np.ndarray; default None
            A 2D array of bools to mask the image pixels, if needed. NOTE: we
            assume a convention where 0 is unmasked and all else is masked
        image: np.ndarray; default None
            The 2D image that is being modeled, if necessary. Most intensity 
            maps do not need this, but some may need to be fit to a specific
            image. If None, the image will be rendered from scratch using the
            image and theta parameters passed
        redo: bool; default True
            Set to remake rendered image regardless of whether it is already 
            internally stored
        im_type: str
            Can set to either `emission`, `continuum`, or `both`
        raise_errors: bool; default True
            If True, raise any errors that occur during the rendering
            process. If False, catch and print the errors instead, and pass
            None to the caller. This is useful for debugging or for models
            that have failure modes that need to be handled by the likelihood
            model

        return: np.ndarray | tuple | None
            The rendered intensity map (emission, continuum, or both). If 
            raise_errors is False and an error occurs, None is returned instead
        '''

        Nrow, Ncol = image_pars.Nrow, image_pars.Ncol

        # if weight and mask info was passed, make sure that they are consistent
        # with the image pars
        if weights is not None:
            if weights.shape != (Nrow, Ncol):
                raise ValueError('Weights shape must match image dimensions!')
        if mask is not None:
            if mask.shape != (Nrow, Ncol):
                raise ValueError('Mask shape must match image dimensions!')

        # the continuum has to be handled separately, as it is not guaranteed
        # to be set at this stage
        if im_type in ['continuum', 'both']:
            continuum = self.continuum

            if continuum is None:
                # if no continuum is set, we make it a zero array
                continuum = np.zeros((Nrow, Ncol), dtype=float)

            elif isinstance(continuum, np.ndarray):
                # check the shape of the continuum image
                if continuum.shape != (Nrow, Ncol):
                    raise ValueError(
                        'Stored continuum image shape does not match image_pars'
                        )
            elif callable(continuum):
                # call the function with the image parameters
                continuum = continuum(image_pars)

            if im_type == 'continuum':
                # we can skip the rest of the method
                # NOTE: this doesn't work for sample-specific continuum images,
                # but that case doesn't make sense with a continuum-only mode
                return continuum

        # now we handle the emission line intensity map

        # skip rendering only if:
        # 1) The imap type is static, and
        # 2) It has already computed, and
        # 3) We are not forcing a re-render
        if (self.is_static is False) or (self._image is None) or (redo is True):
            try:
                # NOTE: while rare, some render methods can produce a 
                # sample-specific continuum image. This is not the same as
                # the continuum template, which is static. So we track it here
                image, continuum = self._render(
                    image_pars,
                    theta_pars,
                    pars,
                    weights=weights,
                    mask=mask,
                    image=image
                    )
                # NOTE: We save the result to the image attribute so that static
                # imaps can be used without having to call render() again
                self._image = image
                self._continuum = continuum

            except Exception as e:
                if raise_errors is True:
                    raise e
                else:
                    print('Rendering failed!')
                    print(f'Error: {e}')
                    return None
        else:
            # under these conditions, we can simply load in the cached image 
            # from the last render
            image = self._image
            continuum = self._continuum

        if im_type == 'emission':
            # return only the emission line intensity image
            return image
        elif im_type == 'both':
            return image, continuum

    @abstractmethod
    def _render(
        self, image_pars, theta_pars, pars, weights=None, mask=None, image=None
        ):
        '''
        Each subclass should define how to render

        Most will need theta and pars, but not all
        '''
        pass

    def plot(self, image=None, show=True, close=True, outfile=None, size=(7,7)):
        if image is None:
            # see if an image was already rendered
            if self._image is not None:
                image = self._image
            else:
                raise Exception(
                    'Must pass an image or render first! This can be done by '
                    'calling render() with relevant params'
                    )

        ax = plt.gca()
        im = ax.imshow(image, origin='lower')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)
        ax.set_title('IntensityMap.render() call')

        plt.gcf().set_size_inches(size)
        plt.tight_layout()

        if outfile is not None:
            plt.savefig(outfile, bbox_inches='tight', dpi=300)

        if show is True:
            plt.show()

        if close is True:
            plt.close()

        return

class InclinedExponential(IntensityMap):
    '''
    This class is mostly for testing purposes. Can give the
    true flux, hlr for a simulated datacube to ensure perfect
    intensity map modeling for validation tests.

    We explicitly use an exponential over a general InclinedSersic
    as it is far more efficient to render, and is only used for
    testing anyway
    '''

    def __init__(self, flux, hlr=None, scale_radius=None, continuum=None):
        '''
        flux: float
            Object flux
        hlr: float
            Object half-light radius (in arcsec). Can only pass one of
            hlr or scale_radius, not both
        scale_radius: float; default None
            Object scale radius (in pixels). Can only pass one of hlr or
            scale_radius, not both
        continuum: np.ndarray | callable | None
            The continuum image to set, or a callable function
        '''

        super(InclinedExponential, self).__init__('inclined_exp')

        if hlr is None and scale_radius is None:
            raise ValueError(
                'Must pass either hlr or scale_radius to InclinedExponential!'
                )
        if hlr is not None and scale_radius is not None:
            raise ValueError(
                'Cannot pass both hlr and scale_radius to InclinedExponential!'
                )

        pars = {'flux': flux, 'hlr': hlr, 'scale_radius': scale_radius}
        for name, val in pars.items():
            if val is None:
                continue
            if not isinstance(val, (float, int)):
                raise TypeError(f'{name} must be a float or int!')

        self.flux = flux
        self.hlr = hlr
        self.scale_radius = scale_radius

        # same as default, but to make it explicit
        self.is_static = False

        return

    def _render(
        self, image_pars, theta_pars, pars, weights=None, mask=None, image=None
        ):
        '''
        image_pars: ImagePars
            The image parameters, including the shape & pixel scale
        theta_pars: dict
            A dict of the sampled mcmc params for both the velocity
            map and the tranformation matrices
        pars: dict
            A dictionary of any additional parameters needed
            to render the intensity map
        weights: np.ndarray; default None
            Generally not needed for inclined exponential imaps
        mask: np.ndarray; default None
            Generally not needed for inclined exponential imaps
        image: np.ndarray; default None
            The 2D image that is being modeled, if necessary. Most intensity 
            maps do not need this, but some may need to be fit to a specific
            image. If None, the image will be rendered from scratch using the
            image and theta parameters passed

        return: (np.ndarray, np.ndarray))
            The rendered intensity map and continuum imaged
        '''

        inc = Angle(np.arcsin(theta_pars['sini']), radians)

        if self.hlr is not None:
            gal = gs.InclinedExponential(
                inc, flux=self.flux, half_light_radius=self.hlr
            )
        else:
            gal = gs.InclinedExponential(
                inc, flux=self.flux, scale_radius=self.scale_radius
            )

        # Only add knots if a psf is provided
        # NOTE: no longer workds due to psf conovlution
        # happening later in modeling
        # if 'psf' in pars:
        #     if 'knots' in pars:
        #         knot_pars = pars['knots']
        #         knots = gs.RandomKnots(**knot_pars)
        #         gal = gal + knots

        rot_angle = Angle(theta_pars['theta_int'], radians)
        gal = gal.rotate(rot_angle)

        # TODO: still don't understand why this occasionally fails
        try:
            g1 = theta_pars['g1']
            g2 = theta_pars['g2']
            gal = gal.shear(g1=g1, g2=g2)
        except Exception as e:
            print('imap generation failed!')
            print(f'Shear values used: g=({g1}, {g2})')
            raise e

        if (pars is not None) and ('psf' in pars):
            psf = pars['psf']
            if psf is not None:
                if not isinstance(psf, gs.GSObject):
                    raise TypeError('PSF must be a galsim.GSObject!')
                gal = gs.Convolve([gal, psf])

        # now for image rendering
        # galsim uses (x,y) for Image-level coordinates
        Nx = image_pars.Nx
        Ny = image_pars.Ny
        pixel_scale = image_pars.pixel_scale

        # assumes all theta_par lengths are in arcsec
        offset = gs.PositionD(
            theta_pars['x0'] / pixel_scale,
            theta_pars['y0'] / pixel_scale
            )

        try:
            image = gal.drawImage(
                nx=Nx,
                ny=Ny,
                scale=pixel_scale,
                offset=offset,
                ).array
        except:
            # Can fail if no PSF & very inclined
            image = gal.drawImage(
                nx=Nx,
                ny=Ny,
                scale=pixel_scale,
                offset=offset,
                method='real'
                ).array

        # NOTE: no support for sample-dependent continuum images at this time
        continuum = self.continuum

        return image, continuum

    def plot_fit(self, data, fit=None, show=True, close=True, outfile=None,
                 size=(9,9), vmin=None, vmax=None):
        '''
        data: np.ndarray
            The 2D image to compare the fit to.
        fit: np.ndarray | None
            The 2D imap fit to compare to the data. If None, the
            latest imap render will be used, if possible
        '''

        if fit is None:
            if self._image is not None:
                fit = self._image
            else:
                raise Exception('Must fit with render() first!')

        fig, axes = plt.subplots(
            nrows=2, ncols=2, sharex=True, sharey=True, figsize=size
            )

        image = [data, fit, data-fit, 100.*(data-fit)/fit]
        titles = ['Data', 'Fit', 'Residual', '% Residual']

        for i in range(len(image)):
            ax = axes[i//2, i%2]
            im = ax.imshow(
                image[i], origin='lower', vmin=vmin, vmax=vmax
                )
            ax.set_title(titles[i])
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            plt.colorbar(im, cax=cax)

        plt.suptitle(f'Fit comparison for flux={self.flux}; hlr={self.hlr}')
        plt.tight_layout()

        if outfile is not None:
            plt.savefig(outfile, bbox_inches='tight', dpi=300)

        if show is True:
            plt.show()

        if close is True:
            plt.close()

        return

class BasisIntensityMap(IntensityMap):
    '''
    This is a catch-all class for Intensity Maps made by fitting arbitrary 
    basis functions to the stacked datacube image.

    basis_type: str
        Name of basis type to use
    basis_kwargs: dict
        Dictionary of kwargs needed to construct basis
    basis_plane: str
        The plane where the basis functions are defined. If 'obs', the
        basis functions are defined in the observed plane. If other (see
        transformation.py), the basis functions are defined in a plane
        other than the observed plane and so the fit is dependent on
        the transformation parameters in theta_pars passed to render().
    continuum: np.ndarray | callable | None
        The continuum image to set, or a callable function
        that takes in ImagePars and returns a numpy array
        of the same shape as the image. NOTE: In this subclass,
        the continuum is interpreted as a template, and will be
        fit to the image along with the chosen basis functions.
    '''

    def __init__(
            self,
            basis_type,
            basis_kwargs=None,
            basis_plane='obs',
            continuum=None
            ):

        super(BasisIntensityMap, self).__init__('basis', continuum=continuum)

        if basis_type not in basis.BASIS_TYPES.keys():
            raise ValueError(f'{basis_type} is not a registered basis type!')
        self.basis_type = basis_type
        self.basis_kwargs = basis_kwargs

        if basis_plane not in SUPPORTED_PLANES:
            raise ValueError(
                f'{basis_plane} is not a supported basis plane! Supported '
                f'planes are: {SUPPORTED_PLANES}'
            )
        self.basis_plane = basis_plane

        if 'psf' in basis_kwargs:
            # always default an explicitly passed psf
            self.psf = basis_kwargs['psf']
        else:
            self.psf = None

        self.basis_kwargs = basis_kwargs

        # cannot setup the fitter until we know the image parameters
        # such as shape and pixel scale, passed at render time
        self.fitter = None

        return

    def get_basis(self):
        if not self.fitter is None:
            return self.fitter.basis
        else:
            print('Cannot get the basis before initializing the fitter')
        return None

    @property
    def basis(self):
        return self.get_basis()

    @property
    def plane(self):
        return self.basis.plane

    def _render(
            self, image_pars, theta_pars, pars, weights=None, mask=None, image=None
            ):

        if image is None:
            raise ValueError(
                'For BasisIntensityMap, you must pass an image to render()'
            )

        Nrow, Ncol = image_pars.Nrow, image_pars.Ncol
        im_shape = image.shape

        if im_shape != (Nrow, Ncol):
            raise ValueError(
                f'Image shape {im_shape} must match image_pars dimensions ({Nrow, Ncol})'
            )

        self._setup_fitter(image_pars, weights=weights, mask=mask)

        # at this stage we now know whether the imap will change per sample
        if self.basis_plane == 'obs':
            self.is_static = True
        else:
            # any other plane for the basis definition will make the fitted
            # imap depend on the sample draw
            self.is_static = False

        image, continuum = self._fit_to_image(
            image, image_pars, theta_pars, pars
            )

        return image, continuum

    def _setup_fitter(self, image_pars, weights=None, mask=None):

        self.fitter = IntensityMapFitter(
            self.basis_type,
            self.basis_kwargs,
            self.basis_plane,
            image_pars,
            continuum_template=self.continuum,
            psf=self.psf,
            weights=weights,
            mask=mask,
            )

        return

    def _fit_to_image(self, image, image_pars, theta_pars, pars):

        image, continuum = self.fitter.fit(
            image, image_pars, theta_pars, pars
            )

        return image, continuum

def get_intensity_types():
    return INTENSITY_TYPES

# NOTE: This is where you must register a new model
INTENSITY_TYPES = {
    'default': BasisIntensityMap,
    'basis': BasisIntensityMap,
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

    basis_type: str
        The name of the basis_type type used
    basis_kwargs: dict
        Keyword args needed to build given basis type
    basis_plane: str
        The plane where the basis functions are defined. See transformation.py
        for the supported planes.
    image_pars: ImagePars
        The image parameters, including the shape & pixel scale
    continuum_template: numpy.ndarray
        A template array for the object continuum
    psf: galsim.GSObject
        A galsim object representing a PSF to convolve the
        basis functions by
    weights: numpy.ndarray
        A 2D array of floats to apply to the fitted stacked image
    mask: numpy.ndarray
        A 2D array of bools to mask the stacked image. NOTE: we assume a
        convention where 0 is unmasked and all else is masked
    '''

    def __init__(
            self,
            basis_type,
            basis_kwargs,
            basis_plane,
            image_pars, 
            continuum_template=None,
            psf=None,
            weights=None,
            mask=None
            ):

        self.basis_type = basis_type
        self.basis_plane = basis_plane
        self.basis_kwargs = basis_kwargs

        self.image_pars = image_pars
        Nrow, Ncol = image_pars.Nrow, image_pars.Ncol
        self.image_shape = (Nrow, Ncol)
        self.image_pixel_scale = image_pars.pixel_scale
        self.Ndata = Nrow * Ncol

        assert self.image_shape == image_pars.shape

        self.continuum_template = continuum_template

        for k, v in {'weights': weights, 'mask': mask}.items():
            if (v is not None) and (v.shape != (Nrow, Ncol)):
                raise ValueError(
                    f'{k} shape {v.shape} must match image shape {self.image_shape}'
                    )

        if weights is None:
            weights = np.ones(self.image_shape, dtype=float)
        if mask is None:
            mask = np.zeros(self.image_shape, dtype=bool)
        self.weights = weights
        self.mask = mask

        self.psf = psf

        # NOTE: The coordinate grid is stored as (x,y) in Cartesian coords,
        # not as (row,col) in numpy format due to the definition of the basis
        # functions in basis.py
        Nx, Ny = image_pars.Nx, image_pars.Ny
        Xpix, Ypix = utils.build_map_grid(Nx, Ny, indexing='xy')
        # account for pixel scale
        X = Xpix * self.image_pixel_scale
        Y = Ypix * self.image_pixel_scale
        self.grid = (X, Y)

        self._initialize_basis(basis_kwargs, image_pars)

        # will be set once transformation params and cov are passed
        self.design_mat = None
        self.pseudo_inv = None
        self.marginalize_det = None
        self.marginalize_det_log = None

        return

    def _initialize_basis(self, basis_kwargs, image_pars):
        if 'use_continuum_template' in basis_kwargs:
            basis_kwargs.pop('use_continuum_template')

        self.basis = basis.build_basis(self.basis_type, basis_kwargs)
        self.Nbasis = self.basis.N

        if self.continuum_template is not None:
            self.Nbasis += 1

        if self.basis.beta is None:
            # try to guess a good scale radius given the image parameters
            # NOTE: while the basis render_im() method can handle varying beta
            # per call, the design matrix cannot and so we set it here
            Nx = image_pars.Nx
            Ny = image_pars.Ny
            pixel_scale = image_pars.pixel_scale
            self.basis.set_default_beta(Nx, Ny, pixel_scale)

        self.mle_coefficients = np.zeros(self.Nbasis)

        return

    def _initialize_design_matrix(self, theta_pars):
        '''
        Setup the design matrix, whose cols are the (possibly transformed)
        basis functions at the obs pixel locations

        theta_pars: dict
            A dictionary of the sampled parameters, including
            transformation parameters
        '''

        Ndata = self.Ndata
        Nbasis = self.Nbasis

        # the plane where the basis is defined
        basis_plane = self.basis_plane

        # build image grid vectors in obs plane
        Xobs, Yobs = self.grid

        # find corresponding gird positions in the basis plane
        if basis_plane == 'obs':
            X, Y = Xobs, Yobs
        else:
            X, Y = transform_coords(
                Xobs, Yobs, 'obs', basis_plane, theta_pars
                )

        x = X.flatten(order='C')
        y = Y.flatten(order='C')

        # apply mask
        mask = self.mask.flatten(order='C')
        select = np.where(mask == 0)
        x = x[select]
        y = y[select]
        Npseudo = len(x)
        self.selection = select

        # the design matrix for a given basis and datacube
        if self.basis.is_complex is True:
            self.design_mat = np.zeros((Npseudo, Nbasis), dtype=np.complex128)
        else:
            self.design_mat = np.zeros((Npseudo, Nbasis))

        for n in range(self.basis.N):
            self.design_mat[:,n] = self.basis.get_basis_func(
                n,
                x,
                y,
                nx=self.image_pars.Nx,
                ny=self.image_pars.Ny,
                pixel_scale=self.image_pars.pixel_scale,
                )

        # handle continuum template separately
        if self.continuum_template is not None:
            template = self.continuum_template.flatten(order='C')[select]

            # make sure we aren't overriding anything
            assert (self.basis.N + 1) == self.Nbasis
            assert np.sum(self.design_mat[:,-1]) == 0
            self.design_mat[:,-1] = template

        # TODO: Undersand why there are nans!
        self.design_mat = np.nan_to_num(self.design_mat)

        return

    # TODO: Add @njit when ready
    def _initialize_pseudo_inv(self, theta_pars, max_fail=10, redo=True):
        '''
        Setup Moore-Penrose pseudo inverse given basis

        theta_pars: dict
            A dictionary of the sampled parameters, including
            transformation parameters
        max_fail: bool
            Maximum number of times it will try to re-compute the
            pseudo inverse if the SVD fails
        redo: bool
            Set to redo computation even if attribute already exists
        '''

        if (redo is True) or (self.design_mat is None):
            self._initialize_design_matrix(theta_pars)

        # now compute the pseudo-inverse:
        fail = 0
        while True:
            try:
                # NOTE: This can quietly fail in rare cases of near-zero
                # design matrices. We check for thix explicitly in the
                # likelihood call
                self.pseudo_inv = np.linalg.pinv(self.design_mat)
                break
            except np.linalg.LinAlgError as e:
                print('Warning: pseudo-inverse calculation failed')
                if fail > max_fail:
                    print('Trying again')
                else:
                    raise Exception('Pseudo-inverse failed after ' +\
                                    f'{max_fail+1} attempts\n' + str(e))

        return

    def compute_marginalization_det(
            self, inv_cov=None, pars=None, redo=True, log=False
            ):
        '''
        Compute the determinant factor needed to scale the marginalized
        posterior over intensity map basis function coefficients

        inv_cov: np.array
            The datacube inv_covariance matrix
            NOTE: For now, we assume it is constant across slices
        pars: dict
            A dictionary holding any additional parameters needed during
            likelihood evaluation
        redo: bool
            Set to recompute the determinant if already stored in memory
        log: bool
            Set to compute the log of the determinant instead
        '''

        if inv_cov is None:
            if pars is None:
                raise Exception('Must pass either an inv_cov matrix or pars dict!')
            else:
                # TODO: A bit hacky, but we are just using diagonal cov matrices
                # right now anyway
                sigma = pars['cov_sigma']
        else:
            # Won't use a constant sigma, instead will use full inv_cov matrix
            sigma = None

        if (redo is True) or (self.marginalize_det is None):
            phi = self.design_mat

            if inv_cov is None:
                M = (1./sigma**2) * phi.T.dot(phi)
            else:
                M = phi.T.dot(inv_cov.dot(phi))

            if log is False:
                det = np.linalg.det(M)
            else:
                # We don't care about the sign in this case
                sign, det = np.linalg.slogdet(M)

        self.marginalize_det = det
        self.marginalize_det_log = log

        return det

    def _initialize_fit(self, theta_pars):
        '''
        Setup needed quantities for finding MLE combination of basis
        funcs, along with marginalization factor from the determinant

        theta_pars: dict
            A dictionary of the sampled parameters, including
            transformation parameters
        '''

        # pseudo_inv calc will also initialize design matrix
        self._initialize_pseudo_inv(theta_pars, redo=True)
        # self._initialize_marginalization_det(theta_pars, pars)

        return

    def fit(self, image, image_pars, theta_pars, pars, cov=None):
        '''
        Fit MLE of the basis intensity map for a given 2D image

        NOTE: For 3D datacubes, this assumes that the datacube is truncated to 
        be only bearound the relevant emission line region, though not 
        necessarily with the continuum emission subtracted

        image: np.ndarray
            The stacked, 2D photometric image to fit to
        image_pars: ImagePars
            An ImagePars instance containing parameters such as image shape
        theta_pars: dict
            A dictionary of sampled parameters
        cov: np.ndarray
            The covariance matrix of the fitted image.
            NOTE: If the covariance matrix is of the form sigma*I for a
                  constant sigma, it does not contribute to the MLE. Thus
                  you should only pass cov if it is non-trivial
            TODO: This should be handled by pars in future
        '''

        if image.shape != self.image_shape:
            raise ValueError('Intensity map and image must have same shape!')

        if cov is None:
            # try looking for it in pars first
            try:
                cov = pars['cov']
            except (TypeError, KeyError):
                # if not found, keep it None
                pass
        else:
            if cov.shape != self.image_shape:
                raise ValueError('Cov matrix and image must have same shape!')

        # Initialize pseudo-inverse given the transformation parameters, plus
        # possibly other parameters
        self._initialize_fit(theta_pars)

        # Find MLE basis coefficients
        mle_coeff = self._fit_mle_coeff(image, cov=cov)

        assert len(mle_coeff) == self.Nbasis
        self.mle_coefficients = mle_coeff

        # Now create MLE intensity map
        if self.continuum_template is None:
            used_coeff = mle_coeff
            mle_continuum = None
        else:
            # the last mle coefficient is for the continuum template
            used_coeff = mle_coeff[:-1]
            mle_continuum = mle_coeff[-1] * self.continuum_template

            if self.basis.is_complex is True:
                mle_continuum = mle_continuum.real

        mle_im = self.basis.render_im(
            used_coeff,
            image_pars,
            plane=self.basis_plane,
            transformation_pars=theta_pars
            )

        assert mle_im.shape == image.shape
        self.mle_im = mle_im
        self.mle_continuum = mle_continuum

        return mle_im, mle_continuum

    def _fit_mle_coeff(self, image, cov=None):
        '''
        image: np.array
            The 2D image to fit to
        cov: np.array
            The 2D covariance matrix to use in the fit. NOTE: non-diagonal
            covariance matrices are not yet implemented!
        '''

        if cov is None:
            # The solution is simply the Moore-Penrose pseudo inverse
            # acting on the data vector
            image_flattened = image.flatten(order='C')
            mle_coeff = self.pseudo_inv.dot(image_flattened[self.selection])
            # import pudb; pudb.set_trace()

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
        data = datacube.stack()
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

class TransformedIntensityMapFitter(object):
    '''
    This class does the same thing as IntensityMapFitter,
    but with basis functions transformed according to
    currently sampled parameters
    '''

    def __init__(self, basis_type, nx, ny, basis_kwargs=None):
        '''
        transform_pars: dict
            A dictionary that holds at least the params needed
            to define the image transformations; g1, g2, theta_int,
            and sini
        '''

        super(TransformedIntensityMapFitter, self).__init__(
            basis_type, nx, ny, basis_kwargs=basis_kwargs
            )

        if not isinstance(transform_pars, dict):
            raise TypeError('transform_pars must be a dict!')
        self.transform_pars = transform_pars

        return


###############################################################################
# helper funcs for guessing best beta value from data

def fit_for_beta(datacube, basis, betas=None, Nbetas=100,
                 bmin=0.001, bmax=5, ncores=1):
    '''
    Scan over beta values for the best fit to the
    stacked datacube image for a given basis set

    datacube: DataCube
        The datacube to find the preferred beta scale for
    basis: basis.Basis subclass
        An instance of a Basis subclass with all optional args set
    betas: list, np.array
        A list or array of beta values to use in finding
        optimal value. Will create one if not passed
    Nbetas: int
        The number of beta values to scane over (if betas is not passed)
    bmin: float
        The minimum beta value to check (it betas is not passed)
    bmax: float
        The maximum beta value to check (it betas is not passed)
    ncores: int
        The number of cores to use for the beta scan

    returns: float
        The value of beta that minimizes the imap chi2
    '''

    if betas is None:
        betas = np.linspace(bmin, bmax, Nbetas)

    if ncores > 1:
        with Pool(ncores) as pool:
            out = stack(pool.starmap(fit_one_beta,
                                     [(i,
                                       beta,
                                       theta,
                                       datacube,
                                       pars,
                                       sky,
                                       vb
                                       )
                                      for i, beta in enumerate(betas)]
                                     )
                        )
    else:
        out = stack([fit_one_beta(
            i, beta, theta, datacube, pars_shapelet_b, pars_sersiclet_b,
            pars_exp_shapelet_b, sky, vb
            ) for i, beta in enumerate(betas)])

    return

def fit_one_beta(i, beta, theta_pars, datacube, basis, sky=1., vb=False):
    '''
    TODO: Incorporate full covariance matrix instead of sky sigma
    TODO: Docstring
    '''

    if (vb is True) and (i % 10 == 0):
        print(f'Fitting beta {i}')

    pars = basis.pars

    # TODO
    imap = likelihood.DataCubeLikelihood._setup_imap(
        theta_pars, datacube, pars
    )
    basis_im = imap.render(
        theta_pars, datacube, pars
                )
    mle = imap.fitter.mle_coefficients

    data = datacube.stack()
    Npix = data.shape[0] * data.shape[1]

    chi2_shapelet = np.sum((basis_im-data)**2/sky**2) / Npix

    return np.array([chi2_shapelet, chi2_sersiclet, chi2_exp_shapelet])
