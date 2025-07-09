import numpy as np
import os
from time import time
from abc import abstractmethod
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from argparse import ArgumentParser
import galsim as gs
from galsim.angle import Angle, radians

import kl_tools.utils as utils
import kl_tools.basis as basis
import kl_tools.likelihood as likelihood
import kl_tools.parameters as parameters
from kl_tools.emission import LINE_LAMBDAS
from kl_tools.transformation import transform_coords, TransformableImage

import ipdb

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

    def __init__(self, name, nx, ny):
        '''
        name: str
            Name of intensity map type
        nx: int
            Size of image on x-axis
        ny: int
            Size of image on y-axis
        '''

        if not isinstance(name, str):
            raise TypeError('IntensityMap name must be a str!')
        self.name = name

        for n in [nx, ny]:
            if not isinstance(n, int):
                raise TypeError('IntensityMap image size params must be ints!')
        self.nx = nx
        self.ny = ny

        # we make a distinction between the intensity coming from emissoin
        # vs the continuum
        self.image = None
        self.continuum = None
        self.gal = None

        # some intensity maps will not change per sample, but in general
        # they might
        self.is_static = False

        return

    def render(self, theta_pars, datacube, pars, redo=False,
               im_type='emission'):
        '''
        Render an image of the emission line intensity

        theta_pars: dict
            A dict of the sampled mcmc params for both the velocity
            map and the tranformation matrices
        pars: dict
            A dictionary of any additional parameters needed
            to render the intensity map
        redo: bool
            Set to remake rendered image regardless of whether
            it is already internally stored
        im_type: str
            Can set to either `emission`, `continuum`, or `both`

        return: np.ndarray, tuple
            The rendered intensity map (emission, continuum, or both)
        '''

        # only render if it has not been computed yet, or if
        # explicitly asked
        if (self.image is None) or (redo is True):
            self._render(theta_pars, datacube, pars)

        _RG_ = pars.get("run_options", {}).get("imap_return_gal", False)
        if im_type == 'emission':
            return self.image if not _RG_ else self.image, self.gal
        elif im_type == 'continuum':
            return self.continuum if not _RG_ else self.continuum, self.gal
        elif im_type == 'both':
            return self.image, self.continuum if not _RG_ else self.image, self.continuum, self.gal
        else:
            raise ValueError('im_type can only be one of ' +\
                             'emission, continuum, or both!')

    @abstractmethod
    def _render(self, theta_pars, datacube, pars):
        '''
        Each subclass should define how to render

        Most will need theta and pars, but not all
        '''
        pass

    def plot(self, show=True, close=True, outfile=None, size=(7,7)):
        if self.image is None:
            raise Exception('Must render profile first! This can be ' +\
                            'done by calling render() with relevant params')

        ax = plt.gca()
        im = ax.imshow(self.image, origin='lower')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)
        ax.set_title('BasisIntensityMap.render() call')

        plt.gcf().set_size_inches(size)
        plt.tight_layout()

        if outfile is not None:
            plt.savefig(outfile, bbox_inches='tight', dpi=300)

        if show is True:
            plt.show()

        if close is True:
            plt.close()

        return

class GMixModel(IntensityMap):
    '''
    This class use the mixture of Gaussian to fit inclined exponential or
    inclined Sersic profile, see NGMIX (Sheldon 2014) and Hogg & Lang (2012)
    **Methodology:**
    - [components]: The model has two components: emission line (exponential
    disk) and continuum (de Vaucouleurs, n=4 Sersic). The two components can
    have different flux and half-light radius, but the inclination and posi-
    tion angle are the same.
    - [projection]: To translate between inclination and ellipticity, we use
    the description in Cappellari (2002) (Eqn 9, the oblate axisymmetric),
        (q')^2 = (q*sin(i))*2 + cos^2(i)
    In this version we fix q to some reasonable value (0.1). But since the q
    of galactic bulge and galactic disk are not necessarily the same, we might
    release q for sampling with some prior in future.
    - [shear transform]: Gaussian profile is so simple that it has an analy-
    tical solution under shear transform. Eqn 2.11 and 2.12 in Bernstein &
    Jarvis (2002) show how the ellipticity of sheared image relates with shear
    and the intrinsic ellipticity
    '''
    def __init__(self, datavector, has_continuum = True,
        theory_Nx = None, theory_Ny = None, scale = None):
        pass

    def _render(self, theta_pars, datacube, pars):
        '''
        theta_pars: dict
            A dict of the sampled mcmc params for both the velocity
            map and the tranformation matrices
        datavector: DataCube
            Truncated data cube of emission line
        pars: dict
            A dictionary of any additional parameters needed
            to render the intensity map

        return: np.ndarray
            The rendered intensity map
        '''
        # update meta parameters & retrieve parameters value
        meta_updated = pars.meta.copy_with_sampled_pars(theta_pars)
        hlr = meta_updated['intensity']['hlr']
        hlr_cont = meta_updated['intensity']['hlr_cont']
        sini, theta_int = meta_updated['sini'], meta_updated['theta_int']
        g1, g2 = meta_updated['g1'], meta_updated['g2']
        inc = Angle(np.arcsin(sini), radians)
        qz = 0.1
        qobs = np.sqrt(1-(1-qz**2)*sini**2)
        # build inclined MoGs for both the emission line and stellar continuum
        # the shear is applied via Eqn 2.12 in Bernstein & Jarvis 2002


        # Only add knots if a psf is provided
        # NOTE: no longer workds due to psf conovlution
        # happening later in modeling
        # if 'psf' in pars:
        #     if 'knots' in pars:
        #         knot_pars = pars['knots']
        #         knots = gs.RandomKnots(**knot_pars)
        #         gal = gal + knots

        rot_angle = Angle(theta_int, radians)
        gal = gal.rotate(rot_angle)

        # TODO: still don't understand why this sometimes randomly fails
        try:
            gal = gal.shear(g1=g1, g2=g2)
        except Exception as e:
            print('imap generation failed!')
            print(f'Shear values used: g=({g1}, {g2})')
            raise e
        self.gal = gal

        self.image = gal.drawImage(
            nx=self.nx, ny=self.ny, scale=self.pix_scale
            ).array
        if pars.get('run_options', {}).get('imap_return_gal', False):
            return self.image, self.gal
        else:
            return self.image




class InclinedExponential(IntensityMap):
    '''
    This class is mostly for testing purposes. Can give the
    true flux, hlr for a simulated datacube to ensure perfect
    intensity map modeling for validation tests.

    We explicitly use an exponential over a general InclinedSersic
    as it is far more efficient to render, and is only used for
    testing anyway
    '''

    def __init__(self, datavector, kwargs):
        ''' Initialize geometry spec and flux of InclinedExponentioal profile
        Note that the `datavector` argument is only used to extract image di-
        mension (Nx, Ny, and pixel scale), which can be replaced by kwargs.
        Inputs:
        =======
        datavector: `cube.DataCube` object
            While this implementation will not use the datacube image expli-
            citly (other than shape info), general intensity generation will,
            like shapelet method relies on some specific intensity profile to
            get reasonably good guess.
            For `InclinedExponential`, you can pass `datavector = None` then
            pass the shape information by kwargs:
                theory_Nx = blahblah, theory_Ny = blahblah, scale = blahblah
        kwargs: dict
            optional keyword arguments, including
            flux, hlr, theory_Nx, theory_Ny, scale, em_PaA_hlr, etc
            - flux: float
                Object flux
            - hlr: float
                Object half-light radius (in pixels)

        '''
        ### Setting the intensity profile shape specification
        if kwargs.get("theory_Nx", None) is not None and kwargs.get("theory_Ny", None) is not None:
            nx, ny = kwargs["theory_Nx"], kwargs["theory_Ny"]
        else:
            nx, ny = datavector.Nx, datavector.Ny
        super(InclinedExponential, self).__init__('inclined_exp', nx, ny)
        self.pix_scale = kwargs.get("scale", None)
        if (self.pix_scale is None) and (datavector is not None):
            self.pix_scale = datavector.pix_scale

        ### Setting the intensity profile astrophysical info
        self.pars = {'flux': kwargs.get("flux", 1), 'hlr': kwargs.get("hlr", None), 'imap_return_gal': kwargs.get("imap_return_gal", False)}
        ### Setting the intensity profile of emission lines
        for emline,vacwave in LINE_LAMBDAS.items():
            if kwargs.get(f'em_{emline}_hlr', None) is not None:
                self.pars[f'em_{emline}_hlr'] = kwargs[f'em_{emline}_hlr']
        for name, val in self.pars.items():
            if not isinstance(val, (float, int)):
                raise TypeError(f'{name} must be a float or int!')
        #self.flux = flux
        #self.hlr = hlr
        # same as default, but to make it explicit
        self.is_static = False

        return

    def _render(self, theta_pars, datavector, pars):
        '''
        theta_pars: dict
            A dict of the sampled mcmc params for both the velocity
            map and the tranformation matrices
        datavector: DataCube
            Truncated data cube of emission line
        pars: dict
            A dictionary of any additional parameters needed
            to render the intensity map

        return: np.ndarray
            The rendered intensity map
        '''
        # pars has higher priority than theta_pas, such that when
        # we want to fix parameters of g1,g2,sini,theta_int, we can
        # set key:fixed_val in the pars.
        #print(theta_pars)
        #print(pars["intensity"])
        #if 'sini' in pars:
        #    sini = pars["sini"]
        #else:
        #    sini = theta_pars['sini']
        #if "g1" in pars:
        #    g1 = pars["g1"]
        #else:
        #    g1 = theta_pars['g1']
        #if "g2" in pars:
        #    g2 = pars["g1"]
        #else:
        #    g2 = theta_pars["g2"]
        #if "theta_int" in pars:
        #    theta_int = pars["theta_int"]
        #else:
        #    theta_int = theta_pars["theta_int"]
        #sini = pars.get('sini', theta_pars['sini'])
        #g1 = pars.get('g1', theta_pars['g1'])
        #g2 = pars.get('g2', theta_pars['g2'])
        #theta_int = pars.get('theta_int', theta_pars['theta_int'])
        #dx_disk = pars["intensity"].get("dx_disk", theta_pars['dx_disk']) * self.pars["hlr"]
        #dy_disk = pars["intensity"].get("dy_disk", theta_pars['dy_disk']) * self.pars["hlr"]
        #dx_spec = pars["intensity"].get("dx_spec", theta_pars['dx_spec']) * self.pars["hlr"]
        #dy_spec = pars["intensity"].get("dy_spec", theta_pars['dy_spec']) * self.pars["hlr"]
        sini = pars['sini']
        g1 = pars['g1']
        g2 = pars['g2']
        theta_int = pars['theta_int']
        dx_disk = pars["intensity"]["dx_disk"] * self.pars["hlr"]
        dy_disk = pars["intensity"]["dy_disk"] * self.pars["hlr"]
        dx_spec = pars["intensity"]["dx_spec"] * self.pars["hlr"]
        dy_spec = pars["intensity"]["dy_spec"] * self.pars["hlr"]


        inc = Angle(np.arcsin(sini), radians)
        rot_angle = Angle(theta_int, radians)

        self.gal = {}
        self.image = {}
        # photometry image intensity profile
        start = time()*1000
        self.gal["phot"] = gs.InclinedExponential(
            inc, flux=self.pars["flux"], half_light_radius=self.pars["hlr"]
        ).rotate(rot_angle).shear(g1=g1, g2=g2).shift(dx_disk, dy_disk)
        #print(self.gal)
        try:
            self.image["phot"] = self.gal["phot"].drawImage(nx=self.nx, ny=self.ny,
                scale=self.pix_scale).array
        except gs.GalSimFFTSizeError:
            self.image["phot"] = np.zeros([self.ny, self.nx])
        # print('\t\t--- photometry profile | %.2f seconds'%(time()*1000-start))
        # emission lines + continuum intensity profile
        for emline,vacwave in LINE_LAMBDAS.items():
            if f'em_{emline}_hlr' in pars['intensity']:
                eml_hlr = pars['intensity'][f'em_{emline}_hlr']
                self.gal[f'em_{emline}'] = gs.InclinedExponential(
                    inc, flux=1, half_light_radius=eml_hlr).rotate(rot_angle).shear(g1=g1, g2=g2).shift(dx_spec,dy_spec)
                try:
                    self.image[f'em_{emline}'] = self.gal[f'em_{emline}'].drawImage(
                        nx=self.nx, ny=self.ny, scale=self.pix_scale).array
                except gs.GalSimFFTSizeError:
                    self.image[f'em_{emline}'] = np.zeros([self.ny, self.nx])
                #print(self.gal)
                # print('\t\t--- emission profile | %.2f seconds'%(time()*1000-start))
            if f'cont_{emline}_hlr' in pars['intensity']:
                self.gal[f'cont_{emline}'] = gs.InclinedSersic(4, inc, half_light_radius=pars['intensity'][f'cont_{emline}_hlr'], flux=1.0, trunc=5*pars['intensity'][f'cont_{emline}_hlr'], flux_untruncated=True)#.rotate(rot_angle).shear(g1=g1, g2=g2)
                try:
                    self.image[f'cont_{emline}'] = self.gal[f'cont_{emline}'].drawImage(
                        nx=self.nx, ny=self.ny, scale=self.pix_scale).array
                except gs.GalSimFFTSizeError:
                    self.image[f'cont_{emline}'] = np.zeros([self.ny, self.nx])

        # Only add knots if a psf is provided
        # NOTE: no longer workds due to psf conovlution
        # happening later in modeling
        # if 'psf' in pars:
        #     if 'knots' in pars:
        #         knot_pars = pars['knots']
        #         knots = gs.RandomKnots(**knot_pars)
        #         gal = gal + knots

        # try:
        #     self.image = gal.drawImage(nx=self.nx, ny=self.ny,
        #         scale=self.pix_scale).array
        # except gs.GalSimFFTSizeError:
        #     #print(f'WARNING: FFT size too large, return -np.inf')
        #     self.image = np.zeros([self.ny, self.nx])
        if pars['run_options']['imap_return_gal']:
            return self.image, self.gal
        else:
            return self.image

    def plot_fit(self, datacube, show=True, close=True, outfile=None,
                 size=(9,9), vmin=None, vmax=None):
        '''
        datacube: DataCube
            Datacube that MLE fit was done on
        '''

        data = datacube.stack()

        if self.image is None:
            raise Exception('Must fit with render() first!')
        fit = self.image

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
    This is a catch-all class for Intensity Maps made by
    fitting arbitrary basis functions to the stacked
    datacube image

    TODO: Need a better name!
    '''

    def __init__(self, datacube, basis_type='default', basis_kwargs=None):
        '''
        basis_type: str
            Name of basis type to use
        datacube: DataCube
            A truncated datacube whose stacked slices will be fit to
        basis_kwargs: dict
            Dictionary of kwargs needed to construct basis
        '''

        nx, ny = datacube.Nx, datacube.Ny
        super(BasisIntensityMap, self).__init__('basis', nx, ny)

        if 'pix_scale' not in basis_kwargs:
            if datacube.pix_scale is None:
                raise Exception('Either datacube or basis_kwargs ' +
                                'must have pix_scale!')
            basis_kwargs['pix_scale'] = datacube.pix_scale

        # TODO: would be nice to generalize for chromatic
        # PSFs in the future!
        if 'psf' in basis_kwargs:
            # always default an explicitly passed psf
            self.psf = basis_kwargs['psf']
        else:
            # should return None if no PSF is stored
            # in datacube pars
            self.psf = datacube.get_psf()

        # One way to handle the emission line continuum is to build
        # a template function from the datacube
        try:
            use_cont = basis_kwargs['use_continuum_template']
            if use_cont is True:
                self.continuum_template = datacube.get_continuum()
                if self.continuum_template is None:
                    raise AttributeError('Datacube continnuum template is None!')
            else:
                self.continuum_template = None

        except KeyError:
            self.continuum_template = None

        # often useful to have a correct basis function scale given
        # stacked datacube image
        # if basis_type == 'shapelets':
        # TODO: Testing!!
        if False:
            pixscale = basis_kwargs['pix_scale']
            am = gs.hsm.FindAdaptiveMom(
                gs.Image(datacube.stack(), scale=pixscale)
                )
            self.am_sigma = am.moments_sigma
            basis_kwargs['beta'] = self.am_sigma
        else:
            self.am_sigma = None

        self.basis_kwargs = basis_kwargs

        self._setup_fitter(basis_type, nx, ny, basis_kwargs=basis_kwargs)

        # at this stage we now know whether the imap will change per sample
        if self.fitter.basis.plane == 'obs':
            self.is_static = True
        else:
            # any other plane for the basis definition will make the fitted
            # imap depend on the sample draw
            self.is_static = False

        self.image = None

        return

    def _setup_fitter(self, basis_type, nx, ny, basis_kwargs=None):

        self.fitter = IntensityMapFitter(
            basis_type, nx, ny,
            continuum_template=self.continuum_template,
            psf=self.psf,
            basis_kwargs=basis_kwargs
            )

        return

    def _fit_to_datacube(self, theta_pars, datacube, pars):
        try:
            if pars['run_options']['remove_continuum'] is True:
                if self.continuum_template is None:
                    print('WANRING: cannot remove continuum as a ' +\
                          'template was not provided')
                    remove_continuum = False
                else:
                    remove_continuum = True
            else:
                remove_continuum = False

        except KeyError:
            remove_continuum = False

        self.image, self.continuum = self.fitter.fit(
            theta_pars, datacube, pars, remove_continuum=remove_continuum
            )

        return

    def get_basis(self):
        return self.fitter.basis

    def render(self, theta_pars, datacube, pars, redo=False,
               im_type='emission'):
        '''
        see IntensityMap.render()
        '''

        return super(BasisIntensityMap, self).render(
            theta_pars, datacube, pars, redo=redo, im_type=im_type
            )

    def _render(self, theta_pars, datacube, pars):

        self._fit_to_datacube(theta_pars, datacube, pars)

        return

def get_intensity_types():
    return INTENSITY_TYPES

# NOTE: This is where you must register a new model
INTENSITY_TYPES = {
    'default': BasisIntensityMap,
    'basis': BasisIntensityMap,
    'inclined_exp': InclinedExponential,
    'mogs': GMixModel,
    }

def build_intensity_map(name, datavector, kwargs):
    '''
    name: str
        Name of intensity map type
    datavector: any class that subclass from DataVector
        The datavector whose stacked image the intensity map
        will represent
    kwargs: dict
        Keyword args to pass to intensity constructor
    '''

    name = name.lower()

    if name in INTENSITY_TYPES.keys():
        # User-defined input construction
        intensity = INTENSITY_TYPES[name](datavector, kwargs)
    else:
        raise ValueError(f'{name} is not a registered intensity!')

    return intensity

class IntensityMapFitter(object):
    '''
    This base class represents an intensity map defined
    by some set of basis functions {phi_i}.
    '''
    def __init__(self, basis_type, nx, ny, continuum_template=None,
                 psf=None, basis_kwargs=None):
        '''
        basis_type: str
            The name of the basis_type type used
        nx: int
            The number of pixels in the x-axis
        ny: int
            The number of pixels in the y-ayis
        continuum_template: numpy.ndarray
            A template array for the object continuum
        psf: galsim.GSObject
            A galsim object representing a PSF to convolve the
            basis functions by
        basis_kwargs: dict
            Keyword args needed to build given basis type
        '''

        for name, n in {'nx':nx, 'ny':ny}.items():
            if name in basis_kwargs:
                if n != basis_kwargs[name]:
                    raise ValueError(f'{name} must be consistent if ' +\
                                       'also passed in basis_kwargs!')

        self.basis_type = basis_type
        self.nx = nx
        self.ny = ny

        self.continuum_template = continuum_template
        self.psf = psf

        self.grid = utils.build_map_grid(nx, ny)

        self._initialize_basis(basis_kwargs)

        # will be set once transformation params and cov
        # are passed
        self.design_mat = None
        self.pseudo_inv = None
        self.marginalize_det = None
        self.marginalize_det_log = None

        return

    def _initialize_basis(self, basis_kwargs):
        if basis_kwargs is not None:
            basis_kwargs['nx'] = self.nx
            basis_kwargs['ny'] = self.ny
        else:
            # TODO: do we really want this to be the default?
            basis_kwargs = {
                'nx': self.nx,
                'ny': self.ny,
                'plane': 'obs',
                'pix_scale': self.pix_scale
                }

        if 'use_continuum_template' in basis_kwargs:
            basis_kwargs.pop('use_continuum_template')

        self.basis = basis.build_basis(self.basis_type, basis_kwargs)
        self.Nbasis = self.basis.N

        if self.continuum_template is not None:
            self.Nbasis += 1

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

        Ndata = self.nx * self.ny
        Nbasis = self.Nbasis

        # the plane where the basis is defined
        basis_plane = self.basis.plane

        # build image grid vectors in obs plane
        Xobs, Yobs = self.grid

        # find corresponding gird positions in the basis plane
        X, Y = transform_coords(
            Xobs, Yobs, 'obs', basis_plane, theta_pars
            )

        x = X.reshape(Ndata)
        y = Y.reshape(Ndata)

        # the design matrix for a given basis and datacube
        if self.basis.is_complex is True:
            self.design_mat = np.zeros((Ndata, Nbasis), dtype=np.complex128)
        else:
            self.design_mat = np.zeros((Ndata, Nbasis))

        for n in range(self.basis.N):
            self.design_mat[:,n] = self.basis.get_basis_func(n, x, y)

        # handle continuum template separately
        if self.continuum_template is not None:
            template = self.continuum_template.reshape(Ndata)

            # make sure we aren't overriding anything
            assert (self.basis.N + 1) == self.Nbasis
            assert np.sum(self.design_mat[:,-1]) == 0
            self.design_mat[:,-1] = template

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

    def compute_marginalization_det(self, inv_cov=None, pars=None, redo=True,
                                    log=False):
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

    def _initialize_fit(theta_pars, pars):
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

    def fit(self, theta_pars, datacube, pars, cov=None, remove_continuum=True):
        '''
        Fit MLE of the intensity map for a given set of datacube
        slices

        NOTE: This assumes that the datacube is truncated to be only be
              around the relevant emission line region, though not
              necessarily with the continuum emission subtracted

        theta: dict
            A dictionary of sampled parameters
        datacube: DataCube
            The truncated datacube around an emission line
        pars: dict
            A dictionary holding any additional parameters needed during
            likelihood evaluation
        cov: np.ndarray
            The covariance matrix of the datacube images.
            NOTE: If the covariance matrix is of the form sigma*I for a
                  constant sigma, it does not contribute to the MLE. Thus
                  you should only pass cov if it is non-trivial
            TODO: This should be handled by pars in future
        remove_continuum: bool
            If true, remove continuum template if fitted
        '''

        nx, ny = self.nx, self.ny
        if (datacube.Nx, datacube.Ny) != (nx, ny):
            raise ValueError('DataCube must have same dimensions ' +\
                             'as intensity map!')

        # Initialize pseudo-inverse given the transformation parameters
        self._initialize_pseudo_inv(theta_pars)

        data = datacube.stack().reshape(nx*ny)

        # Find MLE basis coefficients
        mle_coeff = self._fit_mle_coeff(data, cov=cov)

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

        mle_im = self.basis.render_im(theta_pars, used_coeff)

        assert mle_im.shape == (nx, ny)
        self.mle_im = mle_im
        self.mle_continuum = mle_continuum

        return mle_im, mle_continuum

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

def fit_for_beta(datacube, basis_type, betas=None, Nbetas=100,
                 bmin=0.001, bmax=5):
    '''
    TODO: Finish!
    Scan over beta values for the best fit to the
    stacked datacube image

    datacube: DataCube
        The datacube to find the preferred beta scale for
    basis_type: str
        The type of basis to use for fitting for beta
    betas: list, np.array
        A list or array of beta values to use in finding
        optimal value. Will create one if not passed

    returns: float
        The value of beta that minimizes the imap chi2
    '''

    if betas is None:
        betas = np.linspace(bmin, bmax, Nbetas)

    for beta in betas:
        pass

    return

def main(args):
    '''
    For now, just used for testing the classes
    '''

    from mocks import setup_likelihood_test
    from astropy.units import Unit

    show = args.show

    outdir = os.path.join(utils.TEST_DIR, 'intensity')
    utils.make_dir(outdir)

    print('Creating IntensityMapFitter w/ shapelet basis')
    nmax = 12
    nx, ny = 30,30
    pix_scale = 0.1 # arcsec / pixel
    basis_kwargs = {'Nmax': nmax, 'pix_scale': pix_scale, 'plane':'obs'}
    fitter = IntensityMapFitter(
        'shapelets', nx, ny, basis_kwargs=basis_kwargs
        )

    print('Creating IntensityMapFitter w/ transformed shapelet basis')
    basis_kwargs = {'Nmax': nmax, 'plane':'disk', 'pix_scale':1}
    fitter_transform = IntensityMapFitter(
        'shapelets', nx, ny, basis_kwargs=basis_kwargs
        )

    print('Setting up test datacube and true Halpha image')
    true_pars = {
        'g1': 0.025,
        'g2': -0.0125,
        'theta_int': np.pi / 6,
        'sini': 0.7,
        'v0': 5,
        'vcirc': 200,
        'rscale': 3,
    }

    mcmc_pars = {
        'units': {
            'v_unit': Unit('km / s'),
            'r_unit': Unit('kpc'),
        },
        'intensity': {
            # For this test, use truth info
            'type': 'inclined_exp',
            'flux': 3.8e4, # counts
            'hlr': 3.5,
        },
        'velocity': {
            'model': 'centered'
        },
        'run_options': {
            'use_numba': False,
            }
    }

    datacube_pars = {
        # image meta pars
        'Nx': 40, # pixels
        'Ny': 40, # pixels
        'pix_scale': 0.5, # arcsec / pixel
        # intensity meta pars
        'true_flux': mcmc_pars['intensity']['flux'],
        'true_hlr': mcmc_pars['intensity']['hlr'], # pixels
        # velocty meta pars
        'v_model': mcmc_pars['velocity']['model'],
        'v_unit': mcmc_pars['units']['v_unit'],
        'r_unit': mcmc_pars['units']['r_unit'],
        # emission line meta pars
        'wavelength': 656.28, # nm; halpha
        'lam_unit': 'nm',
        'z': 0.3,
        'R': 5000.,
        'sky_sigma': 0.5, # pixel counts for mock data vector
        # 'psf': mcmc_pars['psf']
    }

    datacube, vmap, true_im = setup_likelihood_test(
        true_pars, datacube_pars
        )
    Nspec = datacube.Nspec
    lambdas = datacube.lambdas

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

    #---------------------------------------------------------
    # Fits to inclined exp

    true_flux = datacube_pars['true_flux']
    true_hlr = datacube_pars['true_hlr']

    imap = InclinedExponential(
        datacube, flux=true_flux, hlr=true_hlr
        )

    outfile = os.path.join(outdir, 'compare-inc-exp-to-data.png')
    print(f'Plotting inclined exp fit compared to stacked data to {outfile}')
    imap.render(true_pars, datacube, mcmc_pars)
    imap.plot_fit(datacube, outfile=outfile, show=show)

    #---------------------------------------------------------
    # Fits to incined exp + knots
    # NOTE: this no longer works due to how psf is now handled

    # add some knot features
    # knot_frac = 0.05 # not really correct, but close enough for tests
    # datacube_pars['psf'] = gs.Gaussian(fwhm=2) # pixels w/o pix_scale defined
    # mcmc_pars['intensity']['knots'] = {
    #     # 'npoints': 25,
    #     'npoints': 15,
    #     'half_light_radius': 1.*true_hlr,
    #     'flux': knot_frac * true_flux,
    # }

    # datacube, vmap, true_im = setup_likelihood_test(
    #     true_pars, datacube_pars
    #     )

    # outfile = os.path.join(outdir, 'datacube-knots.fits')
    # print(f'Saving test datacube to {outfile}')
    # datacube.write(outfile)

    # outfile = os.path.join(outdir, 'datacube-knots-slices.png')
    # print(f'Saving example datacube slice images to {outfile}')
    # # if Nspec < 10:
    # sqrt = int(np.ceil(np.sqrt(Nspec)))
    # slice_indices = range(Nspec)

    # k = 1
    # for i in slice_indices:
    #     plt.subplot(sqrt, sqrt, k)
    #     plt.imshow(datacube.slices[i]._data, origin='lower')
    #     plt.colorbar()
    #     l, r = lambdas[i]
    #     plt.title(f'lambda=({l:.1f}, {r:.1f})')
    #     k += 1
    # plt.gcf().set_size_inches(12,12)
    # plt.tight_layout()
    # plt.savefig(outfile, bbox_inches='tight', dpi=300)
    # if show is True:
    #     plt.show()
    # else:
    #     plt.close()

    # print('Fitting simulated datacube with shapelet basis')
    # start = time.time()
    # mle_im = fitter.fit(true_pars, datacube, pars=pars)
    # t = time.time() - start
    # print(f'Total fit time took {1000*t:.2f} ms for {fitter.Nbasis} basis funcs')

    # outfile = os.path.join(outdir, 'compare-mle-to-data.png')
    # print(f'Plotting MLE fit compared to stacked data to {outfile}')
    # fitter.plot_mle_fit(datacube, outfile=outfile, show=show)

    # print('Fitting simulated datacube with transformed shapelet basis')
    # start = time.time()
    # mle_im_transform = fitter_transform.fit(true_pars, datacube, pars=pars)
    # t = time.time() - start
    # print(f'Total fit time took {1000*t:.2f} ms for {fitter.Nbasis} transformed basis funcs')

    # outfile = os.path.join(outdir, 'compare-transform-mle--to-data.png')
    # print(f'Plotting transform MLE fit compared to stacked data to {outfile}')
    # fitter_transform.plot_mle_fit(datacube, outfile=outfile, show=show)

    # # Now compare the fits
    # outfile = os.path.join(outdir, 'compare-disk-vs-obs-mle-to-data.png')
    # print(f'Comparing MLE fits to data for basis in obs vs. disk')
    # data = datacube.stack()
    # im = fitter.mle_im
    # im_transform = fitter_transform.mle_im

    # X, Y = utils.build_map_grid(nx, ny)

    # images = [data, im, im_transform,
    #           im-im_transform, im-data, im_transform-data]
    # titles = ['Data', 'Obs basis', 'Disk basis',
    #           'Obs-Disk', 'Obs-data', 'Disk-data']

    # diff_max = np.max([np.max(im-data), np.max(im_transform-data)])
    # diff_min = np.max([np.min(im-data), np.min(im_transform-data)])

    # fig, axes = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True,
    #                          figsize=(12,7))
    # for i in range(6):
    #     ax = axes[i//3, i%3]
    #     if (i//3 == 1) and (i%3 > 0):
    #         vmin, vmax = diff_min, diff_max
    #     else:
    #         vmin, vmax = None, None
    #     mesh = ax.pcolormesh(X, Y, images[i], vmin=vmin, vmax=vmax)
    #     divider = make_axes_locatable(ax)
    #     cax = divider.append_axes('right', size='5%', pad=0.05)
    #     plt.colorbar(mesh, cax=cax)
    #     ax.set_title(titles[i])
    # nbasis = fitter.basis.N
    # fig.suptitle(f'Nmax={nmax}; {nbasis} basis functions')#, y=1.05)
    # plt.tight_layout()

    # plt.savefig(outfile, bbox_inches='tight', dpi=300)

    # if show is True:
    #     plt.show()
    # else:
    #     plt.close()

    #---------------------------------------------------------
    # Intensity map constructors and renders

    print('Initializing a BasisIntensityMap for shapelets in obs plane')
    imap = BasisIntensityMap(
        datacube, basis_type='shapelets', basis_kwargs={'Nmax':nmax,
                                                        'pix_scale':pix_scale,
                                                        'plane':'obs'}
        )
    imap.render(true_pars, datacube, mcmc_pars)

    outfile = os.path.join(outdir, 'shapelet-imap-render.png')
    print(f'Saving render for shapelet basis to {outfile}')
    imap.plot(outfile=outfile, show=show)

    print('Initializing a BasisIntensityMap for shapelets in disk plane')
    imap_transform = BasisIntensityMap(
        datacube, basis_type='shapelets', basis_kwargs={'Nmax':nmax,
                                                        'pix_scale':pix_scale,
                                                        'plane':'disk'}
        )
    imap_transform.render(true_pars, datacube, mcmc_pars)

    outfile = os.path.join(outdir, 'shapelet-imap-transform-render.png')
    print(f'Saving render for shapelet basis to {outfile}')
    imap_transform.plot(outfile=outfile, show=show)

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
