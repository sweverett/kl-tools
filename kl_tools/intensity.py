import numpy as np
import os
import time
from abc import abstractmethod
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from argparse import ArgumentParser
import galsim as gs
from galsim.angle import Angle, radians

import utils
import basis
import likelihood
import parameters
from transformation import transform_coords, TransformableImage

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

        self.image = None

        return

    def render(self, theta_pars, datacube, pars, redo=False):
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
                return self.image

        return self._render(theta_pars, datacube, pars)

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

class InclinedExponential(IntensityMap):
    '''
    This class is mostly for testing purposes. Can give the
    true flux, hlr for a simulated datacube to ensure perfect
    intensity map modeling for validation tests.

    We explicitly use an exponential over a general InclinedSersic
    as it is far more efficient to render, and is only used for
    testing anyway
    '''

    def __init__(self, datacube, flux=None, hlr=None):
        '''
        datacube: DataCube
            While this implementation will not use the datacube
            image explicitly (other than shape info), most will
        flux: float
            Object flux
        hlr: float
            Object half-light radius (in pixels)
        '''

        nx, ny = datacube.Nx, datacube.Ny
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

    def _render(self, theta_pars, datacube, pars):
        '''
        theta_pars: dict
            A dict of the sampled mcmc params for both the velocity
            map and the tranformation matrices
        datacube: DataCube
            Truncated data cube of emission line
        pars: dict
            A dictionary of any additional parameters needed
            to render the intensity map

        return: np.ndarray
            The rendered intensity map
        '''

        # A = theta_pars['A']
        inc = Angle(np.arcsin(theta_pars['sini']), radians)

        gal = gs.InclinedExponential(
            inc, flux=self.flux, half_light_radius=self.hlr
        )

        # Only add knots if a psf is provided
        if 'psf' in pars:
            if 'knots' in pars:
                knot_pars = pars['knots']
                knots = gs.RandomKnots(**knot_pars)
                gal = gal + knots

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
        self.basis_kwargs = basis_kwargs

        self._setup_fitter(basis_type, nx, ny, basis_kwargs=basis_kwargs)

        # if fit_now is True:
        #     self._setup_fitter(
        #         basis_type, nx, ny, basis_kwargs=basis_kwargs
        #         )

        #     self._fit_to_datacube(datacube)
        #     self.datacube = None # Not needed in this case
        # else:
        # Not all subclasses can fit to the datacube at instantiation time,
        # so we wait to initialize fitter until rendering
        self.image = None

        return

    def _setup_fitter(self, basis_type, nx, ny, basis_kwargs=None):
        self.fitter = IntensityMapFitter(
            basis_type, nx, ny, basis_kwargs=basis_kwargs
            )

        return

    def _fit_to_datacube(self, theta_pars, datacube, pars):
        self.image = self.fitter.fit(theta_pars, datacube, pars)

        return

    def get_basis(self):
        return self.fitter.basis

    def render(self, theta_pars, datacube, pars, redo=False):
        return super(BasisIntensityMap, self).render(
            theta_pars, datacube, pars, redo=redo
            )

    def _render(self, theta_pars, datacube, pars):

        self._fit_to_datacube(theta_pars, datacube, pars)

        return self.image

def get_intensity_types():
    return INTENSITY_TYPES

# NOTE: This is where you must register a new model
INTENSITY_TYPES = {
    'default': BasisIntensityMap,
    'basis': BasisIntensityMap,
    'inclined_exp': InclinedExponential,
    }

def build_intensity_map(name, datacube, kwargs):
    '''
    name: str
        Name of intensity map type
    datacube: DataCube
        The datacube whose stacked image the intensity map
        will represent
    kwargs: dict
        Keyword args to pass to intensity constructor
    '''

    name = name.lower()

    if name in INTENSITY_TYPES.keys():
        # User-defined input construction
        intensity = INTENSITY_TYPES[name](datacube, **kwargs)
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

        for name, n in {'nx':nx, 'ny':ny}.items():
            if name in basis_kwargs:
                if n != basis_kwargs[name]:
                    raise ValueError(f'{name} must be consistent if ' +\
                                       'also passed in basis_kwargs!')

        self.basis_type = basis_type
        self.nx = nx
        self.ny = ny

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
            basis_kwargs = {
                'nx': self.nx,
                'ny': self.ny,
                'plane': 'disk'
                }

        self.basis = basis.build_basis(self.basis_type, basis_kwargs)
        self.Nbasis = self.basis.N
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
        self.design_mat = np.zeros((Ndata, Nbasis))
        for n in range(Nbasis):
            func, func_args = self.basis.get_basis_func(n)
            args = [x, y, *func_args]
            self.design_mat[:,n] = func(*args)

        return

    # TODO: Add @njit when ready
    def _initialize_pseudo_inv(self, theta_pars, max_fail=5, redo=True):
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
                M = phi.T.dot(inv_cov.dot(M))

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

    def fit(self, theta_pars, datacube, pars, cov=None):
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
        '''

        nx, ny = self.nx, self.ny
        if (datacube.Nx, datacube.Ny) != (nx, ny):
            raise ValueError('DataCube must have same dimensions ' +\
                             'as intensity map!')

        # Initialize pseudo-inverse given the transformation parameters
        self._initialize_pseudo_inv(theta_pars)

        # We will fit to the sum of all slices
        data = np.sum(datacube._data, axis=2).reshape(nx*ny)

        # Find MLE basis coefficients
        mle_coeff = self._fit_mle_coeff(data, cov=cov)

        assert len(mle_coeff) == self.Nbasis
        self.mle_coefficients = mle_coeff

        # Now create MLE intensity map
        mle_im = self.basis.render_im(theta_pars, mle_coeff)

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

def main(args):
    '''
    For now, just used for testing the classes
    '''

    show = args.show

    outdir = os.path.join(utils.TEST_DIR, 'intensity')
    utils.make_dir(outdir)

    print('Creating IntensityMapFitter w/ shapelet basis')
    nmax = 12
    nx, ny = 30,30
    basis_kwargs = {'Nmax': nmax, 'plane':'obs'}
    fitter = IntensityMapFitter(
        'shapelets', nx, ny, basis_kwargs=basis_kwargs
        )

    print('Creating IntensityMapFitter w/ transformed shapelet basis')
    basis_kwargs = {'Nmax': nmax, 'plane':'disk'}
    fitter_transform = IntensityMapFitter(
        'shapelets', nx, ny, basis_kwargs=basis_kwargs
        )

    print('Setting up test datacube and true Halpha image')
    true_pars, pars = likelihood.setup_test_pars(nx, ny)

    # add some knot features
    knot_frac = 0.05 # not really correct, but close enough for tests
    pars['psf'] = gs.Gaussian(fwhm=2) # pixels w/o pix_scale defined
    pars['knots'] = {
        # 'npoints': 25,
        'npoints': 15,
        'half_light_radius': 1.*pars['true_hlr'],
        'flux': knot_frac * pars['true_flux'],
    }

    li, le, dl = 655.8, 656.8, 0.1
    lambdas = [(l, l+dl) for l in np.arange(li, le, dl)]

    Nspec = len(lambdas)
    shape = (nx, ny, Nspec)
    datacube, sed, vmap, true_im = likelihood.setup_likelihood_test(
        true_pars, pars, shape, lambdas
        )

    print('test')

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
    # Fits

    print('Fitting simulated datacube with shapelet basis')
    start = time.time()
    mle_im = fitter.fit(true_pars, datacube, pars=pars)
    t = time.time() - start
    print(f'Total fit time took {1000*t:.2f} ms for {fitter.Nbasis} basis funcs')

    outfile = os.path.join(outdir, 'compare-mle-to-data.png')
    print(f'Plotting MLE fit compared to stacked data to {outfile}')
    fitter.plot_mle_fit(datacube, outfile=outfile, show=show)

    print('Fitting simulated datacube with transformed shapelet basis')
    start = time.time()
    mle_im_transform = fitter_transform.fit(true_pars, datacube, pars=pars)
    t = time.time() - start
    print(f'Total fit time took {1000*t:.2f} ms for {fitter.Nbasis} transformed basis funcs')

    outfile = os.path.join(outdir, 'compare-transform-mle--to-data.png')
    print(f'Plotting transform MLE fit compared to stacked data to {outfile}')
    fitter_transform.plot_mle_fit(datacube, outfile=outfile, show=show)

    # Now compare the fits
    outfile = os.path.join(outdir, 'compare-disk-vs-obs-mle-to-data.png')
    print(f'Comparing MLE fits to data for basis in obs vs. disk')
    data = np.sum(datacube._data, axis=2)
    im = fitter.mle_im
    im_transform = fitter_transform.mle_im

    X, Y = utils.build_map_grid(nx, ny)

    images = [data, im, im_transform,
              im-im_transform, im-data, im_transform-data]
    titles = ['Data', 'Obs basis', 'Disk basis',
              'Obs-Disk', 'Obs-data', 'Disk-data']

    diff_max = np.max([np.max(im-data), np.max(im_transform-data)])
    diff_min = np.max([np.min(im-data), np.min(im_transform-data)])

    fig, axes = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True,
                             figsize=(12,7))
    for i in range(6):
        ax = axes[i//3, i%3]
        if (i//3 == 1) and (i%3 > 0):
            vmin, vmax = diff_min, diff_max
        else:
            vmin, vmax = None, None
        mesh = ax.pcolormesh(X, Y, images[i], vmin=vmin, vmax=vmax)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(mesh, cax=cax)
        ax.set_title(titles[i])
    nbasis = fitter.basis.N
    fig.suptitle(f'Nmax={nmax}; {nbasis} basis functions')#, y=1.05)
    plt.tight_layout()

    plt.savefig(outfile, bbox_inches='tight', dpi=300)

    if show is True:
        plt.show()
    else:
        plt.close()

    #---------------------------------------------------------
    # Intensity map constructors and renders

    print('Initializing a BasisIntensityMap for shapelets in obs plane')
    imap = BasisIntensityMap(
        datacube, basis_type='shapelets', basis_kwargs={'Nmax':nmax, 'plane':'obs'}
        )
    imap.render(true_pars, datacube, pars)

    outfile = os.path.join(outdir, 'shapelet-imap-render.png')
    print(f'Saving render for shapelet basis to {outfile}')
    imap.plot(outfile=outfile, show=show)

    print('Initializing a BasisIntensityMap for shapelets in disk plane')
    imap_transform = BasisIntensityMap(
        datacube, basis_type='shapelets', basis_kwargs={'Nmax':nmax, 'plane':'disk'}
        )
    imap_transform.render(true_pars, datacube, pars)

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
