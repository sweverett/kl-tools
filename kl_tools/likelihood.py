import numpy as np
import os
from scipy.interpolate import interp1d
from astropy.units import Unit
from argparse import ArgumentParser
import galsim as gs
from galsim.angle import Angle, radians
import matplotlib.pyplot as plt
from numba import njit

import utils
import priors
from parameters import PARS_ORDER, theta2pars, pars2theta
from velocity import VelocityMap2D
from cube import DataCube

import pudb

'''
Classes and functions useful for computing kinematic lensing posteriors
with datacubes from IFU data

Much of these are built as independent functions rather than classes
due to how zeus samples the posterior distribution
'''

parser = ArgumentParser()

parser.add_argument('--show', action='store_true', default=False,
                    help='Set to show test plots')
parser.add_argument('--test', action='store_true', default=False,
                    help='Set to run tests')

class DCLogLikelihood(object):
    '''
    Base class to evaluate a datacube log-likelihood.
    '''

    def __init__(self):
        pass

    def __call__(self, datacube, vmap):
        '''
        datacube: A DataCube object, truncated to desired bounds
        vmap: A VelocityMap2D object
        '''

        # Can remove loop eventually
        for data in datacube.slices:
            model = None
            # ...

        return

def log_posterior(theta, data, pars={}):
    '''
    Natural log of the posterior dist. Target function for zeus
    to sample over

    theta: Parameters sampled by zeus. Order defined in PARS_ORDER
    data: DataCube object, truncated to desired lambda bounds
    [pars]: Dict of any other parameters needed to evaluate
            likelihood
    '''


    logprior = log_prior(theta, pars)

    if logprior == -np.inf:
        return -np.inf

    else:
        return logprior + log_likelihood(theta, data, pars)

def log_prior(theta, pars):
    '''
    theta: Parameters sampled by zeus. Order defined in PARS_ORDER
    '''

    theta_priors = pars['priors']

    logprior = 0

    for name, prior in theta_priors.items():
        indx = PARS_ORDER[name]
        logprior += prior(theta[indx], log=True)

    return logprior

def log_likelihood(theta, datacube, pars):
    '''
    Do setup and type / sanity checking here
    before calling _loglikelihood

    theta: Parameters sampled by zeus. Order defined in PARS_ORDER
    datacube: DataCube object, truncated to desired lambda bounds
    pars: Dict of any other parameters needed to evaluate likelihood
    '''

    # unpack sampled zeus params
    theta_pars = theta2pars(theta)

    # add to the rest of pars
    # pars = {**theta_pars, **pars}

    Nx = datacube.Nx
    Ny = datacube.Ny
    Nspec = datacube.Nspec

    lambdas = np.array(datacube.lambdas)

    # create grid of pixel centers in image coords
    X, Y = utils.build_map_grid(Nx, Ny)

    # create 2D velocity & intensity maps given sampled transformation
    # parameters
    vmap = _setup_vmap(theta_pars, pars)
    imap = _setup_imap(theta_pars, pars)

    # evaluate maps at pixel centers in obs plane
    if 'use_numba' in pars:
        use_numba = pars['use_numba']
    else:
        use_numba = False
    v_array = vmap(
        'obs', X, Y, normalized=True, use_numba=use_numba
        )

    # try:
    #     print(theta)
    #     plt.subplot(1,2,1)
    #     plt.imshow(v_array, origin='lower')
    #     plt.colorbar()
    #     plt.title('obs plane')
    #     plt.subplot(1,2,2)
    #     plt.imshow(vmap('disk', X, Y, normalized=True, speed=True), origin='lower')
    #     plt.colorbar()
    #     plt.title('disk plane')
    #     plt.tight_layout()
    #     plt.show()
    # except KeyboardInterrupt:
    #     raise

    # TODO: Implement!
    # i_array = imap(X, Y)
    i_array = imap

    # numba can't handle interp tables, so we do it ourselves
    sed = pars['sed']
    sed_array = np.array([sed.x, sed.y])

    Npix = Nx * Ny

    # sigma = pars['cov_sigma']
    # cov = _setup_cov_matrix(Npix, sigma)

    # NOTE: This doesn't work with numba
    cov = _setup_cov_matrix(Npix, pars)

    return _log_likelihood(
        datacube._data, lambdas, v_array, i_array, sed_array, cov
        )

@njit
def _log_likelihood(datacube, lambdas, vmap, imap, sed, cov):
    '''
    datacube: The (Nx,Ny,Nspec) data array of a DataCube object,
                truncated to desired lambda bounds
    # lambdas: Wavelength values at the center of each datacube
    #             slice
    lambdas: List of lambda tuples that define slice edges
    vmap: An (Nx,Ny) array of normalized (by c) velocity values in the
            obs plane returned by a call from a VelocityMap2D object
    imap: An (Nx,Ny) array of intensity values in the obs plane
            returned by a call from a IntensityMap object
    sed: A 2D numpy array with axis=0 being the lambda values of
         a 1D interpolation table, with axis=1 being the corresponding
         SED values
    cov: A (Nx*Ny, Nx*Ny) covariance matrix for the image pixels

    TODO: Need to check that the sed values are correct for this expression
    '''

    Nx = datacube.shape[0]
    Ny = datacube.shape[1]
    Nslices = datacube.shape[2]

    # vmap is ~z in this case, as it has been normalized by c
    # approx valid for v << c
    zfactor = 1. / (1. + vmap)

    # model = np.empty(datacube.shape)

    loglike = 0

    # can figure out how to remove this for loop later
    # will be fast enough with numba anyway
    for i in range(Nslices):

        model = _compute_slice_model(
            lambdas[i], sed, zfactor, imap
            )

        diff = (datacube[:,:,i] - model).reshape(Nx*Ny)
        chi2 = diff.T.dot(cov.dot(diff))

        loglike += -0.5*chi2

    return loglike

@njit
def _compute_slice_model(lambdas, sed, zfactor, imap):
    '''
    lambdas: tuple
        The wavelength tuple (lambda_blue, lambda_red) for a
        given slice
    sed: np.ndarray
        A 2D numpy array with axis=0 being the lambda values of
        a 1D interpolation table, with axis=1 being the corresponding
        SED values
    zfactor: np.ndarray
        A 2D numpy array corresponding to the (normalized by c) velocity
        map at each pixel
    imap: np.ndarray
        A 2D numpy array corresponding to the source intensity map
        at the emission line
    '''

    lblue, lred = lambdas[0], lambdas[1]

    # Get mean SED vlaue in slice range
    # NOTE: We do it this way as numba won't allow
    #       interp1d objects
    sed_b = _interp1d(sed, lblue*zfactor)
    sed_r = _interp1d(sed, lred*zfactor)

    # Numba won't let us use np.mean w/ axis=0
    mean_sed = (sed_b + sed_r) / 2.
    int_sed = (lred - lblue) * mean_sed
    model = imap * int_sed

    return model

@njit
def _interp1d(table, values, kind='linear'):
    '''
    Interpolate table(value)

    table: A 2D numpy array with axis=0 being the x-values and
           axis=1 being the function evaluated at x
    values: The values to interpolate the table on
    '''

    if kind == 'linear':
        # just use numpy linear interpolation, as it works with numba
        interp = np.interp(values, table[0], table[1])
    else:
        raise ValueError('Non-linear interpolations not yet implemented!')

    return interp

def _setup_vmap(theta_pars, pars, model_name='default'):
    '''
    theta_pars: dict
        A dict of the sampled mcmc params for both the velocity
        map and the tranformation matrices
    pars: dict
        A dict of anything else needed to compute the posterior
    model_name: str
        The model name to use when constructing the velocity map
    '''

    vmodel = theta_pars

    for name in ['r_unit', 'v_unit']:
        if name in pars:
            vmodel[name] = Unit(pars[name])
        else:
            raise AttributeError(f'pars must have a value for {name}!')

    return VelocityMap2D(model_name, vmodel)

def _setup_imap(theta_pars, pars, model_name='default'):
    '''
    TODO: Implement!

    NOTE: For now, just doing an inclined exp with truth info
    for testing
    '''

    # A = theta_pars['A']
    inc = Angle(np.arcsin(theta_pars['sini']), radians)

    imap = gs.InclinedExponential(
        inc, flux=pars['true_flux'], half_light_radius=pars['true_hlr']
        )

    rot_angle = Angle(theta_pars['theta_int'], radians)
    imap = imap.rotate(rot_angle)

    imap = imap.shear(g1=theta_pars['g1'], g2=theta_pars['g2'])

    psf = pars['psf']

    imap = gs.Convolve([imap, psf])

    Nx, Ny = 30, 30
    imap = imap.drawImage(nx=Nx, ny=Ny)

    # imap = IntensityMap2D(pars)

    return imap.array

def _setup_sed(pars):
    '''
    Build interpolation table for normalized emission line SED

    pars: dict contianing emission line parameters

    TODO: See if constructing the SED once and passing it in args
          will cause problems for optimizations
    '''

    start = pars['sed_start']
    end = pars['sed_end']
    res = pars['sed_resolution']

    lambdas = np.arange(start, end+res, res)

    # Model emission line SED as gaussian
    mu  = pars['line_value']
    std = pars['line_std']
    unit = pars['line_unit']

    norm = 1. / (std * np.sqrt(2.*np.pi))
    gauss = norm * np.exp(-0.5*(lambdas - mu)**2 / std**2)

    sed = interp1d(lambdas, gauss)

    return sed

def _setup_cov_matrix(Npix, pars):
    '''
    Build covariance matrix for slice images

    Npix: int
        number of pixels
    pars: dict
        dictionary containing parameters needed to build cov matrix

    # TODO: For now, a single cov matrix for each slice. However, this could be
            generalized to a list of cov matrices
    '''

    # TODO: For now, treating pixel covariance as diagonal
    #       and uniform for a given slice
    sigma = pars['cov_sigma']

    cov = sigma**2 * np.identity(Npix)

    return cov

def _setup_likelihood_test(true_pars, pars, shape, lambdas):

    throughput = pars['bandpass_throughput']
    unit = pars['bandpass_unit']
    zp = pars['bandpass_zp']

    bandpasses = []
    # dl = lambdas[1] - lambdas[0]
    # for l1, l2 in zip(lambdas, lambdas+dl):
    for l1, l2 in lambdas:
        bandpasses.append(gs.Bandpass(
            throughput, unit, blue_limit=l1, red_limit=l2, zeropoint=zp
            ))

    assert shape[2] == len(bandpasses)

    sed = _setup_sed(pars)

    datacube, vmap, true_im = _setup_test_datacube(
        shape, lambdas, bandpasses, sed, true_pars, pars
        )

    return datacube, sed, vmap, true_im

def _setup_test_datacube(shape, lambdas, bandpasses, sed, true_pars, pars):
    Nx, Ny, Nspec = shape[0], shape[1], shape[2]

    # make true obs image at Halpha
    flux = pars['true_flux']
    hlr = pars['true_hlr'] # without pixscale set, this is in pixels

    inc = Angle(np.arcsin(true_pars['sini']), radians)

    true = gs.InclinedExponential(
        inc, flux=flux, half_light_radius=hlr
        )

    rot_angle = Angle(true_pars['theta_int'], radians)
    true = true.rotate(rot_angle)
    true = true.shear(g1=true_pars['g1'], g2=true_pars['g2'])

    # use psf set in pars
    psf = pars['psf']

    true = gs.Convolve([true, psf])

    true_im = true.drawImage(nx=Nx, ny=Ny)

    vel_pars = {}
    for name in PARS_ORDER.keys():
        vel_pars[name] = true_pars[name]
    vel_pars['v_unit'] = pars['v_unit']
    vel_pars['r_unit'] = pars['r_unit']

    vmap = VelocityMap2D('default', vel_pars)

    X, Y = utils.build_map_grid(Nx, Ny)
    Vnorm = vmap('obs', X, Y, normalized=True)

    # We use this one for the return map
    V = vmap('obs', X, Y)

    data = np.zeros(shape)

    # numba won't allow a scipy.interp1D object
    sed_array = np.array([sed.x, sed.y])

    for i in range(Nspec):
        zfactor = 1. / (1 + Vnorm)

        # TODO: add _model call!!
        obs_array = _compute_slice_model(
            lambdas[i], sed_array, zfactor, true_im.array
            )

        obs_im = gs.Image(obs_array)

        # lblue, lred = lambdas[i]
        # sed_b = sed(lblue * zfactor)
        # sed_r = sed(lred  * zfactor)

        # mean_sed = (sed_b + sed_r) / 2.
        # int_sed = (lred - lblue) * mean_sed

        # obs_im = true_im * int_sed

        noise = gs.GaussianNoise(sigma=pars['cov_sigma'])
        obs_im.addNoise(noise)

        data[:,:,i] = obs_im.array

    datacube = DataCube(data=data, bandpasses=bandpasses)

    return datacube, V, true_im.array

def main(args):

    show = args.show

    outdir = os.path.join(utils.TEST_DIR, 'likelihood')
    utils.make_dir(outdir)

    true_pars = {
        'g1': 0.1,
        'g2': 0.2,
        'sini': 0.75,
        'theta_int': np.pi / 3,
        'v0': 250,
        'vcirc': 25,
        'rscale': 20,
    }

    # additional args needed for prior / likelihood evaluation
    pars = {
        'Nx': 30, # pixels
        'Ny': 30, # pixels
        'true_flux': 1e4, # counts
        'true_hlr': 3, # pixels
        'v_unit': Unit('km / s'),
        'r_unit': Unit('kpc'),
        'line_std': 2, # emission line SED std; nm
        'line_value': 656.28, # emission line SED std; nm
        'line_unit': Unit('nm'),
        'sed_start': 600,
        'sed_end': 700,
        'sed_resolution': 0.5,
        'sed_unit': Unit('nm'),
        'cov_sigma': 1., # pixel counts; dummy value
        'bandpass_throughput': '0.2',
        'bandpass_unit': 'nm',
        'bandpass_zp': 30,
        'priors': {
            'g1': priors.GaussPrior(0., 0.1),# clip_sigmas=3),
            'g2': priors.GaussPrior(0., 0.1),# clip_sigmas=3),
            'theta_int': priors.UniformPrior(0., np.pi),
            'sini': priors.UniformPrior(0., 1.),
            'v0': priors.UniformPrior(1400, 1600),
            'vcirc': priors.UniformPrior(175, 225),
            'rscale': priors.UniformPrior(0, 10),
        },
        'psf': gs.Gaussian(fwhm=3), # fwhm in pixels
    }

    li, le, dl = 654, 657, 1
    lambdas = [(l, l+dl) for l in range(li, le, dl)]

    Nx, Ny = pars['Nx'], pars['Ny']
    Nspec = len(lambdas)
    shape = (Nx, Ny, Nspec)

    print('Setting up test datacube and true Halpha image')
    datacube, sed, vmap, true_im = _setup_likelihood_test(
        true_pars, pars, shape, lambdas
        )

    outfile = os.path.join(outdir, 'true-im.png')
    print(f'Saving true intensity profile in obs plane to {outfile}')
    plt.imshow(true_im, origin='lower')
    plt.colorbar()
    plt.title('True Halpha profile in obs plane')
    plt.savefig(outfile, bbox_inches='tight')
    if show is True:
        plt.show()
    else:
        plt.close()

    outfile = os.path.join(outdir, 'vmap.png')
    print(f'Saving true vamp in obs plane to {outfile}')
    plt.imshow(vmap, origin='lower')
    plt.colorbar(label='v / c')
    plt.title('True velocity map in obs plane')
    plt.savefig(outfile, bbox_inches='tight')
    if show is True:
        plt.show()
    else:
        plt.close()

    outfile = os.path.join(outdir, 'datacube.fits')
    print(f'Saving test datacube to {outfile}')
    datacube.write(outfile)

    outfile = os.path.join(outdir, 'datacube-slice.png')
    print(f'Saving example datacube slice im to {outfile}')
    lc = Nspec // 2
    s = datacube.slices[lc]
    plt.imshow(s._data, origin='lower')
    plt.colorbar()
    plt.title(f'Test datacube slice at lambda={lambdas[lc]}')
    plt.savefig(outfile, bbox_inches='tight')
    if show is True:
        plt.show()
    else:
        plt.close()

    theta = pars2theta(true_pars)

    print('Computing log posterior for correct theta')
    ptrue = log_posterior(theta, datacube, pars)
    chi2_true = -2.*ptrue / (Nx*Ny*Nspec - len(theta))
    print(f'Posterior value = {ptrue:.2f}')

    print('Computing log posterior for random scatter about correct theta')
    N = 1000
    p = []
    chi2 = []
    new_thetas = []
    radius = 0.25
    for i in range(N):
        scale = radius * np.array(theta)
        new_theta = theta + scale * np.random.rand(len(theta))
        new_thetas.append(new_theta)
        p.append(log_posterior(new_theta, datacube, pars))
        chi2.append(-2.*p[i] / (Nx*Ny*Nspec - len(new_theta)))
    if N <= 10:
        print(f'Posterior values:\n{p}')

    # outfile = os.path.join(outdir, 'posterior-dist-ball.png')
    outfile = os.path.join(outdir, 'posterior-dist-ball.png')
    print('Plotting hist of reduced chi2 vs. chi2 at truth to {outfile}')
    cmin = np.min(chi2)
    cmax = np.max(chi2)
    Nbins = 20
    bins = np.linspace(cmin, cmax, num=Nbins, endpoint=True)
    plt.hist(chi2, ec='k', bins=bins)
    plt.axvline(chi2_true, ls='--', c='k', lw=2, label='Eval at true theta')
    plt.legend()
    plt.xlabel('Reduced Chi2')
    plt.ylabel('Counts')
    plt.yscale('log')
    plt.title('Reduced chi2 evaluations for a random ball centered at truth\n '
              f'with radius = {radius} * truth')
    plt.savefig(outfile, bbox_inches='tight')
    if show is True:
        plt.show()
    else:
        plt.close()

    outfile = os.path.join(outdir, 'theta-diff.png')
    print('Plotting diff between true theta and MAP')
    best = new_thetas[int(np.where(chi2 == np.min(chi2))[0])]
    plt.plot(100. * (best-theta) / theta, 'o', c='k', markersize=5)
    plt.axhline(0, c='k', ls='--', lw=2)
    xx = range(0, len(best))
    plt.fill_between(
        xx, -10*np.ones(len(xx)), 10*np.ones(len(xx)), color='gray', alpha=0.25
        )
    my_xticks = len(best)*['']
    for name, indx in PARS_ORDER.items():
        my_xticks[indx] = name
    plt.xticks(xx, my_xticks)
    plt.xlabel('theta params')
    plt.ylabel('Percent Error (MAP - true)')
    plt.title('% Error in MAP vs. true sampled params')
    plt.savefig(outfile, bbox_inches='tight')
    if show is True:
        plt.show()
    else:
        plt.close()

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
