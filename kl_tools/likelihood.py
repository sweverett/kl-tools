import numpy as np
import os
from scipy.interpolate import interp1d
from astropy.units import Unit
from argparse import ArgumentParser
import galsim as gs
from galsim.angle import Angle, radians
import matplotlib.pyplot as plt

import utils
import priors
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

# order of sampled mcmc parameters
# NOTE: This won't be accessible if we use numba
PARS_ORDER = {
    'g1': 0,
    'g2': 1,
    'theta_int': 2,
    'sini': 3,
    'r0': 4,
    'rscale': 5,
    'v0': 6,
    'vcirc': 7
    }

def theta2pars(theta):
    '''
    uses PARS_ORDER to convert list of sampled params to dict
    '''

    assert len(theta) == len(PARS_ORDER)

    pars = {}

    for key, indx in PARS_ORDER.items():
        pars[key] = theta[indx]

    return pars

def pars2theta(pars):
    '''
    convert dict of paramaeters to theta list
    '''

    # initialize w/ junk
    theta = len(PARS_ORDER) * ['']

    for name, indx in PARS_ORDER.items():
        theta[indx] = pars[name]

    return theta

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

    max_size = pars['max_size']
    max_velocity =pars['max_velocity']
    theta_priors = {
        'g1': priors.GaussPrior(0., 0.15),
        'g2': priors.GaussPrior(0., 0.15),
        'theta_int': priors.UniformPrior(0., np.pi),
        'sini': priors.UniformPrior(0., 1.),
        'v0': priors.GaussPrior(0, max_velocity),
        'vcirc': priors.GaussPrior(0, max_velocity),
        'r0': priors.UniformPrior(0, max_size),
        'rscale': priors.UniformPrior(0, max_size),
    }

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

    lambdas = datacube.lambdas

    # create grid of pixel centers in image coords
    X, Y = _build_map_grid(Nx, Ny)

    # create 2D velocity & intensity maps given sampled transformation
    # parameters
    vmap = _setup_vmap(theta_pars)
    imap = _setup_imap(theta_pars, pars)

    # Setup SED interpolation table
    # TODO: Prefer to set this up once and pass into function,
    #       but may cause problems with numba
    sed  = _setup_sed(pars)

    # evaluate maps at pixel centers in obs plane
    v_array = vmap('obs', X, Y, normalized=True)

    # TODO: Implement!
    # i_array = imap(X, Y)
    i_array = imap

    Npix = Nx * Ny
    cov = _setup_cov_matrix(Npix, pars)

    return _log_likelihood(
        datacube._data, lambdas, v_array, i_array, sed, cov
        )

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
    sed: An interpolation table of the SED of the emission line
    cov: A (Nx*Ny, Nx*Ny) covariance matrix for the image pixels

    TODO: Need to check that the sed values are correct for this expression
    '''

    Nx = datacube.shape[0]
    Ny = datacube.shape[1]
    Nslices = datacube.shape[2]

    # vmap is ~z in this case, as it has been normalized by c
    # approx valid for v << c
    zfactor = 1. / (1 + vmap)

    # model = np.empty(datacube.shape)

    loglike = 0

    # can figure out how to remove this for loop later
    # will be fast enough with numba anyway
    for i in range(Nslices):
        # Get mean SED vlaue in slice range
        lblue, lred = lambdas[i]
        sed_b = sed(lblue * zfactor)
        sed_r = sed(lred  * zfactor)
        mean_sed = np.mean([sed_b, sed_r], axis=0)
        # model[:,:,i] = imap * sed(lam * zfactor)

        # model[:,:,i] = imap * mean_sed
        model = imap * mean_sed

        diff = (datacube[:,:,i] - model).reshape(Nx*Ny)
        chi2 = diff.T.dot(cov.dot(diff))

        loglike += -0.5*chi2

    return loglike

def _build_map_grid(Nx, Ny):
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

def _setup_vmap(pars, model_name='default', runit='kpc', vunit='km/s'):
    '''
    pars is already a dict of the sampled
    parameters for both the velocity map
    and transformation matrices
    '''

    vmodel = pars
    vmodel['r_unit'] = Unit(runit)
    vmodel['v_unit'] = Unit(vunit)

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
        inc, flux=1e4, half_light_radius=3
        )

    rot_angle = Angle(theta_pars['theta_int'], radians)
    imap = imap.rotate(rot_angle)
    imap = imap.shear(g1=theta_pars['g1'], g2=theta_pars['g2'])

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

    Npix: number of pixels
    pars: dict containing parameters needed to build cov matrix

    # TODO: For now, a single cov matrix for each slice. However, this could be
            generalized to a list of cov matrices
    '''

    # TODO: For now, treating pixel covariance as diagonal
    #       and uniform for a given slice
    sigma = pars['cov_sigma']

    cov = sigma * np.identity(Npix)

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

    return datacube, vmap, true_im

def _setup_test_datacube(shape, lambdas, bandpasses, sed, true_pars, pars):
    Nx, Ny, Nspec = shape[0], shape[1], shape[2]

    # make true obs image at Halpha
    flux = true_pars['flux']
    hlr = true_pars['hlr'] # without pixscale set, this is in pixels

    inc = Angle(np.arcsin(true_pars['sini']), radians)

    true = gs.InclinedExponential(
        inc, flux=flux, half_light_radius=hlr
        )

    rot_angle = Angle(true_pars['theta_int'], radians)
    true = true.rotate(rot_angle)
    true = true.shear(g1=true_pars['g1'], g2=true_pars['g2'])

    true_im = true.drawImage(nx=Nx, ny=Ny)

    vel_pars = {}
    for name in PARS_ORDER.keys():
        vel_pars[name] = true_pars[name]
    vel_pars['v_unit'] = true_pars['v_unit']
    vel_pars['r_unit'] = true_pars['r_unit']

    vmap = VelocityMap2D('default', vel_pars)

    X, Y = _build_map_grid(Nx, Ny)
    V = vmap('obs', X, Y, normalized=True)

    data = np.zeros(shape)

    for i in range(Nspec):
        zfactor = 1. / (1 + V)

        lblue, lred = lambdas[i]
        sed_b = sed(lblue * zfactor)
        sed_r = sed(lred  * zfactor)
        mean_sed = np.mean([sed_b, sed_r], axis=0)

        obs_im = true_im * mean_sed

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
        'r0': 10,
        'rscale': 20,
        'v_unit': Unit('km / s'),
        'r_unit': Unit('kpc'),
        'flux': 1e4, # counts
        'hlr': 3, # pixels
    }

    # additional args needed for prior / likelihood evaluation
    pars = {
        'Nx': 30, # pixels
        'Ny': 30, # pixels
        'max_size': 30, # pixels
        'max_velocity': 0.01, # for v / c
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
    }

    li, le, dl = 654, 657, 1
    lambdas = [(l, l+dl) for l in range(li, le, dl)]

    Nx, Ny = pars['Nx'], pars['Ny']
    Nspec = len(lambdas)
    shape = (Nx, Ny, Nspec)

    print('Setting up test datacube and true Halpha image')
    datacube, vmap, true_im = _setup_likelihood_test(
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
