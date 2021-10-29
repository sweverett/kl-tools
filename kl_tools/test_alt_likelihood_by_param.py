import numpy as np
import os
from multiprocessing import Pool
import velocity
import parameters
from astropy.units import Unit
from argparse import ArgumentParser
import matplotlib.pyplot as plt

import velocity
import utils
from likelihood import setup_likelihood_test, log_likelihood
from parameters import pars2theta

parser = ArgumentParser()

parser.add_argument('--show', action='store_true',
                    help='Set to show plots')

def main(args):

    show = args.show

    outdir = os.path.join(utils.TEST_DIR, 'test_alt_likelihood_by_param')
    utils.make_dir(outdir)

    true_pars = {
        'g1': 0.05,
        'g2': -0.025,
        'theta_int': np.pi / 3,
        'sini': 0.8,
        'v0': 10.,
        'vcirc': 200,
        'rscale': 5,
    }

    # additional args needed for prior / likelihood evaluation
    halpha = 656.28 # nm
    R = 5000.
    z = 0.3
    pars = {
        'Nx': 30, # pixels
        'Ny': 30, # pixels
        'true_flux': 1e5, # counts
        'true_hlr': 5, # pixels
        'v_unit': Unit('km / s'),
        'r_unit': Unit('kpc'),
        'z': z,
        'spec_resolution': R,
        # 'line_std': 0.17,
        'line_std': halpha * (1.+z) / R, # emission line SED std; nm
        'line_value': 656.28, # emission line SED std; nm
        'line_unit': Unit('nm'),
        'sed_start': 650,
        'sed_end': 660,
        'sed_resolution': 0.025,
        'sed_unit': Unit('nm'),
        'cov_sigma': 0.5, # pixel counts; dummy value
        'bandpass_throughput': '.2',
        'bandpass_unit': 'nm',
        'bandpass_zp': 30,
        # 'priors': {
        #     'g1': priors.GaussPrior(0., 0.1),#, clip_sigmas=2),
        #     'g2': priors.GaussPrior(0., 0.1),#, clip_sigmas=2),
        #     'theta_int': priors.UniformPrior(0., np.pi),
        #     # 'theta_int': priors.UniformPrior(np.pi/3, np.pi),
        #     'sini': priors.UniformPrior(0., 1.),
        #     # 'sini': priors.GaussPrior()
        #     'v0': priors.UniformPrior(0, 20),
        #     'vcirc': priors.GaussPrior(200, 10, clip_sigmas=2),
        #     # 'vcirc': priors.UniformPrior(190, 210),
        #     'rscale': priors.UniformPrior(0, 10),
        # },
        'intensity': {
            # For this test, use truth info
            'type': 'inclined_exp',
            'flux': 1e5, # counts
            'hlr': 5, # pixels
            # 'type': 'basis',
            # 'basis_type': 'shapelets',
            # 'basis_kwargs': {
            #     'Nmax': 15,
                # 'plane': 'disk'
                # 'plane': 'obs'
                # }
        },
        # 'marginalize_intensity': True,
        # 'psf': gs.Gaussian(fwhm=3), # fwhm in pixels
        'use_numba': False,
    }

    li, le, dl = 655.8, 656.8, 0.1
    lambdas = [(l, l+dl) for l in np.arange(li, le, dl)]

    Nx, Ny = pars['Nx'], pars['Ny']
    Nspec = len(lambdas)
    shape = (Nx, Ny, Nspec)

    print('Setting up test datacube and true Halpha image')
    datacube, sed, vmap, true_im = setup_likelihood_test(
        true_pars, pars, shape, lambdas
        )

    # update pars w/ SED object
    pars['sed'] = sed

    outfile = os.path.join(outdir, 'datacube.fits')
    print(f'Saving test datacube to {outfile}')
    datacube.write(outfile)

    outfile = os.path.join(outdir, 'vmap.png')
    print(f'Saving true vamp in obs plane to {outfile}')
    plt.imshow(vmap, origin='lower')
    plt.colorbar(label='v')
    plt.title('True velocity map in obs plane')
    plt.savefig(outfile, bbox_inches='tight', dpi=300)
    if show is True:
        plt.show()
    else:
        plt.close()

    outfile = os.path.join(outdir, 'datacube-slices.png')
    print(f'Saving example datacube slice images to {outfile}')
    if Nspec < 10:
        sqrt = int(np.ceil(np.sqrt(Nspec)))
        slice_indices = range(Nspec)

    else:
        sqrt = 3
        slice_indices = np.sort(
            np.random.choice(
                range(Nspec),
                size=sqrt**2,
                replace=False
                )
            )

    k = 1
    for i in slice_indices:
        plt.subplot(sqrt, sqrt, k)
        plt.imshow(datacube.slices[i]._data, origin='lower')
        plt.colorbar()
        plt.title(f'Test datacube slice\n lambda={lambdas[i]}')
        k += 1
    plt.gcf().set_size_inches(9,9)
    plt.tight_layout()
    plt.savefig(outfile, bbox_inches='tight', dpi=300)
    if show is True:
        plt.show()
    else:
        plt.close()


    test_pars = {
        'g1': (-0.4, 0.4, .005),
        'g2': (-0.4, 0.4, .005),
        'theta_int': (0., np.pi, .05),
        'sini': (0., 0.99, .01),
        'v0': (0, 20, .05),
        'vcirc': (100, 300, 1),
        'rscale': (0, 10, .05),
    }

    # NOTE: Just for testing, can remove later
    # These are centered at an alt solution,
    # using disk basis Nmax=10 (cov_sig=1)
    # alt_pars = {
    #     'g1': 0.1703,
    #     'g2': -0.2234,
    #     'theta_int': 1.0537,
    #     'sini': 0.9205,
    #     'v0': 9.0550,
    #     'vcirc': 170.6623,
    #     'rscale': 6.0641,
    #     }

    # These are centered at an alt solution,
    # using correct intensity map (cov_sig=0.5)
    alt_pars = {
        'g1': -0.0249,
        'g2': 0.1070,
        'theta_int': 1.0423,
        'sini': 0.5770,
        'v0': 13.7093,
        'vcirc': 275.8384,
        'rscale': 4.2344,
        }

    size = (14,5)
    # sqrt = int(np.ceil(np.sqrt(len(true_pars))))
    nrows = 2
    ncols = 4
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=size
        )

    # TODO: Can add a multiprocessing pool here if needed
    k = 1
    for par, par_range in test_pars.items():
        print(f'Starting loop over {par}: {par_range}')
        theta_pars = true_pars.copy()
        theta_alt_pars = alt_pars.copy()
        theta_true = pars2theta(theta_pars)
        theta_alt = pars2theta(alt_pars.copy())

        # Compute best-fit log likelihood
        true_loglike = log_likelihood(theta_true, datacube, pars)
        alt_loglike = log_likelihood(theta_alt, datacube, pars)

        # Now update w/ test param
        left, right, dx = par_range
        assert right > left
        # N = int((right - left) / dx)
        # N = 1000
        N = 250

        loglike = np.zeros(N)
        par_val = np.zeros(N)

        loglike_alt = np.zeros(N)
        par_val_alt = np.zeros(N)

        for i, val in enumerate(np.linspace(
                left, right, num=N, endpoint=True
                )):
            theta_pars[par] = val
            theta = pars2theta(theta_pars)

            loglike[i] = log_likelihood(theta, datacube, pars)
            par_val[i] = val

            theta_alt_pars[par] = val
            theta_alt = pars2theta(theta_alt_pars)

            loglike_alt[i] = log_likelihood(theta_alt, datacube, pars)
            par_val_alt[i] = val

        plt.subplot(nrows, ncols, k)
        plt.plot(par_val, loglike, c='tab:blue', label='Slice around truth')
        plt.plot(par_val_alt, loglike_alt, c='tab:orange', label='Slice around alt solution')
        truth = true_pars[par]
        alt = alt_pars[par]
        plt.axvline(truth, lw=2, c='tab:blue', ls='--')#, label='Truth')
        plt.axvline(alt  , lw=2, c='tab:orange', ls='--')#, label='Alt')
        plt.legend()
        plt.xlabel(par)

        if (k-1) % ncols == 0:
            plt.ylabel('Log likelihood')
        k += 1

    fig.delaxes(axes[-1,-1])

    plt.suptitle('Log likelihood slice for 1 varying param')
    plt.tight_layout()

    outfile = os.path.join(outdir, f'loglike-slices.png')
    plt.savefig(outfile, bbox_inches='tight', dpi=300)

    if show is True:
        plt.show()

    else:
        plt.close()

    return 0

if __name__ == '__main__':
    args = parser.parse_args()

    print('Starting tests')
    rc = main(args)

    if rc == 0:
        print('All tests ran succesfully')
    else:
        print(f'Tests failed with return code of {rc}')
