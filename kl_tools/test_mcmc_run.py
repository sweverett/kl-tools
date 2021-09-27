import numpy as np
import os
import pickle
from argparse import ArgumentParser
from astropy.units import Unit
import galsim as gs
import matplotlib.pyplot as plt
import zeus

import utils
from mcmc import ZeusRunner
import priors
from likelihood import log_posterior, _setup_likelihood_test, PARS_ORDER

import pudb

parser = ArgumentParser()

parser.add_argument('--show', action='store_true', default=False,
                    help='Set to show test plots')

def main(args):

    show = args.show

    outdir = os.path.join(utils.TEST_DIR, 'test-mcmc-run')
    utils.make_dir(outdir)

    true_pars = {
        'g1': 0.05,
        'g2': -0.025,
        'theta_int': np.pi / 3,
        'sini': 0.9,
        'v0': 1500,
        'vcirc': 200,
        'r0': 5,
        'rscale': 5,
        'v_unit': Unit('km / s'),
        'r_unit': Unit('kpc'),
    }

    # additional args needed for prior / likelihood evaluation
    pars = {
        'Nx': 30, # pixels
        'Ny': 30, # pixels
        'true_flux': 5e4, # counts
        'true_hlr': 5, # pixels
        'line_std': 2, # emission line SED std; nm
        'line_value': 656.28, # emission line SED std; nm
        'line_unit': Unit('nm'),
        'sed_start': 600,
        'sed_end': 700,
        'sed_resolution': 0.5,
        'sed_unit': Unit('nm'),
        'cov_sigma': 1, # pixel counts; dummy value
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
            'r0': priors.UniformPrior(0, 10),
            'rscale': priors.UniformPrior(0, 10),
        },
        'psf': gs.Gaussian(fwhm=3), # fwhm in pixels
    }

    # li, le, dl = 656, 657, 1
    li, le, dl = 651, 661, 1
    lambdas = [(l, l+dl) for l in range(li, le, dl)]

    Nx, Ny = 30, 30
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
    plt.savefig(outfile, bbox_inches='tight', dpi=300)
    if show is True:
        plt.show()
    else:
        plt.close()

    outfile = os.path.join(outdir, 'vmap.png')
    print(f'Saving true vamp in obs plane to {outfile}')
    plt.imshow(vmap, origin='lower')
    plt.colorbar(label='v / c')
    plt.title('True velocity map in obs plane')
    plt.savefig(outfile, bbox_inches='tight', dpi=300)
    if show is True:
        plt.show()
    else:
        plt.close()

    outfile = os.path.join(outdir, 'datacube.fits')
    print(f'Saving test datacube to {outfile}')
    datacube.write(outfile)

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


    pars['sed'] = sed

    print('Setting up ZeusRunner object')
    ndims = len(PARS_ORDER)
    nwalkers = 2*ndims
    args = [datacube]
    kwargs = {'pars': pars}
    runner = ZeusRunner(
        nwalkers, ndims, log_posterior, args=args, kwargs=kwargs,
        priors=pars['priors']
        )

    nsteps = 10000
    runner.run(nsteps, ncores=8)

    outfile = os.path.join(outdir, 'test-mcmc-sampler.pkl')
    print(f'Pickling result to {outfile}')
    with open(outfile, 'wb') as f:
        pickle.dump(runner.sampler, f)

    truth = np.zeros(len(PARS_ORDER))
    for name, indx in PARS_ORDER.items():
        truth[indx] = true_pars[name]
    outfile = os.path.join(outdir, 'test-mcmc-truth.pkl')
    print(f'Pickling truth to {outfile}')
    with open(outfile, 'wb') as f:
        pickle.dump(truth, f)

    return 0

if __name__ == '__main__':
    args = parser.parse_args()

    print('Starting tests')
    rc = main(args)

    if rc == 0:
        print('All tests ran succesfully')
    else:
        print(f'Tests failed with return code of {rc}')
