import numpy as np
import os
from argparse import ArgumentParser
from astropy.units import Unit
import zeus

import utils
from mcmc import ZeusRunner
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
        'g1': 0.1,
        'g2': 0.2,
        'sini': 0.8,
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

    Nx, Ny = 30, 30
    Nspec = len(lambdas)
    shape = (Nx, Ny, Nspec)

    print('Setting up test datacube and true Halpha image')
    datacube, vmap, true_im = _setup_likelihood_test(
        true_pars, pars, shape, lambdas
        )

    print('Setting up ZeusRunner object')
    ndims = len(PARS_ORDER)
    nwalkers = 2*ndims
    args = [datacube]
    kwargs = {'pars': pars}
    runner = ZeusRunner(
        nwalkers, ndims, log_posterior, args=args, kwargs=kwargs
        )

    nsteps = 10
    # pudb.set_trace()
    runner.run(nsteps)

    return 0

if __name__ == '__main__':
    args = parser.parse_args()

    print('Starting tests')
    rc = main(args)

    if rc == 0:
        print('All tests ran succesfully')
    else:
        print(f'Tests failed with return code of {rc}')
