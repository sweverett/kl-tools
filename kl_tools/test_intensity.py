import numpy as np
import os
import time
from argparse import ArgumentParser
from astropy.units import Unit
import matplotlib.pyplot as plt

import utils
import parameters
from likelihood import setup_likelihood_test, log_likelihood

parser = ArgumentParser()

parser.add_argument('N', type=int,
                    help='Number of iterations')

def main():
    args = parser.parse_args()
    N = args.N

    outdir = os.path.join(utils.TEST_DIR, 'test_intensity')
    utils.make_dir(outdir)

    true_pars = {
        'g1': 0.05,
        'g2': -0.025,
        'theta_int': np.pi / 3.,
        'sini': 0.8,
        'v0': 10.,
        'vcirc': 200.,
        'rscale': 5,
    }

    # additional args needed for prior / likelihood evaluation
    pars = {
        'Nx': 30, # pixels
        'Ny': 30, # pixels
        'true_flux': 5e4, # counts
        'true_hlr': 5, # pixels
        'v_unit': Unit('km / s'),
        'r_unit': Unit('pixel'),
        'line_std': .17, # emission line SED std; nm
        'line_value': 656.28, # emission line SED std; nm
        'line_unit': Unit('nm'),
        'sed_start': 650,
        'sed_end': 660,
        'sed_resolution': 0.025,
        'sed_unit': Unit('nm'),
        'cov_sigma': 1, # pixel counts; dummy value
        'bandpass_throughput': '0.2',
        'bandpass_unit': 'nm',
        'bandpass_zp': 30,
        'use_numba': False,
    }

    li, le, dl = 655.8, 656.8, 0.1
    lambdas = [(l, l+dl) for l in np.arange(li, le, dl)]

    Nx, Ny = pars['Nx'], pars['Ny']
    Nspec = len(lambdas)
    shape = (Nspec, Nx, Ny)

    print('Setting up test datacube and true Halpha image')
    datacube, sed, vmap, true_im = setup_likelihood_test(
        true_pars, pars, shape, lambdas
        )

    # update pars w/ SED object
    pars['sed'] = sed

    true_theta = parameters.pars2theta(true_pars)
    scale = 0.025

    # Run w/ galsim drawImage intensity
    pars['intensity'] = {
        'type': 'inclined_exp',
        'flux': pars['true_flux'], # counts
        'hlr': pars['true_hlr'], # pixels
    }

    itype = pars['intensity']['type']
    print(f'Starting {N} likelihood evals for intensity type = {itype}')
    exp_times = np.zeros(N)
    for i in range(N):
        theta = true_theta + \
                scale * np.random.rand(len(true_theta)) - \
                scale/2.
        start = time.time()
        log_likelihood(theta, datacube, pars)
        exp_times[i] = 1000.*(time.time() - start)

    # Run w/ shapelet basis intensity
    pars['intensity'] = {
        'type': 'shapelets',
        'Nmax': 10,
    }

    itype = pars['intensity']['type']
    print(f'Starting {N} likelihood evals for intensity type = {itype}')
    shapelet_times = np.zeros(N)
    for i in range(N):
        theta = true_theta + \
                scale * np.random.rand(len(true_theta)) - \
                scale/2.
        start = time.time()
        log_likelihood(theta, datacube, pars)
        shapelet_times[i] = 1000.*(time.time() - start)

    plt.hist(exp_times, ec='k', label='inclined_exp')
    plt.hist(shapelet_times, ec='k', label='inclined_exp')
    plt.yscale('log')
    plt.xlabel('Log likelihood time (ms)')
    plt.ylabel('Counts')
    plt.show()

    return 0

if __name__ == '__main__':

    print('Starting tests')
    rc = main()

    if rc == 0:
        print('All tests ran succesfully')
    else:
        print(f'Tests failed with return code of {rc}')
