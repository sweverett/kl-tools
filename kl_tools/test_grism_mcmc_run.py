# This is to ensure that numpy doesn't have
# OpenMP optimizations clobber our own multiprocessing
import os
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
import sys
import pickle
import schwimmbad
import mpi4py
from schwimmbad import MPIPool
from argparse import ArgumentParser
from astropy.units import Unit
import galsim as gs
import matplotlib.pyplot as plt
import zeus

import utils
from mcmc import KLensZeusRunner, KLensEmceeRunner
import priors
from muse import MuseDataCube
import likelihood
from parameters import Pars
from likelihood import LogPosterior
from velocity import VelocityMap

import ipdb

parser = ArgumentParser()

parser.add_argument('nsteps', type=int,
                    help='Number of mcmc iterations per walker')
parser.add_argument('-sampler', type=str, choices=['zeus', 'emcee'],
                    default='emcee',
                    help='Which sampler to use for mcmc')
parser.add_argument('-run_name', type=str, default='',
                    help='Name of mcmc run')
parser.add_argument('--show', action='store_true', default=False,
                    help='Set to show test plots')

group = parser.add_mutually_exclusive_group()
group.add_argument('-ncores', default=1, type=int,
                    help='Number of processes (uses `multiprocessing`)')
group.add_argument('--mpi', dest='mpi', default=False, action='store_true',
                   help='Run with MPI.')

def main(args, pool):

    # TODO: Try making datacube a global variable, as it may significantly
    # decrease execution time due to pickling only once vs. every model eval

    nsteps = args.nsteps
    sampler = args.sampler
    ncores = args.ncores
    mpi = args.mpi
    run_name = args.run_name
    show = args.show

    outdir = os.path.join(
        utils.TEST_DIR, 'grism-mcmc-run', run_name
        )
    utils.make_dir(outdir)

    sampled_pars = [
        'g1',
        'g2',
        'theta_int',
        'sini',
        'v0',
        'vcirc',
        'rscale',
        ]

    # additional args needed for prior / likelihood evaluation
    meta_pars = {
        'units': {
            'v_unit': Unit('km/s'),
            'r_unit': Unit('kpc')
            },
        'priors': {
            'g1': priors.GaussPrior(0., 0.1, clip_sigmas=2),
            'g2': priors.GaussPrior(0., 0.1, clip_sigmas=2),
            'theta_int': priors.UniformPrior(0., 2.*np.pi),
            # 'theta_int': priors.UniformPrior(np.pi/3, np.pi),
            'sini': priors.UniformPrior(0, 1.),
            'v0': priors.UniformPrior(0, 20),
            # 'vcirc': priors.GaussPrior(200, 20, zero_boundary='positive'),# clip_sigmas=2),
            'vcirc': priors.UniformPrior(0, 800),
            # 'vcirc': priors.GaussPrior(188, 2.5, zero_boundary='positive', clip_sigmas=2),
            # 'vcirc': priors.UniformPrior(190, 210),
            'rscale': priors.UniformPrior(0, 40),
            #'x0': priors.GaussPrior(0, 5),
            #'y0': priors.GaussPrior(0, 5),
            #'z': priors.GaussPrior(0.2466, .005),
            #'R': priors.GaussPrior(3200, 50, clip_sigmas=3),
            # 'beta': priors.UniformPrior(0, .2),
            # 'hlr': priors.UniformPrior(0, 8),
            # 'flux': priors.UniformPrior(5e3, 7e4),
            },
        'velocity': {
            'model': 'default'
        },
        'intensity': {
            # For this test, use truth info
            'type': 'inclined_exp',
            'flux': 1.0, # counts
            'hlr': 0.5,
            # 'flux': 'sampled', # counts
            # 'hlr': 'sampled', # pixels
            # 'type': 'inclined_exp',
            # 'basis_type': 'shapelets',
            # 'basis_type': 'sersiclets',
            #'basis_type': 'exp_shapelets',
            #'basis_kwargs': {
            #    'use_continuum_template': True,
            #    'Nmax': 7,
            #     # 'plane': 'disk',
            #    'plane': 'obs',
            #    'beta': 0.17,
                # 'beta': 'sampled',
            #     # 'index': 1,
            #     # 'b': 1,
            #    }
            },
        # 'marginalize_intensity': True,
        'psf': gs.Gaussian(fwhm=0.13),
        'run_options': {
            #'remove_continuum': True,
            'use_numba': False
            }
    }

    cube_dir = os.path.join(utils.TEST_DIR, 'test_data')

    ### replace these information with dict
    # SED is provided by the `specfile` for muse version. For grism,
    # we may consider 1d reduced spectrum as an initial guess. But how would you like to summarize the information? Go with dict now...
    cubefile = os.path.join(cube_dir, '')
    #specfile = os.path.join(cube_dir, '')
    specpars = {
        'template' : '../data/Simulation/GSB2.spec',
        'wave_type' : 'Ang',
        'flux_type' : 'flambda',
        'z' : 0.9513,
        'wave_range' : [500., 3000.], # nm
        # obs-frame continuum normalization (nm, erg/s/cm2/nm)
        'obs_cont_norm' : [614, 2.6e-17],
        # a dict of line names and obs-frame flux values (erg/s/cm2)
        'lines': {
            'Halpha' : 1.25e-16,
            'OII' : [1.0e-15, 1.2e-15],
            'OIII' : [1.0e-15, 1.2e-15],
        }
        # intrinsic linewidth in nm
        'line_sigma_int' :{
            'Halpha' : 0.5,
            'OII' : [0.2, 0.2],
            'OIII' : [0.2, 0.2],
        }
    }
    #catfile = os.path.join(cube_dir, '')
    # observation parameters
    # TODO: encode these pars into datavector fits header etc.
    obspars = {
        'number_of_observations': 3,
        'obs_1' : {
            # HST WFC3/IR G141 observation, roll angle 1
            'inst_name': 'HST/WFC3',
            'type' : 'grism',
            'bandpass': '../data/Bandpass/HST/WFC3_IR_G141_1st.dat',
            'Nx': 38,
            'Ny': 38,
            'pixel_scale': 0.065, # arcsec
            'R_spec': 215, # at 1 micron
            # can be 'airy'/'moffat'/'kolmogorov'/'vonkarman'/'opticalpsf'
            'psf_type': 'airy_mean',
            # pass the needed params to build PSF model here
            # in case of airy, we don't need any params
            'psf_kwargs':{
                'fwhm': 0.13, # arcsec
            },
            'disp_ang': 0.0, # radian
            'offset': -550.96322, # pix (-275.48161224045805 )
            'diameter': 240, # cm
            'exp_time': 5000., # seconds
            'gain': 1., # electron per ADU
            'noise':{
                'type': 'ccd',
                'sky_level': 0.1, # 1.0 electron per sec per pix
                'read_noise': 20, # 20 electron per pix per readout, x2
                'apply_to_data': False,
            }
        }
        'obs_2': {# HST WFC3/IR G141 observation, roll angle 2
            'inst_name': 'HST/WFC3',
            'type': 'grism',
            'bandpass': '../data/Bandpass/HST/WFC3_IR_G141_1st.dat',
            'Nx': 38,              # number of pixels
            'Ny': 38,
            'pixel_scale': 0.065,      # arcsec
            'R_spec':  215,          # at 1 micron
            'psf_type': 'airy_mean',
            'psf_kwargs':{ 
                'fwhm': 0.13,     # arcsec
            },
            'disp_ang': 1.57,  # radian
            'offset': -550.96322, #pix
            'diameter': 240, # cm
            'exp_time': 5000., # seconds
            'gain': 1.,
            'noise':{
                'type': 'ccd',
                'sky_level': 0.1,
                'read_noise': 20,
                'apply_to_data': False,
            }
        },
        'obs_3': { # HST WFC3/IR image observation, F125W
            'inst_name': 'HST/WFC3',
            'type': 'photometry',
            'bandpass': '../data/Bandpass/HST/WFC3_IR_F125W.dat',
            'Nx': 38, # number of pixels
            'Ny': 38,
            'pixel_scale': 0.065, # arcsec
            'psf_type': 'airy',
            'psf_kwargs':{
                'fwhm': 0.13, # arcsec
            },
            'diameter': 240, # cm
            'exp_time': 800.0, # seconds
            'gain': 1.0,
            'noise':{
                'type': 'ccd',
                'sky_level': 0.1,
                'read_noise': 20,
                'apply_to_data': False,
            },
        },
    }
    #linefile = os.path.join(cube_dir, '')

    print(f'Setting up Grism datacube from file {cubefile}')
    datacube = GrismDataCube(
        cubefile, #specfile, catfile, linefile
        )

    # default, but we'll make it explicit:
    datacube.set_line(line_choice='strongest')
    Nspec = datacube.Nspec
    lambdas = datacube.lambdas

    print(f'Strongest emission line has {Nspec} slices')

    outfile = os.path.join(outdir, 'datacube-slices.png')
    print(f'Saving example theorycube slice images to {outfile}')
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

    #-----------------------------------------------------------------
    # Setup sampled posterior

    pars = Pars(sampled_pars, meta_pars)
    pars_order = pars.sampled.pars_order

    log_posterior = LogPosterior(pars, datacube, likelihood='grism')

    #-----------------------------------------------------------------
    # Setup sampler

    ndims = log_posterior.ndims
    nwalkers = 2*ndims

    if sampler == 'zeus':
        print('Setting up KLensZeusRunner')

        runner = KLensZeusRunner(
            nwalkers, ndims, log_posterior, datacube, pars
            )

    elif sampler == 'emcee':
        print('Setting up KLensEmceeRunner')

        runner = KLensEmceeRunner(
            nwalkers, ndims, log_posterior, datacube, pars
            )

    print('Starting mcmc run')
    runner.run(nsteps, pool)

    runner.burn_in = nsteps // 2

    if (sampler == 'zeus') and ((ncores > 1) or (mpi == True)):
        # The sampler isn't pickleable for some reason in this scenario
        # Save whole chain
        outfile = os.path.join(outdir, 'test-mcmc-chain.pkl')
        chain = runner.sampler.get_chain(flat=True)
        print(f'pickling chain to {outfile}')
        with open(outfile, 'wb') as f:
            pickle.dump(chain, f)
    else:
        outfile = os.path.join(outdir, 'test-mcmc-sampler.pkl')
        print(f'Pickling sampler to {outfile}')
        with open(outfile, 'wb') as f:
            pickle.dump(runner.sampler, f)

        outfile = os.path.join(outdir, 'test-mcmc-runner.pkl')
        print(f'Pickling runner to {outfile}')
        with open(outfile, 'wb') as f:
            pickle.dump(runner, f)

    outfile = os.path.join(outdir, 'chains.png')
    print(f'Saving chain plots to {outfile}')
    runner.plot_chains(
        outfile=outfile, show=show
        )

    runner.compute_MAP()
    map_medians = runner.MAP_medians
    print('(median) MAP values:')
    for name, indx in pars_order.items():
        m = map_medians[indx]
        print(f'{name}: {m:.4f}')

    outfile = os.path.join(outdir, 'compare-data-to-map.png')
    print(f'Plotting MAP comparison to data in {outfile}')
    runner.compare_MAP_to_data(outfile=outfile, show=show)

    outfile = os.path.join(outdir, 'corner-map.png')
    print(f'Saving corner plot compare to MAP in {outfile}')
    title = 'Reference lines are param MAP values'
    runner.plot_corner(
        outfile=outfile, reference=runner.MAP_medians, title=title, show=show
        )

    if sampler == 'emcee':
        blobs = runner.sampler.blobs
    elif sampler == 'zeus':
        blobs = runner.sampler.get_blobs()

    outfile = os.path.join(outdir, 'chain-blob.pkl')
    print(f'Saving prior & likelihood values to {outfile}')
    data = {
        'prior': blobs[:,:,0],
        'likelihood': blobs[:,:,1],
        # TODO: generalize or remove after debugging!
        # 'image': blobs[:,:,2],
        # 'cont_template': blobs[:,:,3],
        # 'mle_coeff': blobs[:,:,4],
    }
    with open(outfile, 'wb') as f:
        pickle.dump(data, f)

    outfile = os.path.join(outdir, 'chain-probabilities.png')
    print(f'Saving prior & likelihood value plot to {outfile}')
    prior = blobs[:,indx,0]
    like = blobs[:,indx,1]
    indx = np.random.randint(0, high=nwalkers)
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 4))
    plt.subplot(131)
    plt.plot(prior, label='prior', c='tab:blue')
    plt.xlabel('Sample')
    plt.ylabel('Log probability')
    plt.legend()
    plt.subplot(132)
    plt.plot(like, label='likelihood', c='tab:orange')
    plt.xlabel('Sample')
    plt.ylabel('Log probability')
    plt.legend()
    plt.subplot(133)
    plt.plot(prior, label='prior', c='tab:blue')
    plt.plot(like, label='likelihood', c='tab:orange')
    plt.xlabel('Sample')
    plt.ylabel('Log probability')
    plt.legend()

    plt.savefig(outfile, bbox_inches='tight', dpi=300)

    if show is True:
        plt.show()
    else:
        plt.close()

    return 0

if __name__ == '__main__':
    args = parser.parse_args()

    pool = schwimmbad.choose_pool(
        mpi=args.mpi, processes=args.ncores
        )

    if isinstance(pool, MPIPool):
        if not pool.is_master():
            pool.wait()
            sys.exit(0)

    print('Starting tests')
    rc = main(args, pool)

    if rc == 0:
        print('All tests ran succesfully')
    else:
        print(f'Tests failed with return code of {rc}')
