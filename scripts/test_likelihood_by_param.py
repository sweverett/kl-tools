import numpy as np
import os
import galsim as gs
from multiprocessing import Pool
from astropy.units import Unit
import astropy.constants as const
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from copy import deepcopy
import pickle

import kl_tools.velocity as velocity
from kl_tools.likelihood import LogPosterior
from kl_tools.parameters import Pars
import kl_tools.mocks as mocks
import kl_tools.utils as utils

import ipdb
import pudb

def parse_args():
    parser = ArgumentParser()

    parser.add_argument('-N', type=int, default=200,
                        help='Number of grid points')
    parser.add_argument('-run_name', type=str, default=None,
                        help='Name of likelihood slice run')
    parser.add_argument('-imap', type=str, default='inclined_exp',
                        choices=['inclined_exp', 'basis'],
                        help='Name of likelihood slice run')
    parser.add_argument('--psf', action='store_true',
                        help='Set to use PSF')
    parser.add_argument('--show', action='store_true',
                        help='Set to show plots')

    return parser.parse_args()

def plot_vmap_residuals(theta_pars, datacube, loglike, true_vmap_array,
                        norm=True, s=(22,5), outfile=None, show=False):

    Nx, Ny = datacube.Nx, datacube.Ny
    Nspec = datacube.Nspec

    # create grid of pixel centers in image coords
    X, Y = utils.build_map_grid(Nx, Ny)

    vmap = loglike.setup_vmap(theta_pars)

    v_array = vmap(
        'obs', X, Y, normalized=norm, use_numba=False
        )

    plt.subplot(131)
    plt.imshow(true_vmap_array, origin='lower')
    plt.colorbar(fraction=0.047)
    plt.title('Datacube vmap')

    plt.subplot(132)
    plt.imshow(v_array, origin='lower')
    plt.colorbar(fraction=0.047)
    plt.title('True Model')

    plt.subplot(133)
    plt.imshow(true_vmap_array-v_array, origin='lower',
               cmap='RdBu',
               norm=utils.MidpointNormalize(midpoint=0)
               )
    plt.colorbar(fraction=0.047)
    plt.title('Datacube - Model')

    plt.gcf().set_size_inches(s)

    if outfile is not None:
        plt.savefig(outfile, bbox_inches='tight', dpi=300)

    if show is True:
        plt.show()
    else:
        plt.close()

    return

def plot_imap_residuals(theta_pars, datacube, loglike,
                        s=(30,5), outfile=None, show=False):

    Nspec = datacube.Nspec

    model_pars = theta_pars.copy()

    model = loglike._setup_model(theta_pars, datacube)

    for i in range(datacube.Nspec):
        d = datacube.data[i]
        m = model.data[i]

        lam = datacube.lambdas[i]
        lb, lr = lam[0], lam[1]

        k = i + 1

        plt.subplot(3,Nspec,k)
        plt.imshow(d, origin='lower')
        plt.colorbar(fraction=0.047)
        plt.ylabel('Datacube')
        plt.title(f'({lb:.1f}, {lr:.1f})')

        plt.subplot(3,Nspec,k+Nspec)
        plt.imshow(m, origin='lower')
        plt.colorbar(fraction=0.047)
        plt.ylabel('True Model')

        plt.subplot(3,Nspec,k+2*Nspec)
        plt.imshow(d-m, origin='lower')
        plt.colorbar(fraction=0.047)
        plt.ylabel('Datacube - Model')

    plt.gcf().set_size_inches(s)

    if outfile is not None:
        plt.savefig(outfile, bbox_inches='tight', dpi=300)

    if show is True:
        plt.show()
    else:
        plt.close()

    return

def main(args):

    N = args.N
    run_name = args.run_name
    imap = args.imap
    psf = args.psf
    show = args.show

    outdir = os.path.join(utils.SCRIPT_DIR, 'out', 'test_likelihood_by_param')

    if run_name is not None:
        outdir = os.path.join(outdir, run_name)

    utils.make_dir(outdir)

    # We use the same mock observation for the baseline tests
    # This could be modified to create your own custom observation for the test
    obs = mocks.DefaultMockObservation(imap=imap, psf=psf, s2n=100000)

    velocity_dict = {
        'model': obs.datacube_pars['v_model'],
    }
    if imap == 'inclined_exp':
        intensity_dict = {
            'type': 'inclined_exp',
            'flux': 'sampled',
            'hlr': 'sampled',
        }
    elif imap == 'basis':
        # For most tests, you want to use an identical basis model as the datacube generation. You can of course use a different basis model if testing model misspecification
        intensity_dict = {
            'type': 'basis',
            'basis_type': obs.datacube_pars['basis_type'],
            'basis_kwargs': obs.datacube_pars['basis_kwargs'],
        }

    mcmc_pars = {
        'units': {
            'v_unit': Unit('km / s'),
            'r_unit': Unit('kpc'),
        },
        'priors': {},
        'intensity': intensity_dict,
        'velocity': velocity_dict,
        # 'marginalize_intensity': True,
        'run_options': {
            'use_numba': False,
            }
    }

    print('Setting up test datacube and true Halpha image')
    datacube = obs.datacube
    true_vmap = obs.true_vmap
    true_im = obs.true_im
    Nspec, Nx, Ny = datacube.shape
    lambdas = datacube.lambdas

    if psf is True:
        datacube.set_psf(obs.psf)

    outfile = os.path.join(outdir, 'true_pars.pkl')
    print(f'Saving true pars to {outfile}')
    with open(outfile, 'wb') as out:
        pickle.dump(obs.true_pars, out)

    outfile = os.path.join(outdir, 'datacube.fits')
    print(f'Saving test datacube to {outfile}')
    datacube.write(outfile)
    outfile = os.path.join(outdir, 'datacube.pkl')
    with open(outfile, 'wb') as out:
        pickle.dump(datacube, out)

    outfile = os.path.join(outdir, 'true_imap.npy')
    print(f'Saving test datacube true imap to {outfile}')
    np.save(outfile, true_im)

    outfile = os.path.join(outdir, 'vmap.png')
    print(f'Saving true vamp in obs plane to {outfile}')
    plt.imshow(true_vmap, origin='lower')
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
        l1 = lambdas[i][0]
        l2 = lambdas[i][1]
        plt.title(f'Test datacube slice\n lambda=({l1:.1f}, {l2:.1f}')
        k += 1
    plt.gcf().set_size_inches(9,9)
    plt.tight_layout()
    plt.savefig(outfile, bbox_inches='tight', dpi=300)
    if show is True:
        plt.show()
    else:
        plt.close()

    #-----------------------------------------------------------------
    # Setup sampled posterior

    sampled_pars = list(obs.true_pars)

    pars = Pars(sampled_pars, mcmc_pars)
    pars_order = pars.sampled.pars_order

    log_posterior = LogPosterior(pars, datacube, likelihood='datacube')
    log_likelihood = log_posterior.log_likelihood

    outfile = os.path.join(outdir, 'pars.pkl')
    print(f'Saving Pars to {outfile}')
    with open(outfile, 'wb') as out:
        pickle.dump(pars, out)

    #-----------------------------------------------------------------
    # Plot log likelihood slices
    # 
    # Center each slice around truth, with ranges of +/- d[var]

    t = obs.true_pars
    dg = 0.1
    dtheta = np.pi/32
    dsini = 0.01
    dv0 = 0.05
    dvcirc = 1
    drscale = 0.05
    test_pars = {
        'g1': (t['g1']-dg, t['g1']+dg, dg/N),
        'g2': (t['g2']-dg, t['g2']+dg, dg/N),
        'theta_int': (t['theta_int']-dtheta, t['theta_int']+dtheta, dtheta/N),
        'sini': (t['sini']-dsini, t['sini']+dsini, dsini/N),
        'v0': (t['v0']-dv0, t['v0']+dv0, dv0/N),
        'vcirc': (t['vcirc']-dvcirc, t['vcirc']+dvcirc, dvcirc/N),
        'rscale': (t['rscale']-drscale, t['rscale']+drscale, drscale/N),
    }

    # plot residuals for true log likelihood
    # cov = datacube_pars['sky_sigma']
    outfile = os.path.join(outdir, 'true-imap-residuals.png')
    print(f'Saving true imap residuals to {outfile}')
    plot_imap_residuals(
        obs.true_pars, datacube, log_likelihood, outfile=outfile, show=show
        )

    outfile = os.path.join(outdir, 'true-vmap-residuals.png')
    print(f'Saving true vmap residuals to {outfile}')
    # normalize datacube vmap
    norm = 1. / const.c.to(mcmc_pars['units']['v_unit']).value
    plot_vmap_residuals(
        obs.true_pars, datacube, log_likelihood, norm*true_vmap,
        outfile=outfile, show=show
        )

    # Compute best-fit log likelihood
    theta_true = pars.pars2theta(obs.true_pars)
    true_loglike = log_likelihood(theta_true, datacube)

    # NOTE: Just for testing, can remove later
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
        start = f'{par_range[0]:.3f}'
        end = f'{par_range[1]:.3f}'
        delta = f'{par_range[2]:.3f}'
        print(f'Starting loop over {par}: ({start}, {end}; {delta})')

        # fresh copy
        theta_pars = deepcopy(obs.true_pars)

        # Now update w/ test param
        left, right, dx = par_range
        assert right > left
        # N = int((right - left) / dx)
        loglike = np.zeros(N)
        par_val = np.zeros(N)

        for i, val in enumerate(np.linspace(
                left, right, num=N, endpoint=True
                )):
            theta_pars[par] = val
            theta = pars.pars2theta(theta_pars)

            loglike[i] = log_likelihood(theta, datacube)
            par_val[i] = val

        plt.subplot(nrows, ncols, k)
        plt.plot(par_val, loglike)
        truth = obs.true_pars[par]
        plt.axvline(truth, lw=2, c='k', ls='--', label='Truth')
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

    #-----------------------------------------------------------------
    # Check how a given par offsets affect vmap residuals

    theta_pars = obs.true_pars.copy()

    var = 'g1'
    print(f'Making plots of vmap residuals while varying {var}')
    utils.make_dir(os.path.join(outdir, var))
    vmin, vmax = test_pars[var][0], test_pars[var][1]
    vals = np.linspace(vmin, vmax, 11, endpoint=True)
    for val in vals:
        theta_pars[var] = val
        theta = pars.pars2theta(theta_pars)

        outfile = os.path.join(outdir, var, f'{val:.4f}-{var}-vmap-residuals.png')
        print(f'Saving {var}={val:.4f} vmap residuals to {outfile}')

        # norm = const.c.to(mcmc_pars['units']['v_unit']).value
        plot_vmap_residuals(
            theta_pars, datacube, log_likelihood, true_vmap, norm=False,
            outfile=outfile, show=show
        )

    return 0

if __name__ == '__main__':

    args = parse_args()

    print('Starting tests')
    rc = main(args)

    if rc == 0:
        print('All tests ran succesfully')
    else:
        print(f'Tests failed with return code of {rc}')
