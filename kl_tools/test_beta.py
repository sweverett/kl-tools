import numpy as np
import os, sys
import galsim as gs
from galsim.angle import Angle, radians
import astropy.units as u
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from copy import deepcopy
from argparse import ArgumentParser
from multiprocessing import Pool

from velocity import VelocityMap
import transformation as transform
import likelihood, cube, priors, utils, basis, parameters
from muse import MuseDataCube

import ipdb

def parse_args():
    parser = ArgumentParser()

    parser.add_argument('Nbeta', type=int,
                        help='The number of betas to scan')
    parser.add_argument('-ncores', type=int, default=1,
                        help='The number of cpu cores to use')
    parser.add_argument('-bmin', type=float, default=0.001,
                        help='The minimum value of beta to scan')
    parser.add_argument('-bmax', type=float, default=3,
                        help='The maximum value of beta to scan')
    parser.add_argument('-plane', type=str, default='obs',
                        help='The KL plane to define the basis in')
    parser.add_argument('--show', action='store_true', default=False,
                        help='Show plots')
    parser.add_argument('--vb', action='store_true', default=False,
                        help='Make verbose')

    return parser.parse_args()

def make_basis_imap(theta_pars, datacube, pars):
    imap = likelihood.DataCubeLikelihood._setup_imap(
        theta_pars, datacube, pars
    )

    im = imap.render(
        theta_pars, datacube, pars
                   )
    mle = imap.fitter.mle_coefficients

    return im, mle, imap

def fit_one_beta(i, beta, theta, datacube,
                 pars_shapelet, pars_sersiclet, pars_exp_shapelet,
                 sky=1., vb=False):

    if (vb is True) and (i % 10 == 0):
        print(f'Fitting beta {i}')

    pars_shapelet['intensity']['basis_kwargs']['beta'] = beta
    pars_sersiclet['intensity']['basis_kwargs']['beta'] = beta
    pars_exp_shapelet['intensity']['basis_kwargs']['beta'] = beta

    shapelet, mle_shapelet, imap_shapelet = make_basis_imap(
        theta, datacube, pars_shapelet
        )
    sersiclet, mle_sersiclet, imap_sersiclet = make_basis_imap(
        theta, datacube, pars_sersiclet
        )
    exp_shapelet, mle_exp_shapelet, imap_exp_shapelet = make_basis_imap(
        theta, datacube, pars_exp_shapelet
        )

    data = datacube.stack()
    Npix = data.shape[0] * data.shape[1]

    chi2_shapelet = np.sum((shapelet-data)**2/sky**2) / Npix
    chi2_sersiclet = np.sum((sersiclet-data)**2/sky**2) / Npix
    chi2_exp_shapelet = np.sum((exp_shapelet-data)**2/sky**2) / Npix

    return np.array([chi2_shapelet, chi2_sersiclet, chi2_exp_shapelet])

def stack(chi_list):
    '''
    Stack the chi tuple result of each fit_one_beta() call
    '''

    chi_array = np.array(chi_list)

    chi2_shapelet = chi_array[:,0]
    chi2_sersiclet = chi_array[:,1]
    chi2_exp_shapelet = chi_array[:,2]

    return (chi2_shapelet, chi2_sersiclet, chi2_exp_shapelet)

def main(args):

    Nbeta = args.Nbeta
    ncores = args.ncores
    plane = args.plane
    bmin = args.bmin
    bmax = args.bmax
    show = args.show
    vb = args.vb

    outdir = os.path.join(utils.TEST_DIR, 'test-beta')
    utils.make_dir(outdir)

    #-------------------------------------------------------------
    # setup basic pars

    sampled_pars = [
        'g1',
        'g2',
        'theta_int',
        'sini',
        'v0',
        'vcirc',
        'rscale',
        'beta'
        ]

    meta_pars = {
        'units': {
            'v_unit': u.Unit('km/s'),
            'r_unit': u.Unit('kpc')
        },
        'velocity': {
            'model': 'offset'
        },
        'run_options': {
            'remove_continuum': True,
            'use_numba': False
            }
    }

    #-------------------------------------------------------------
    # setup imap basis pars

    pars_shapelet = meta_pars.copy()
    pars_sersiclet = meta_pars.copy()
    pars_exp_shapelet = meta_pars.copy()

    nmax_cart  = 14
    nmax_polar = 12
    pars_shapelet['intensity'] = {
        'type': 'basis',
        'basis_type': 'shapelets',
        'basis_kwargs': {
            'use_continuum_template': True,
            'Nmax': nmax_cart,
        }
    }
    pars_sersiclet['intensity'] = {
        'type': 'basis',
        'basis_type': 'sersiclets',
        'basis_kwargs': {
            'use_continuum_template': True,
            'Nmax': nmax_polar,
            'index': 1,
            'b': 1,
        }
    }
    pars_exp_shapelet['intensity'] = {
        'type': 'basis',
        'basis_type': 'exp_shapelets',
        'basis_kwargs': {
            'use_continuum_template': True,
            'Nmax': nmax_polar,
        }
    }

    for p in [pars_shapelet, pars_sersiclet, pars_exp_shapelet]:
        p['intensity']['basis_kwargs']['plane'] = plane

    Pars = parameters.Pars(sampled_pars, meta_pars)
    pars_shapelet = parameters.MCMCPars(pars_shapelet)
    pars_sersiclet = parameters.MCMCPars(pars_sersiclet)
    pars_exp_shapelet = parameters.MCMCPars(pars_exp_shapelet)

    # Setup MUSE DataCube
    cube_dir = os.path.join(utils.TEST_DIR, 'test_data')

    cubefile = os.path.join(cube_dir, '102021103_objcube.fits')
    specfile = os.path.join(cube_dir, 'spectrum_102021103.fits')
    catfile = os.path.join(cube_dir, 'MW_1-24_main_table.fits')
    linefile = os.path.join(cube_dir, 'MW_1-24_emline_table.fits')

    #-------------------------------------------------------------
    # setup muse datacube

    print(f'Setting up MUSE datacube from file {cubefile}')
    datacube = MuseDataCube(
        cubefile, specfile, catfile, linefile
    )

    # default, but we'll make it explicit:
    datacube.set_line(line_choice='strongest')
    Nspec = datacube.Nspec
    lambdas = datacube.lambdas

    psf = gs.Gaussian(fwhm=0.7, flux=1.)
    datacube.set_psf(psf)

    data = datacube.stack()
    sky = np.std(data[:16,:16])

    #-------------------------------------------------------------
    # scan over beta values for each imap basis type

    # make dummy sample
    theta = np.random.rand(len(sampled_pars))

    betas = np.linspace(bmin, bmax, Nbeta)

    pars_shapelet_b = deepcopy(pars_shapelet)
    pars_sersiclet_b = deepcopy(pars_sersiclet)
    pars_exp_shapelet_b = deepcopy(pars_exp_shapelet)

    with Pool(ncores) as pool:
        out = stack(pool.starmap(fit_one_beta,
                                 [(i,
                                   beta,
                                   theta,
                                   datacube,
                                   pars_shapelet_b,
                                   pars_sersiclet_b,
                                   pars_exp_shapelet_b,
                                   sky,
                                   vb
                                   )
                                  for i, beta in enumerate(betas)]
                                 )
                    )

    chi2_shapelet = out[0]
    chi2_sersiclet = out[1]
    chi2_exp_shapelet = out[2]

    # now find the best fits
    shapelet_min = betas[np.argmin(chi2_shapelet)]
    sersiclet_min = betas[np.argmin(chi2_sersiclet)]
    exp_shapelet_min = betas[np.argmin(chi2_exp_shapelet)]

    shapelet_min_chi2 = np.min(chi2_shapelet)
    sersiclet_min_chi2 = np.min(chi2_sersiclet)
    exp_shapelet_min_chi2 = np.min(chi2_exp_shapelet)

    #-------------------------------------------------------------
    # Plot beta scans

    plt.plot(betas, chi2_shapelet,
             label=f'shapelet (beta_min={shapelet_min:.2f}; ' +\
                   f'$\chi^2$={shapelet_min_chi2:.1f})')
    plt.plot(betas, chi2_sersiclet,
             label=f'sersiclet (beta_min={sersiclet_min:.2f}; ' +\
                   f'$\chi^2$={sersiclet_min_chi2:.1f})')
    plt.plot(betas, chi2_exp_shapelet,
             label=f'exp_shapelet (beta_min={exp_shapelet_min:.2f}; ' +\
                   f'$\chi^2$={exp_shapelet_min_chi2:.1f})')
    plt.legend()
    plt.xlabel(r'$\beta$')
    plt.ylabel(r'$\chi^2$ (arbitrary units)')
    plt.yscale('log')
    plt.axhline(0, lw=2, c='k')

    # this is just to get the num of basis funcs
    _, mle_shapelet, _ = make_basis_imap(
        theta, datacube, pars_shapelet_b
    )
    _, mle_sersiclet, _ = make_basis_imap(
        theta, datacube, pars_sersiclet_b
    )
    _, mle_exp_shapelet, _ = make_basis_imap(
        theta, datacube, pars_exp_shapelet_b
    )

    nfuncs = (len(mle_shapelet), len(mle_sersiclet), len(mle_exp_shapelet))
    del mle_shapelet
    del mle_sersiclet
    del mle_exp_shapelet
    plt.title(rf'{plane}-basis $\chi^2$ dependence on $\beta$; N={nfuncs}')

    plt.gcf().patch.set_facecolor('w')
    plt.gcf().set_size_inches(8,6)

    outfile = os.path.join(outdir, 'beta-scan.png')
    if vb is True:
        print(f'Saving result to {outfile}...')
    plt.savefig(outfile, bbox_inches='tight', dpi=300)

    if show is True:
        plt.show()
    else:
        plt.close()

    #-------------------------------------------------------------
    # Plot image residuals for best fit

    data = datacube.stack()
    Npix = data.shape[0] * data.shape[1]

    pars_shapelet['beta'] = shapelet_min
    pars_sersiclet['beta'] = sersiclet_min
    pars_exp_shapelet['beta'] = exp_shapelet_min

    if vb is True:
        print(f'min shapelet beta: {shapelet_min}')
        print(f'min sersiclet beta: {sersiclet_min}')
        print(f'min exp_shapelet beta: {exp_shapelet_min}')

    shapelet, mle_shapelet, imap_shapelet = make_basis_imap(
        theta, datacube, pars_shapelet
    )
    sersiclet, mle_sersiclet, imap_sersiclet = make_basis_imap(
        theta, datacube, pars_sersiclet
    )
    exp_shapelet, mle_exp_shapelet, imap_exp_shapelet = make_basis_imap(
        theta, datacube, pars_exp_shapelet
    )

    plt.subplot(241)
    im = plt.imshow(data, origin='lower')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title('stacked data')

    plt.subplot(242)
    im = plt.imshow(shapelet, origin='lower')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title('shapelet fit')

    plt.subplot(243)
    im = plt.imshow(sersiclet, origin='lower')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title('sersiclet fit')

    plt.subplot(244)
    im = plt.imshow(exp_shapelet, origin='lower')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title('exp_shapelet fit')

    plt.subplot(246)
    im = plt.imshow(shapelet-data, origin='lower')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title('shapelet-data')

    plt.subplot(247)
    im = plt.imshow(sersiclet-data, origin='lower')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title('sersiclet-data')

    plt.subplot(248)
    im = plt.imshow(exp_shapelet-data, origin='lower')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title('exp_shapelet-data')

    plt.gcf().set_size_inches(16,8)

    outfile = os.path.join(outdir, 'im-residuals.png')
    if vb is True:
        print(f'Saving result to {outfile}...')
    plt.savefig(outfile, bbox_inches='tight', dpi=300)

    if show is True:
        plt.show()
    else:
        plt.close()

    #-------------------------------------------------------------
    # Plot basis components for fit

    outfile = os.path.join(outdir, 'shapelet-basis-fit.png')
    if vb is True:
        print(f'Plotting basis components for shapelet fit to {outfile}')
    plot_basis_components(
        'shapelet', imap_shapelet, outfile=outfile, show=show
        )

    outfile = os.path.join(outdir, 'sersiclet-basis-fit.png')
    if vb is True:
        print(f'Plotting basis components for sersiclet fit to {outfile}')
    plot_basis_components(
        'sersiclet', imap_sersiclet, outfile=outfile, show=show
        )

    outfile = os.path.join(outdir, 'exp_shapelet-basis-fit.png')
    if vb is True:
        print(f'Plotting basis components for exp_shapelet fit to {outfile}')
    plot_basis_components(
        'exp_shapelet', imap_exp_shapelet, outfile=outfile, show=show
        )

    return 0

def plot_basis_components(name, imap, N=25, outfile=None, show=False,
                          size=(9,9)):
    '''
    for now, just plot the first N funcs
    '''

    mle_coeff = imap.fitter.mle_coefficients
    basis = imap.fitter.basis
    im_nx, im_ny = imap.nx, imap.ny

    X, Y = utils.build_map_grid(im_nx, im_ny)

    sqrt = int(np.ceil(np.sqrt(N)))

    fig, axes = plt.subplots(
        nrows=sqrt, ncols=sqrt, sharex=True, sharey=True, figsize=size
        )

    for n in range(N):
        i = n // sqrt
        j = n % sqrt
        ax = axes[i,j]

        bfunc = basis.get_basis_func(n, X, Y)
        image = (mle_coeff[n] * bfunc).real
        im = ax.imshow(image, origin='lower')

        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)
        ax.set_title(f'n={n}')

    plt.subplots_adjust(hspace=0.15, wspace=0.5)
    plt.suptitle(f'fitted {name} basis functions', y=0.95)

    if outfile is not None:
        plt.savefig(outfile, bbox_inches='tight', dpi=300)

    if show is True:
        plt.show()
    else:
        plt.close()

    return

if __name__ == '__main__':
    args = parse_args()

    rc = main(args)

    if rc == 0:
        print('\nTests have completed without errors')
    else:
        print(f'Tests failed with return code of {rc}')
