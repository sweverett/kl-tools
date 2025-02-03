import numpy as np
import os
from astropy.units import Unit
import astropy.constants as constants
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import utils
import intensity
import priors
import likelihood
from parameters import PARS_ORDER, theta2pars, pars2theta
from velocity import VelocityMap
from cube import DataCube

import pudb

parser = ArgumentParser()

parser.add_argument('--show', action='store_true', default=False,
                    help='Set to show test plots')

def main(args):

    show = args.show

    outdir = os.path.join(utils.TEST_DIR, 'test-true-model')
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
        # 'cov_sigma': 3, # pixel counts; dummy value
        'cov_sigma': 0.05, # pixel counts; dummy value
        'bandpass_throughput': '.2',
        'bandpass_unit': 'nm',
        'bandpass_zp': 30,
        'priors': {
            'g1': priors.GaussPrior(0., 0.1),#, clip_sigmas=2),
            'g2': priors.GaussPrior(0., 0.1),#, clip_sigmas=2),
            'theta_int': priors.UniformPrior(0., np.pi),
            # 'theta_int': priors.UniformPrior(np.pi/3, np.pi),
            'sini': priors.UniformPrior(-1., 1.),
            'v0': priors.UniformPrior(0, 20),
            'vcirc': priors.GaussPrior(200, 10),# clip_sigmas=2),
            # 'vcirc': priors.UniformPrior(190, 210),
            'rscale': priors.UniformPrior(0, 10),
        },
        'intensity': {
            # For this test, use truth info
            'type': 'inclined_exp',
            'flux': 1e5, # counts
            'hlr': 5, # pixels
            # 'type': 'basis',
            # 'basis_type': 'shapelets',
            # 'basis_kwargs': {
            #     'Nmax': 12,
            #     'plane': 'disk'
            #     }
        },
        # 'marginalize_intensity': True,
        # 'psf': gs.Gaussian(fwhm=3), # fwhm in pixels
        'use_numba': False,
    }

    # li, le, dl = 656, 656.5, 0.1
    li, le, dl = 655.8, 656.8, 0.1
    # li, le, dl = 655.4, 657.1, 0.1
    lambdas = [(l, l+dl) for l in np.arange(li, le, dl)]

    Nx, Ny = 30, 30
    Nspec = len(lambdas)
    shape = (Nx, Ny, Nspec)
    print('Setting up test datacube and true Halpha image')
    datacube, sed, true_vmap, true_im = likelihood.setup_likelihood_test(
        true_pars, pars, shape, lambdas
        )

    pars['sed'] = sed

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
    plt.imshow(true_vmap, origin='lower')
    plt.colorbar(label='v')
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

    #-------------------------------------------------------
    # compare with true imap + vmap

    # setup true vmap
    vmap_pars = {**true_pars}
    units = ['v_unit', 'r_unit']
    for unit in units:
        vmap_pars[unit] =pars[unit]
    Vmap = VelocityMap('default', vmap_pars)

    X, Y = utils.build_map_grid(Nx, Ny)
    vmap = Vmap('obs', X, Y)

    outfile = os.path.join(outdir, 'compare-vmaps.png')
    print(f'Saving vmap comparison to {outfile}')
    fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True,
                             figsize=(9,9))
    maps = [vmap, true_vmap, vmap-true_vmap, 100.*(vmap-true_vmap)/true_vmap]
    titles = ['Using true pars', 'Used to create datacube', 'Residual', '% Residual']

    for i in range(4):
        ax = axes[i//2, i%2]
        im = ax.pcolormesh(X, Y, maps[i])
        s = np.sum(maps[i])
        ax.set_title(f'{titles[i]}; sum={s:.1f}')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)

    plt.tight_layout()

    plt.savefig(outfile, bbox_inches='tight', dpi=300)

    if show is True:
        plt.show()
    else:
        plt.close()

    # setup true imap
    Imap = intensity.InclinedExponential(
        datacube, flux=pars['true_flux'], hlr=pars['true_hlr']
        )
    imap = Imap.render(true_pars, datacube, pars)

    outfile = os.path.join(outdir, 'compare-imaps.png')
    print('')
    fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True,
                             figsize=(9,9))
    maps = [imap, true_im, imap-true_im, 100.*(imap-true_im)/true_im]
    titles = ['Using true pars', 'Used to create datacube', 'Residual', '% Residual']

    for i in range(4):
        ax = axes[i//2, i%2]
        im = ax.pcolormesh(X, Y, maps[i])
        s = np.sum(maps[i])
        ax.set_title(f'{titles[i]}; sum={s:.1f}')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)

    plt.tight_layout()

    plt.savefig(outfile, bbox_inches='tight', dpi=300)

    if show is True:
        plt.show()
    else:
        plt.close()

    #-------------------------------------------------------
    # now compare individual slice models with the datacube

    outfile = os.path.join(outdir, 'compare-slices.png')
    print(f'Saving datacube slice comparison to {outfile}')

    slice_indices = range(Nspec)
    if Nspec > 10:
        slice_indices = np.random.choice(slice_indices, size=10, replace=False)

    N = len(slice_indices)

    fig, axes = plt.subplots(nrows=3, ncols=N, sharex=True, sharey=True,
                             figsize=(16,4))
    for i in range(N):
        indx = slice_indices[i]
        # datacube slice
        ax = axes[0,i]
        dc = datacube.slices[i]._data
        im = ax.imshow(dc, origin='lower')
        l1, l2 = datacube.lambdas[indx]
        ax.set_title(f'({l1:.1f}, {l2:.1f})')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)
        if i == 0:
            ax.set_ylabel('Datacube')

        # model slice
        ax = axes[1,i]
        # TODO: update w/ psf if we want to run this old test again
        model = compute_slice_model((l1, l2), vmap, imap, sed)
        im = ax.imshow(model, origin='lower')
        l1, l2 = datacube.lambdas[i]
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)
        if i == 0:
            ax.set_ylabel('Model')

        # residual
        ax = axes[2,i]
        im = ax.imshow(dc-model, origin='lower')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)
        if i == 0:
            ax.set_ylabel('Data-Model')

    plt.tight_layout()

    plt.savefig(outfile, bbox_inches='tight', dpi=300)

    if show is True:
        plt.show()
    else:
        plt.close()

    return 0

# TODO: update w/ psf if we want to run this old test again
def compute_slice_model(lambdas, vmap, imap, sed):
    '''
    Compute the slice model of the given index
    (lambdas is (l1, l2))
    '''

    vmap_norm = (vmap / constants.c.to('km/s')).value
    zfactor = 1. / (1. + vmap_norm)

    sed_array = np.array([sed.x, sed.y])

    return likelihood._compute_slice_model(
            lambdas, sed_array, zfactor, imap
        )

    return model

if __name__ == '__main__':
    args = parser.parse_args()

    print('Starting tests')
    rc = main(args)

    if rc == 0:
        print('All tests ran succesfully')
    else:
        print(f'Tests failed with return code of {rc}')
