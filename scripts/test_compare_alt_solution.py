import numpy as np
import os
from multiprocessing import Pool
import velocity
import parameters
from astropy.units import Unit
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import velocity
import intensity
import utils
# TODO: update likelihood imports if we want to run this old test again
from likelihood import setup_likelihood_test, log_likelihood, _compute_slice_model
from parameters import pars2theta

import pudb

parser = ArgumentParser()

parser.add_argument('--show', action='store_true',
                    help='Set to show plots')

def main(args):

    show = args.show

    outdir = os.path.join(utils.TEST_DIR, 'test_compare_alt_solution')
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
        'pix_scale': 1, # arcsec / pix
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
        'cov_sigma': 3.0, # pixel counts; dummy value
        'bandpass_throughput': '.2',
        'bandpass_unit': 'nm',
        'bandpass_zp': 30,
        'intensity': {
            # For this test, use truth info
            # 'type': 'inclined_exp',
            # 'flux': 1e5, # counts
            # 'hlr': 5, # pixels
            'type': 'basis',
            'basis_type': 'shapelets',
            'basis_kwargs': {
                'Nmax': 15,
                'plane': 'disk'
                # 'plane': 'obs'
                }
        },
        # 'psf': gs.Gaussian(fwhm=3), # fwhm in pixels
        'use_numba': False,
    }

    li, le, dl = 655.8, 656.8, 0.1
    lambdas = [(l, l+dl) for l in np.arange(li, le, dl)]

    Nx, Ny = pars['Nx'], pars['Ny']
    Nspec = len(lambdas)
    shape = (Nspec, Nx, Ny)

    print('Setting up test datacube and true Halpha image')
    datacube, sed, true_vmap, true_im = setup_likelihood_test(
        true_pars, pars, shape, lambdas
        )

    # update pars w/ SED object
    pars['sed'] = sed

    outfile = os.path.join(outdir, 'datacube.fits')
    print(f'Saving test datacube to {outfile}')
    datacube.write(outfile)

    outfile = os.path.join(outdir, 'true_vmap.png')
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
        l1, l2 = lambdas[i]
        plt.title(f'Test datacube slice\n lambda=({l1:.1f},{l2:.1f})')
        k += 1
    plt.gcf().set_size_inches(9,9)
    plt.tight_layout()
    plt.savefig(outfile, bbox_inches='tight', dpi=300)
    if show is True:
        plt.show()
    else:
        plt.close()

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
        # }

    # These are centered at an alt solution,
    # using correct intensity map (cov_sig=0.5)
    # OLD code
    # alt_pars = {
    #     'g1': -0.0249,
    #     'g2': 0.1070,
    #     'theta_int': 1.0423,
    #     'sini': 0.5770,
    #     'v0': 13.7093,
    #     'vcirc': 275.8384,
    #     'rscale': 4.2344,
    #     }

    # These are centered at an alt solution,
    # using correct intensity map (cov_sig=1)
    # alt_pars = {
    #     'g1': -0.0255,
    #     'g2': 0.1082,
    #     'theta_int': 1.0437,
    #     'sini': 0.5724,
    #     'v0': 10.0144,
    #     'vcirc': 278.4138,
    #     'rscale': 4.2491,
    #     }

    # These are centered at an alt solution,
    # using basis funcs (cov_sig=3)
    alt_pars = {
        'g1': 0.4781,
        'g2': -0.7357,
        'theta_int': 1.0699,
        'sini': 0.9991,
        'v0': 10.0008,
        'vcirc': 161.1112,
        'rscale': 8.9986,
    }

    print('Setting up alternative vmap solution')
    u = {'v_unit': pars['v_unit'], 'r_unit':pars['r_unit']}
    vmap_pars = {**true_pars, **u}
    vmap_alt_pars = {**alt_pars, **u}
    VMap = velocity.VelocityMap('default', vmap_pars)
    VMap_alt = velocity.VelocityMap('default', vmap_alt_pars)
    X, Y = utils.build_map_grid(datacube.Nx, datacube.Ny)

    outfile = os.path.join(outdir, 'compare-vmaps.png')
    print(f'Saving velocity map comparison to {outfile}')
    vmap = VMap('obs', X, Y)
    vmap_alt = VMap_alt('obs', X, Y)
    maps = [true_vmap, vmap_alt, vmap_alt-true_vmap, 100.*(vmap_alt-true_vmap)/true_vmap]
    tg1, tg2 = true_pars['g1'], true_pars['g2']
    ag1, ag2 = alt_pars['g1'], alt_pars['g2']
    titles = ['Alt Solution', 'True',
              'Alt-True', '% Residual']

    fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True,
                             figsize=(9,9))
    for i in range(4):
        ax = axes[i//2, i%2]
        if '%' in titles[i]:
            vmin, vmax = -10, 10
        else:
            vmin, vmax = None, None
        im = ax.imshow(maps[i], origin='lower', vmin=vmin, vmax=vmax)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)
        ax.set_title(titles[i])

    plt.suptitle(f'g_true=({tg1},{tg2}); g_alt=({ag1},{ag2})')
    plt.savefig(outfile, bbox_inches='tight', dpi=300)

    if show is True:
        plt.show()
    else:
        plt.close()

    print('Setting up imaps')
    basis_kwargs = pars['intensity'].copy()
    del basis_kwargs['type']
    imap = intensity.build_intensity_map(pars['intensity']['type'], datacube, basis_kwargs)

    im = imap.render(true_pars, datacube, pars)
    im_alt = imap.render(alt_pars, datacube, pars, redo=True)

    outfile = os.path.join(outdir, 'compare-imaps.png')
    print(f'Saving intensty map comparison to {outfile}')
    images = [true_im, im, im_alt, im_alt-im, im-true_im, im_alt-true_im]
    titles = ['Truth', 'True basis', 'Alt basis',
              'Alt basis - True basis', 'True basis - Truth', 'Alt basis - Truth']

    fig, axes = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True,
                               figsize=(10,6))

    for i in range(6):
        ax = axes[i//3, i%3]
        ishow = ax.imshow(images[i], origin='lower')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(ishow, cax=cax)
        ax.set_title(titles[i])

    plt.suptitle(f'g_true=({tg1},{tg2}); g_alt=({ag1},{ag2})')
    plt.savefig(outfile, bbox_inches='tight', dpi=300)

    if show is True:
        plt.show()
    else:
        plt.close()

    outfile = os.path.join(outdir, 'compare-datacube-slices.png')
    print(f'Saving comparison of datacube slices to {outfile}')
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

    Nslices = len(slice_indices)
    fig, axes = plt.subplots(nrows=3, ncols=Nslices, sharex=True, sharey=True,
                             figsize=(18, 8))
    tchi2 = 0
    achi2 = 0
    cov_sig = pars['cov_sigma']
    for i, indx in enumerate(slice_indices):
        # plot datacube slice
        ax = axes[0,i]
        dslice = datacube.slices[indx]._data
        ishow = ax.imshow(dslice, origin='lower')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(ishow, cax=cax)
        l1, l2 = lambdas[indx]
        if i == 0:
            ax.set_ylabel(f'Datacube slice')
        ax.set_title(f'({l1:.1f},{l2:.1f}) nm')
        # plot true model slices
        ax = axes[1,i]
        zfactor = 1. / (1 + VMap('obs', X, Y, normalized=True))
        sed_array = np.array([sed.x, sed.y])
        # TODO: update w/ psf if we want to run this old test again
        true_model = _compute_slice_model(
            (l1,l2), sed_array, zfactor, im)
        true_resid = dslice-true_model
        true_chi2 = np.sum((true_resid/cov_sig)**2) / (Nx*Ny)
        tchi2 += true_chi2
        ishow = ax.imshow(true_resid, origin='lower')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(ishow, cax=cax)
        ax.set_title(f'red chi2={true_chi2:.1f}')
        # l1, l2 = lambdas[indx]
        if i == 0:
            ax.set_ylabel('True model resid')
        # plot alt model slices
        ax = axes[2,i]
        alt_zfactor = 1. / (1 + VMap_alt('obs', X, Y, normalized=True))
        # TODO: update w/ psf if we want to run this old test again
        alt_model= _compute_slice_model(
            (l1,l2), sed_array, alt_zfactor, im_alt)
        alt_resid = dslice-alt_model
        alt_chi2 = np.sum((alt_resid/cov_sig)**2) / (Nx*Ny)
        achi2 += alt_chi2
        ishow = ax.imshow(alt_resid, origin='lower')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(ishow, cax=cax)
        ax.set_title(f'red chi2={alt_chi2:.1f}')
        l1, l2 = lambdas[indx]
        if i == 0:
            ax.set_ylabel('Alt model resid')

    plt.suptitle(f'True pars red chi2: {tchi2:.1f}; Alt pars red chi2: {achi2:.1f}')
    plt.tight_layout()
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
