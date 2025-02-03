import numpy as np
import os
import sys
import astropy.units as u
import astropy.constants as constants
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

kl_path = '/Users/sweveret/repos/KLens'
sys.path.insert(0, kl_path)
from tfCube2 import GalaxyImage, TFCube

from velocity import VelocityMap
import utils

import pudb

parser = ArgumentParser()

parser.add_argument('--show', action='store_true', default=False,
                    help='Set to show test plots')
def main(args):
    show = args.show

    outdir = os.path.join(
        utils.TEST_DIR, 'test-vmap-vs-tfcube'
        )
    utils.make_dir(outdir)

    sampled_pars = {
        'g1': 0.05,
        'g2': -0.025,
        #     'g1': 0.0,
        #     'g2': 0.0,
    'theta_int': np.pi / 3,
        'sini': 0.8,
        'v0': 0.,
        'vcirc': 200,
        'rscale': 5,
    }

    im_pars = {
        'flux': 5e5,
        'hlr': 5,
       }

    pars = {
        'redshift': 0.3,
        'resolution': 5000,
        'pixel_scale': 1., # to simplify a few things
        'Nx': 100,
        'Ny': 100,
        'psf': {
            'type': 'Gaussian', # Matches what is in tfCube
            'fwhm': 3, # arcsec
        },
        'v_unit': u.Unit('km/s'),
        'r_unit': u.Unit('pix')
       }

    print('Setting up TFCube params')
    tf_pars = {}

    keys = ['g1', 'g2', 'theta_int', 'sini', 'vcirc']
    for key in keys:
        tf_pars[key] = sampled_pars[key]

    tf_pars['v_0'] = sampled_pars['v0']
    tf_pars['vscale'] = sampled_pars['rscale']
    tf_pars['r_0'] = 0.

    tf_pars['redshift'] = pars['redshift']
    tf_pars['Resolution'] = pars['resolution']
    tf_pars['psfFWHM'] = pars['psf']['fwhm']
    tf_pars['pixScale'] = pars['pixel_scale']
    tf_pars['r_hl_image'] = im_pars['hlr']
    tf_pars['r_hl_spec'] = im_pars['hlr']
    tf_pars['aspect'] = 0.1 # galsim default

    assert pars['Nx'] == pars['Ny']
    tf_pars['ngrid'] = pars['Nx']
    tf_pars['image_size'] = pars['Nx']

    line_species='Halpha'

    TF = TFCube(pars=tf_pars, line_species=line_species)

    print('Making the velocity maps')
    # TFCube
    c_kms = constants.c.to('km / s').value
    tf_vmap = c_kms * TF.getVmap(tf_pars['vcirc'],
                      tf_pars['sini'],
                      tf_pars['g1'],
                      tf_pars['g2'],
                      tf_pars['vscale'],
                      tf_pars['v_0'],
                      tf_pars['r_0'],
                      tf_pars['theta_int']
                     )

    # kl-tools
    vmap_units = {
        'v_unit': pars['v_unit'],
        'r_unit': pars['r_unit']
       }
    vmap_pars = {**sampled_pars, **vmap_units}
    kl_Vmap = VelocityMap('default', vmap_pars)
    X, Y = utils.build_map_grid(pars['Nx'], pars['Ny'])
    kl_vmap = kl_Vmap('obs', X, Y, normalized=False)

    fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True,
                            figsize=(9,9), facecolor='w')

    vmaps = [tf_vmap, kl_vmap, tf_vmap-kl_vmap, 100*(tf_vmap-kl_vmap)/kl_vmap]
    titles = ['TFCube2', 'kl-tools', 'Residual (TFCube=Truth)', '% Residual (TFCube=Truth)']

    outfile = os.path.join(outdir, 'compare-tfcube-vmaps.png')
    print(f'Saving vmap comparison to {outfile}')
    for i in range(4):
        ax = axes[i//2, i%2]
        vmin, vmax = None, None
        # if i == 4-1:
        #     vmin, vmax = None, None
        # else:
        #     vmin, vmax = None, None
        im = ax.imshow(vmaps[i], origin='lower', vmin=vmin, vmax=vmax, cmap='RdBu')
        ax.set_title(titles[i])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        if i % 2 == 1:
            label = 'v (km/s)'
        else:
            label = None
        plt.colorbar(im, cax=cax, label=label)

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
