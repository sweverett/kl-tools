import numpy as np
import fitsio
import matplotlib.pyplot as plt
import pathlib
import os
from argparse import ArgumentParser
import ipdb

import kl_tools.utils as utils
from kl_tools.cube import DataCube
from kl_tools.muse import MuseDataCube

parser = ArgumentParser()

parser.add_argument('--show', action='store_true', default=False,
                    help='Set to show test plots')
parser.add_argument('--test', action='store_true', default=False,
                    help='Set to run tests')

def main(args):
    show = args.show

    outdir = os.path.join(utils.TEST_DIR, 'continuum')
    utils.make_dir(outdir)

    testdir = utils.get_test_dir()
    testpath = pathlib.Path(os.path.join(testdir, 'test_data'))
    spec1dPath = testpath / pathlib.Path("spectrum_102021103.fits")
    spec3dPath = testpath / pathlib.Path("102021103_objcube.fits")
    catPath = testpath / pathlib.Path("MW_1-24_main_table.fits")
    emlinePath = testpath / pathlib.Path("MW_1-24_emline_table.fits")

    # Try initializing a datacube object with these paths.
    muse = MuseDataCube(
        cubefile=spec3dPath,
        specfile=spec1dPath,
        catfile=catPath,
        linefile=emlinePath
        )

    muse.set_line(truncate=False)

    eline = muse.pars['emission_lines'][0]
    line = eline.line_pars['value']
    line_unit = eline.line_pars['unit']
    lblue, lred = eline.sed_pars['lblue'], eline.sed_pars['lred']

    lambdas = np.array(muse.lambdas)
    iblue = np.where(lambdas[:,0] < lblue)[0][-1]
    ired = np.where(lambdas[:,1] > lred)[0][0]

    Nbox = 30
    new_lblue = lambdas[:,0][iblue-Nbox]
    new_lred = lambdas[:,1][ired+Nbox]
    args, kwargs = muse.truncate(new_lblue, new_lred, trunc_type='return-args')
    new_cube = DataCube(*args, **kwargs)

    Ngrid = int(np.ceil(np.sqrt(new_cube.Nspec)))

    fig, axes = plt.subplots(nrows=Ngrid, ncols=Ngrid, sharex=True, sharey=True)
    for k in range(Ngrid**2):
        i = k // Ngrid
        j = k % Ngrid
        ax = axes[i,j]
        try:
            sl = new_cube.slice(k)
            im = ax.imshow(sl.data, origin='lower')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(f'({sl.blue_limit:.1f}, {sl.red_limit:.1f}) nm')
        except:
            ax.axis('off')

    plt.suptitle(f'Slices around emission line at {line:.2f} {line_unit}', y=0.92)
    plt.subplots_adjust(wspace=0.1, hspace=0.3)
    fig.set_size_inches(20,20)

    outfile = os.path.join(outdir, 'continuum-slices.png')
    plt.savefig(outfile, bbox_inches='tight', dpi=300)

    if show is True:
        plt.show()

    return 0

if __name__ == '__main__':
    args = parser.parse_args()

    if args.test is True:
        print('Starting tests')
        rc = main(args)

        if rc == 0:
            print('All tests ran succesfully')
        else:
            print(f'Tests failed with return code of {rc}')
