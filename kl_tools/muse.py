import numpy as np
from astropy.io import fits
from astropy.table import Table, join, hstack
import os
import pickle
import fitsio
from astropy.table import Table
from scipy.sparse import identity, dia_matrix
import matplotlib.pyplot as plt
import galsim as gs
import pathlib
import re
import astropy.wcs as wcs
import astropy.units as u
from argparse import ArgumentParser

import kl_tools.utils as utils
from kl_tools.cube import DataCube, CubePars
from kl_tools.emission import EmissionLine, LINE_LAMBDAS

parser = ArgumentParser()

parser.add_argument('--show', action='store_true', default=False,
                    help='Set to show test plots')
parser.add_argument('--test', action='store_true', default=False,
                    help='Set to run tests')

class MuseDataCube(DataCube):
    '''
    Sublass DataCube for MUSE observations.
    '''

    def __init__(self, cubefile, specfile, catfile, linefile, pars=None):
        '''
        Initialize a DataCube object from the contents of a MUSE fits datacube.

        cubefile: string, or anything parsable by pathlib.Path
            Filename containing MUSE datacube
        specfile: string, or anything parsable by pathlib.Path
            Filename containing MUSE 1d calibrated spectrum.
        catfile: str, or anything parsable by pathlib.Path
            Catalog containing general data from the MUSE wide survey
        linefile: str, or anything parsable by pathlib.Path
            Catalog containing emission line tables from the MUSE wide survey
        pars: dict, cube.CubePars
            A dictionary or CubePars that holds any additional metadata
        '''

        for f in [cubefile, specfile, catfile, linefile]:
            utils.check_file(f)

        self.cubefile = cubefile
        self.specfile = specfile
        self.catfile = catfile
        self.linefile = linefile

        self.spec1d = fitsio.read(specfile)
        self.cube_hdr = fitsio.read_header(cubefile)
        self.cube_wcs = wcs.WCS(self.cube_hdr)

        # determine pixel scale
        # NOTE: We use a geometric average to keep the right pixel area,
        # in case the pixels aren't square
        scales = self.cube_wcs.proj_plane_pixel_scales()
        pix_scale = np.sqrt(
            scales[0].to(u.arcsec)*scales[1].to(u.arcsec)
            ).value

        # Make the list of bandpasses.
        dlam = (self.spec1d['WAVE_VAC'][1:] - self.spec1d['WAVE_VAC'][:-1])
        dlam = np.append(dlam,dlam[-1]) # array of spaxel widths
        bandpasses = [gs.Bandpass(
            1.0, 'A', blue_limit=il-dl, red_limit=il+dl, zeropoint=22.5
            ) for il,dl in zip(self.spec1d['WAVE_VAC'], dlam/2.)]

        pars_dict = {
            'pix_scale': pix_scale,
            'bandpasses': bandpasses
        }
        pars = CubePars(pars_dict)

        data = fitsio.read(cubefile)

        super(MuseDataCube, self).__init__(data, pars=pars)

        self.pars['files'] = {
            'cubefile': cubefile,
            'specfile': specfile,
            'catfile': catfile,
            'linefile': linefile,
        }

        # grab obj data from available catalogs
        catalog = Table.read(catfile)
        linecat = Table.read(linefile)

        # we want to save only the relevant entry from the two catalogs
        try:
            obj_id = re.search('(\d+)_*', cubefile).groups()[0]
        except TypeError:
            # if a Pathlib obj is passed
            obj_id = re.search('(\d+)_*', cubefile.name).groups()[0]

        self.obj_id = int(obj_id)
        self.obj_data = catalog[catalog['UNIQUE_ID'] == obj_id]
        self.lines = linecat[linecat['UNIQUE_ID'] == obj_id]

        # some emission line data is incorrect in the MUSE catalog for certain objects; check & fix if necessary
        self.fix_emission_lines()

        self._set_parameters()

        self.set_weights()
        self.set_masks()

        return

    def fix_emission_lines(self):
        '''
        For certain objects in the MUSE emission line table, the emission line is labeled with an incorrect IDENT. This method checks for such objects and corrects the stored emission line rows accordingly.
        '''

        if self.obj_id in MuseFixer._emline_ident_fix:
            ident_map = MuseFixer._emline_ident_fix[self.obj_id]
            for i, line in enumerate(self.lines):
                real_ident = ident_map[line['IDENT']]
                self.lines[i]['IDENT'] = real_ident

        return

    def _set_parameters(self):
        '''
        Set a parameter dictionary populated with (some) of the fields that
        will be needed in the likelihood parameters dictionary, based
        on the data.

        NOTE: Some of these are duplicates w/ obj attributes for now, might
        clean up later
        '''

        # TODO: Should we put this anywhere else?
        # self.pars['z'] = self.obj_data['Z']

        # some are set later
        specs = {
            # A guess, based on throughput here:
            # https://www.eso.org/sci/facilities/paranal/instruments/muse/inst.html
            'throughput': 0.2,
            'resolution': 3200.,
        }

        Nlines = len(self.lines)

        z = self.z
        R = specs['resolution']

        lines = []
        for line in self.lines:
            # TODO: We should investigate whether to use LAMBDA_SN directly,
            # or even marginalize over z
            line_name = line['IDENT']
            line_lambda = LINE_LAMBDAS[line_name]

            line_pars = {
                'name': line_name,
                'value': line_lambda.value,
                'R': R,
                'z': z,
                'unit': line_lambda.unit
            }
            # will be updated later
            sed_pars = {
                'lblue': float(line['LAMBDA_NB_MIN']),
                'lred': float(line['LAMBDA_NB_MAX']),
                'resolution': 0.1, # angstroms
                'unit': line_lambda.unit
            }
            lines.append(EmissionLine(line_pars, sed_pars))

        self.pars['emission_lines'] = lines

        return

    def set_weights(self, weights=None, ext=1):
        '''
        Set weights from a muse datacube file
        '''

        if weights is None:
            # should be an array of single weight values per slice
            weights = fitsio.read(self.cubefile, ext=1)

            l = len(weights)
            if l != self.Nspec:
                raise ValueError(f'The weight array has len {l} ' +\
                                 f'but {self.Nspec=}!')

        # MUSE wgt maps are actually sky background, so take inverse
        # and handle bad weights
        actual_weights = 1./weights
        bad = weights <= 0
        if np.sum(bad) > 0:
            actual_weights[bad] = 0

        super(MuseDataCube, self).set_weights(actual_weights)

        return

    def set_masks(self, masks=None, ext=2):
        '''
        Set masks from a muse datacube file
        '''

        if masks is None:
            # should be an array of single mask values per slice
            muse_mask = fitsio.read(self.cubefile, ext=2)

            # TODO: lots of info here, but for now we need to just mask out pixels where the mask is 0
            masks = np.zeros_like(muse_mask)
            masks[muse_mask == 0] = 1

        # will set all masks to the same mask
        super(MuseDataCube, self).set_masks(masks)

        return

    def set_continuum(self, lmin_line, lmax_line, lmin_cont, lmax_cont,
                      method='sum'):
        '''
        Build a 2d template for the continuum from the spectrum near the line.
        '''

        # Make a new cube object for the blue continuum and the red continuum.
        lblue = lmin_cont
        lred = lmin_line
        args, kwargs = self.truncate(lblue, lred, trunc_type='return-args')
        blue_cont_cube = DataCube(*args,**kwargs)
        blue_cont_template = np.sum(
            blue_cont_cube.data * blue_cont_cube.weights, axis=0
            ) / np.sum(blue_cont_cube.weights, axis=0)

        lblue = lmax_line
        lred = lmax_cont
        args, kwargs = self.truncate(lblue, lred, trunc_type='return-args')
        red_cont_cube = DataCube(*args,**kwargs)
        red_cont_template = np.sum(
            red_cont_cube.data * red_cont_cube.weights, axis=0
            ) / np.sum(red_cont_cube.weights, axis=0)

        self._continuum_template = (blue_cont_template + red_cont_template) / 2.

        return

    def cutout(self, shape, center=None, cutout_type='in_place'):
        '''
        Cutout a smaller datacube from the current one. See DataCube.cutout()

        shape: tuple
            The (Nx, Ny) shape of the new datacube (keeps all wavelengths)
        center: tuple
            The center of the new datacube. If None, will use the existing center
        cutout_type: str
            Select whether to apply the cutout w/ the DataCube constructor (in-place) or to just return the (args, kwargs) needed to produce the cutout (return-args). This is particularly useful for subclasses of DataCube
        '''

        args, kwargs = super(MuseDataCube, self).cutout(
            shape, center=center, cutout_type='return-args'
            )

        if cutout_type == 'in_place':
            super(MuseDataCube, self).__init__(*args, **kwargs)
        else:
            return args, kwargs

    def set_line(self, line_choice='strongest', truncate=True):
        '''
        Set the emission line actually used in an analysis of a datacube,
        and possibly truncate it to slices near the line
        '''

        # If no line indicated, set parameters for fitting the strongest emission line.
        if line_choice == 'strongest':
            line_index = np.argmax(self.lines['SN'])
        else:
            thing = np.where(self.lines['IDENT'] == line_choice)
            if len(thing) <1:
                print(f'your choice of emission line, {line_choice}, is ' +\
                      'not in the linelist for this object, which ' +\
                      f'contains only {self.obj_data["IDENT"]}.')
            line_index = thing[0][0] # EH: This had better be unique.

        # update all line-related meta data
        self.lines = self.lines[line_index]
        self.pars['emission_lines'] = [self.pars['emission_lines'][line_index]]

        # Estimate a continuum template
        boxwidth = self.pars['emission_lines'][0].sed_pars['lred'] -\
                   self.pars['emission_lines'][0].sed_pars['lblue']
        self.set_continuum(
            self.pars['emission_lines'][0].sed_pars['lblue'],
            self.pars['emission_lines'][0].sed_pars['lred'],
            self.pars['emission_lines'][0].sed_pars['lblue'] - boxwidth,
            self.pars['emission_lines'][0].sed_pars['lred'] + boxwidth
            )

        # have to store it temporarily, and then set it after truncation
        continuum_template = self._continuum_template

        # create new truncated datacube around line, if desired
        if truncate is True:
            lblue = self.pars['emission_lines'][0].sed_pars['lblue']
            lred = self.pars['emission_lines'][0].sed_pars['lred']
            args, kwargs = self.truncate(lblue, lred, trunc_type='return-args')
            super(MuseDataCube, self).__init__(*args, **kwargs)

            self._continuum_template = continuum_template

        return

    @property
    def z(self) -> float:
        return self.obj_data['Z'].value[0]

#-----------------------------------------------------------------------------
# This section is for helper classes and methods in support of MUSE datacubes

class MuseFixer(object):
    '''
    This class handles any fixes or corrections to ingested MUSE datacubes. It spawned from the need to handle a specific object in the MUSE emission line catalog that had its line indexing scrambled
    '''

    # dict of object IDs: {stated_line: true_line}
    _emline_ident_fix = {
        122003050: {
            'Hd': 'Hb',
            'Hg': 'N2_1',
            'Hb': 'Ha',
            'O3_1': 'N2_2',
            'O3_2': 'S2_1',
            'He1': 'S2_2',
            'O1': 'Hd',
            'N2_1': 'Hg',
            'Ha': 'O3_1',
            'N2_2': 'O3_2',
            'S2_1': 'He1',
            'S2_2': 'O1'
        }
    }

    # dict of object IDs: (index tuple) for the emission line catalog that need reindexing
    _emline_indx_fix = {
        # 122003050: (6,  7,  0,  8,  9, 10, 11,  1,  2,  3,  4,  5),
        122003050: {
            'Hd': 6,
            'Hg': 7,
            'Hb': 0,
            'O3_1': 8,
            'O3_2': 9,
            'He1': 10,
            'O1': 11,
            'N2_1': 1,
            'Ha': 2,
            'N2_2': 3,
            'S2_1': 4,
            'S2_2': 5
        }
    }

    def __init__(self):
        return

    # @classmethod

    @classmethod
    def fix_emission_line_table(self, emline_table):
        '''
        Take in a MUSE emission line table and make any corrections necessary. Return the fixed table
        '''
        # TODO
        pass

def main(args):

    data_dir = utils.get_script_dir() / 'test_data'
    spec1dPath = data_dir / pathlib.Path("spectrum_102021103.fits")
    spec3dPath = data_dir / pathlib.Path("102021103_objcube.fits")
    catPath = data_dir / pathlib.Path("MW_1-24_main_table.fits")
    emlinePath = data_dir / pathlib.Path("MW_1-24_emline_table.fits")

    # Try initializing a datacube object with these paths.
    muse = MuseDataCube(
        cubefile=spec3dPath,
        specfile=spec1dPath,
        catfile=catPath,
        linefile=emlinePath
        )

    # check that selecting max line s2n works correctly
    muse.set_line()

    # check that redshift acessing works correctly
    muse.z

    outdir = os.path.join(utils.TEST_DIR, 'muse')
    utils.make_dir(outdir)

    import matplotlib.pyplot as plt
    # plt.subplots(4,4)
    for i in range(muse.Nspec):
        lam = muse.lambdas[i]
        # sci
        plt.subplot(muse.Nspec,3,3*i+1)
        plt.imshow(muse.data[i])
        plt.colorbar()
        plt.ylabel(f'({lam[0]:.1f},{lam[1]:.1f})')
        if i == 0:
            plt.title('sci')
        # wgt
        plt.subplot(muse.Nspec,3,3*i+2)
        plt.imshow(muse.weights[i])
        plt.colorbar()
        if i == 0:
            plt.title('wgt')
        # msk
        plt.subplot(muse.Nspec,3,3*i+3)
        plt.imshow(muse.masks[i])
        plt.colorbar()
        if i == 0:
            plt.title('msk')

    plt.gcf().set_size_inches(7,24)

    outfile = os.path.join(outdir, 'muse-datacube-truncated.png')
    plt.savefig(outfile, bbox_inches='tight', dpi=300)

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

