import cube
import numpy as np
from astropy.io import fits
from astropy.table import Table, hstack
import os
import pickle
import fitsio
from astropy.table import Table
import matplotlib.pyplot as plt
import galsim as gs
import pathlib
import re
import astropy.wcs as wcs
import astropy.units as u
from argparse import ArgumentParser

import utils

import ipdb

parser = ArgumentParser()

parser.add_argument('--show', action='store_true', default=False,
                    help='Set to show test plots')
parser.add_argument('--test', action='store_true', default=False,
                    help='Set to run tests')

class MuseDataCube(cube.FitsDataCube):
    '''
    Sublass DataCube for MUSE observations.
    '''

    def __init__(self, cubefile=None, specfile=None, catfile=None,
                 linefile=None, pars=None):
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
        pars: dict
        '''

        # cubefile already saved & checked in FitsDataCube constructor
        for f in [specfile, catfile, linefile]:
            utils.check_file(f)

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
            ) for il,dl in zip(self.spec1d['WAVE_VAC'], dlam)]

        # construct a FitsDataCube
        super(MuseDataCube, self).__init__(
            cubefile=cubefile, bandpasses=bandpasses, pix_scale=pix_scale
            )

        # grab obj data from available catalogs
        catalog = Table.read(catfile)
        linecat = Table.read(linefile)

        # we want to save only the relevant entry from the two catalogs
        obj_id = re.search('(\d+)_*', cubefile).groups()[0]

        cat_row = catalog[catalog['UNIQUE_ID'] == obj_id]
        line_row = linecat[linecat['UNIQUE_ID'] == obj_id]

        self.obj_data = hstack([cat_row, line_row])

        self._set_parameters()

        # TODO: turn on once we understand muse weights
        self.set_weights()

        # Once everything is set up for MUSE, populate the parameters dict
        # that the runner will need.
        # TODO: implement!

        return

    def _set_parameters(self):
        '''
        Set a parameter dictionary populated with (some) of the fields that
        will be needed in the likelihood parameters dictionary, based
        on the data.

        NOTE: Some of these are duplicates w/ obj attributes for now, might
        clean up later
        '''

        self.pars = {}
        self.pars['pix_scale'] = self.pixel_scale
        self.pars['Nx'] = self.Nx
        self.pars['Ny'] = self.Ny
        self.pars['z'] = self._catalogEntry['Z']
        self.pars['spec_resolution'] = 3000.

        # A guess, based on throughput here:
        # https://www.eso.org/sci/facilities/paranal/instruments/muse/inst.html
        self.pars['bandpass_throughput'] = 0.2

        # are set later
        self.pars['sed_start'] = None
        self.pars['sed_end'] = None

        return

    def set_weights(self, ext=1):
        '''
        Set weights from a muse datacube file
        '''

        # should be an array of single weight values per slice
        weights = fitsio.read(self.cubefile, ext=1)

        l = len(weights)
        if l != self.Nspec:
            raise ValueError(f'The weight array has len {l} ' +\
                             f'but {self.Nspec=}!')

        super(MuseDataCube, self).set_weights(weights)

        return

    def set_line(line_choice='strongest'):

        # If no line indicated, set parameters for fitting the strongest emission line.
        if line_choice == 'strongest':
            line_index = np.argmax(self.obj_data['SN'])
        else:
            thing = np.where(self.obj_data['IDENT'] == line_choice)
            if len(thing) <1:
                print(f'your choice of emission line, {line_choice}, is ' +\
                      'not in the linelist for this object, which ' +\
                      f'contains only {self.obj_data["IDENT"]}.')
                ipdb.set_trace()
            line_index = thing[0][0] # EH: This had better be unique.

        self.line_name = self.obj_data['IDENT'][line_index]

        self.pars['sed_start'] = self.obj_data['LAMBDA_NB_MIN'][line_index]
        self.pars['sed_end'] = self.obj_data['LAMBDA_NB_MIN'][line_index]

        return

def main(args):

    testdir = utils.get_test_dir()
    testpath = pathlib.Path(os.path.join(testdir, 'test_data'))
    spec1dPath = testpath / pathlib.Path("spectrum_102021103.fits")
    spec3dPath = testpath / pathlib.Path("102021103_objcube.fits")
    catPath = testpath / pathlib.Path("MW_1-24_main_table.fits")
    emlinePath = testpath / pathlib.Path("MW_1-24_emline_table.fits")

    # Try initializing a datacube object with these paths.
    thisCube = MuseDataCube(
        cubefile=spec3dPath,
        specfile=spec1dPath,
        catfile=catPath,
        linefile=emlinePath
        )

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

