import unittest
import numpy as np
import matplotlib.pyplot as plt

from kl_tools.muse import MuseDataCube, MuseFixer
from kl_tools.emission import LINE_LAMBDAS
from kl_tools.utils import get_test_dir, make_dir

class TestMuseDataCube(unittest.TestCase):

    def setUp(self) -> None:
        # Set up any necessary objects or data for the tests
        self.data_dir = get_test_dir() / 'data'
        self.out_dir = get_test_dir() / 'out/muse/'
        self.obj_id = 122003050

        self.emline_file = self.data_dir / 'MW_1-24_emline_table.fits'
        self.cat_file = self.data_dir / 'MW_1-24_main_table.fits'
        self.spec_file = self.data_dir / f'spectrum_{self.obj_id}.fits'
        self.cube_file = self.data_dir / f'{self.obj_id}_objcube.fits'

        self.muse_dc = MuseDataCube(
            self.cube_file, self.spec_file, self.cat_file, self.emline_file
            )

        self.outdir = get_test_dir() / 'out/muse/'

        make_dir(self.outdir)

        return

    def tearDown(self) -> None:
        # Clean up any resources used by the tests
        pass

    def test_get_2d_mask(self) -> None:

        data = self.muse_dc.stack()
        self.muse_dc.set_line('Ha')
        data_ha = self.muse_dc.stack()
        mask = self.muse_dc.get_2d_mask()

        plt.subplot(131)
        plt.imshow(data, origin='lower')
        plt.colorbar()
        plt.title('Full datacube stack')
        plt.subplot(132)
        plt.imshow(data_ha, origin='lower')
        plt.colorbar()
        plt.title('Ha datacube stack')
        plt.subplot(133)
        plt.imshow(mask, origin='lower')
        plt.colorbar()
        plt.title('Stacked mask')

        plt.gcf().set_size_inches(20,8)

        plt.savefig(self.outdir / '2d_mask.png')
        plt.close()

        return

    def test_stack_for_nans(self) -> None:
        # Early tests of the imap on certain MUSE datacube stacks resulted in nan's. While we allow it in the datacube.data attribute, we overwrite with a fill value in the stack method

        stack = self.muse_dc.stack()
        self.assertFalse(np.any(np.isnan(stack)))

        fill_val = -1000
        stack = self.muse_dc.stack(nan_fill=fill_val)

        return

    def test_set_parameters(self) -> None:
        # Test the _set_parameters method of MuseDataCube
        pass

    def test_set_weights(self) -> None:
        # Test the set_weights method of MuseDataCube
        pass

    def test_set_masks(self) -> None:
        # Test the set_masks method of MuseDataCube
        pass

    def test_set_continuum(self) -> None:
        # Test the set_continuum method of MuseDataCube
        pass

    def test_set_line(self) -> None:
        # Test the set_line method of MuseDataCube
        pass

class TestMuseFixer(unittest.TestCase):
    '''
    Some MUSE objects appear to have emission line tables that are not correct, particularly the naming of the emission lines. This class is designed to fix those tables for the relevant objects
    '''

    def setUp(self) -> None:
        # Set up any necessary objects or data for the tests

        # for the moment, only one object is known to be bugged
        self.obj_id = 122003050

        self.data_dir = get_test_dir() / 'data'
        self.out_dir = get_test_dir() / 'out/muse/'

        self.emline_file = self.data_dir / 'MW_1-24_emline_table.fits'
        self.cat_file = self.data_dir / 'MW_1-24_main_table.fits'
        self.spec_file = self.data_dir / f'spectrum_{self.obj_id}.fits'
        self.cube_file = self.data_dir / f'{self.obj_id}_objcube.fits'

        self.muse_dc = MuseDataCube(
            self.cube_file, self.spec_file, self.cat_file, self.emline_file
            )

    def tearDown(self):
        # Clean up any resources used by the tests
        pass

    def test_muse_fixer_map(self, line_delta=10) -> None:

        # as we know the redshift of the object, we can make sure that the emission lines are roughly where we expect them to be in the spectrum (the unfixed values are off by thousands of Angstroms)

        # NOTE: At this stage, the line IDENT's should already be corrected, so we are just confirming this

        z = self.muse_dc.z

        for line in self.muse_dc.lines:
            # TODO: Understand why the differences are still of order a few Anstroms...
            observed = line['LAMBDA_PEAK_SN']
            expected = LINE_LAMBDAS[line['IDENT']].value * (1. + z)
            self.assertAlmostEqual(expected, observed, delta=line_delta)

        return

    def test_fix_emission_line_table(self) -> None:
        # Test the fix_emission_line_table method of MuseFixer
        pass

if __name__ == '__main__':
    unittest.main()
