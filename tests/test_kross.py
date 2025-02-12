import unittest
import astropy.units as u

from kl_tools.kross.cube import KROSSDataCube, get_kross_obj_data
from kl_tools.utils import get_base_dir

class TestKROSS(unittest.TestCase):

    def setUp(self):
        return

    def tearDown(self):
        pass

    def test_get_kross_obj_data(self):
        # test the get_kross_obj_data function
        kid = 171

        obj_data = get_kross_obj_data(kid)

        self.assertIn('cube', obj_data)
        self.assertIn('cube_hdr', obj_data)
        self.assertIn('cube_file', obj_data)
        self.assertIn('velocity', obj_data)
        self.assertIn('velocity_file', obj_data)
        self.assertIn('sigma', obj_data)
        self.assertIn('sigma_file', obj_data)
        self.assertIn('halpha', obj_data)
        self.assertIn('halpha_file', obj_data)
        self.assertIn('hst', obj_data)
        self.assertIn('hst_file', obj_data)

        cube_file = get_base_dir() / 'data/kross/cubes//C-zcos_z1_925.fits'
        self.assertEqual(obj_data['cube_file'], cube_file)

        return

    def test_KROSSDataCube_init(self):
        # test the KROSSDataCube class initialization

        kid = 171
        cube = KROSSDataCube(kid)

        self.assertEqual(cube.pars['pix_scale'], 0.1*u.arcsec)
        self.assertEqual(1, len(cube.pars['emission_lines']))
        self.assertEqual('Ha', cube.pars['emission_lines'][0].line_pars['name'])
        self.assertIn('bandpasses', cube.pars)
        self.assertIn('cube', cube.pars['files'])
        self.assertIn('velocity', cube.pars['files'])
        self.assertIn('halpha', cube.pars['files'])
        self.assertIn('sigma', cube.pars['files'])
        self.assertIn('hst', cube.pars['files'])
        self.assertAlmostEqual(0.9128704, cube.z)

        return

    def test_KROSSDataCube_set_line(self):
        # test the KROSSDataCube class set_line method

        kid = 171
        cube = KROSSDataCube(kid)

        cube.set_line('Ha')

        self.assertEqual(1, len(cube.pars['emission_lines']))
        self.assertEqual(
            'Ha', cube.pars['emission_lines'][0].line_pars['name']
            )

        return

    def test_KROSSDataCube_truncation(self):

        kid = 171
        cube = KROSSDataCube(kid)
        cube_shape = cube.shape

        cube.set_line('Ha', truncate=False)
        self.assertEqual(cube.data.shape, cube_shape)
        self.assertEqual(cube.shape, cube_shape)

        cube.set_line('Ha', truncate=True)
        self.assertNotEqual(cube.data.shape, cube_shape)
        self.assertNotEqual(cube.shape, cube_shape)

        return

if __name__ == '__main__':
    unittest.main()