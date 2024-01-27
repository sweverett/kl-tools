import unittest
import numpy as np
from astropy import units

from kl_tools.velocity import VelocityModel, OffsetVelocityModel, VelocityMap, get_model_types, build_model
from kl_tools.utils import get_test_dir, make_dir

class TestVelocityModel(unittest.TestCase):
    def setUp(self) -> None:
        # Set up any necessary objects or variables for the tests
        self.model_name = 'centered'
        self.model_pars = {
            'g1': 0.1,
            'g2': -0.075,
            'theta_int': np.pi / 3,
            'sini': 0.8,
            'v0': 10.,
            'vcirc': 200,
            'rscale': 5,
            'r_unit': units.Unit('pixel'),
            'v_unit': units.km / units.s,
        }

        self.model = VelocityModel(self.model_pars)

        return

    def test_model_pars(self) -> None:
        model = self.model
        pars = self.model_pars
        self.assertIsInstance(model, VelocityModel)
        self.assertEqual(model.name, self.model_name)
        self.assertEqual(model.pars['v0'], pars['v0'])
        self.assertEqual(model.pars['vcirc'], pars['vcirc'])
        self.assertEqual(model.pars['rscale'], pars['rscale'])
        self.assertEqual(model.pars['sini'], pars['sini'])

        return

class TestOffsetVelocityModel(unittest.TestCase):
    def setUp(self) -> None:
        # Set up any necessary objects or variables for the tests
        self.model_name = 'offset'
        self.model_pars = {
            'g1': 0.1,
            'g2': -0.075,
            'theta_int': np.pi / 3,
            'sini': 0.8,
            'v0': 10.,
            'vcirc': 200,
            'rscale': 5,
            'x0': 10,
            'y0': 10,
            'r_unit': units.Unit('pixel'),
            'v_unit': units.km / units.s,
        }

        self.model = OffsetVelocityModel(self.model_pars)

        return

    def test_model_pars(self) -> None:
        model = self.model
        pars = self.model_pars
        self.assertIsInstance(model, VelocityModel)
        self.assertEqual(model.name, self.model_name)
        self.assertEqual(model.pars['v0'], pars['v0'])
        self.assertEqual(model.pars['vcirc'], pars['vcirc'])
        self.assertEqual(model.pars['rscale'], pars['rscale'])
        self.assertEqual(model.pars['sini'], pars['sini'])
        self.assertEqual(model.pars['x0'], pars['x0'])
        self.assertEqual(model.pars['y0'], pars['y0'])

        return

class TestVelocityMap(unittest.TestCase):

    def setUp(self) -> None:
        self.plot_dir = get_test_dir() / 'plots' / 'velocity'
        make_dir(self.plot_dir)

        # used for plots
        self.rmax = 30
        self.show = False

        # setup models
        self.model_name = 'centered'
        self.model_pars = {
            'g1': 0.1,
            'g2': -0.075,
            'theta_int': np.pi / 3,
            'sini': 0.8,
            'v0': 10.,
            'vcirc': 200,
            'rscale': 5,
            'r_unit': units.Unit('pixel'),
            'v_unit': units.km / units.s,
        }

        self.vmap = VelocityMap(self.model_name, self.model_pars)

        offset_model_name = 'offset'
        offset_model_pars = {
            'g1': 0.1,
            'g2': -0.075,
            'theta_int': np.pi / 3,
            'sini': 0.8,
            'v0': 10.,
            'vcirc': 200,
            'rscale': 5,
            'x0': 10,
            'y0': 10,
            'r_unit': units.Unit('pixel'),
            'v_unit': units.km / units.s,
        }
        self.offset_vmap = VelocityMap(offset_model_name, offset_model_pars)

        return

    def test_plot_vmap(self) -> None:
        outdir = self.plot_dir
        vmap = self.vmap
        rmax = self.rmax
        show = self.show
        center = True

        outfile = str(outdir /  'speedmap-transorms.png')
        vmap.plot_map_transforms(
            outfile=outfile, show=show, speed=True, center=center,
            rmax=rmax
            )

        outfile = str(outdir / 'vmap-transorms.png')
        vmap.plot_map_transforms(
            outfile=outfile, show=show, speed=False, center=center,
            rmax=rmax
            )

        # now for offset model
        offset_vmap = self.offset_vmap

        outfile = str(outdir / 'speedmap-transorms-offset.png')
        offset_vmap.plot_map_transforms(
            outfile=outfile, show=show, speed=True, center=center,
            rmax=rmax
            )

        outfile = str(outdir / 'vmap-transorms-offset.png')
        offset_vmap.plot_map_transforms(
            outfile=outfile, show=show, speed=False, center=center,
            rmax=rmax
            )

        return

    def test_plot_planes(self) -> None:
        outdir = self.plot_dir
        vmap = self.vmap
        rmax = self.rmax
        show = self.show
        center = True

        for plane in ['disk', 'gal', 'source', 'obs']:
            outfile = str(outdir / f'vmap-{plane}.png')
            vmap.plot(
                plane, outfile=outfile, show=show, center=center,
                rmax=rmax
                )

        # plot combined velocity field planes
        plot_kwargs = {'rmax':rmax}
        outfile = str(outdir / 'vmap-all.png')
        vmap.plot_all_planes(
            plot_kwargs=plot_kwargs, show=show, outfile=outfile, center=center,
            )

        # plot combined velocity field planes normalized by c
        outfile = str(outdir / 'vmap-all-normed.png')
        vmap.plot_all_planes(
            plot_kwargs=plot_kwargs, show=show, outfile=outfile,
                            normalized=True, center=center,
            )

        # now for offset model

        offset_vmap = self.offset_vmap
        outfile = str(outdir / 'vmap-all-offset.png')
        offset_vmap.plot_all_planes(
            plot_kwargs=plot_kwargs, show=show, outfile=outfile, center=center,
            )

        return

if __name__ == '__main__':
    unittest.main()