import unittest
import numpy as np
from astropy.wcs import WCS
from astropy.coordinates import Angle

from kl_tools.coordinates import OrientedAngle
from testing_utils import create_wcs_with_rotation

class TestOrientedAngle(unittest.TestCase):

    def test_init_str(self):
        # Test initialization with string input
        angle = OrientedAngle('30 deg', orientation='cartesian')
        self.assertEqual(angle.deg, 30)
        self.assertAlmostEqual(angle.rad, np.pi/6)

        return

    def test_init_unit(self):
        # Test initialization with an explicit unit
        angle = OrientedAngle(30, unit='deg', orientation='cartesian')
        self.assertEqual(angle.deg, 30)
        self.assertAlmostEqual(angle.rad, np.pi/6)

        return

    def test_property_conversion_equality(self):
        # Test if the property & conversion methods are equivalent

        wcs_angle = 60 # degrees, relative to Sky
        wcs = create_wcs_with_rotation(f'{wcs_angle} deg')
        angle = OrientedAngle('30 deg', orientation='cartesian', wcs=wcs)

        self.assertAlmostEqual(
            angle.cartesian.deg, angle.to_orientation('cartesian').deg
            )
        self.assertAlmostEqual(
            angle.sky.deg, angle.to_orientation('sky').deg
            )
        self.assertAlmostEqual(
            angle.east_of_north.deg, angle.to_orientation('east-of-north').deg
            )

        return

    def test_cartesian_conversions_no_wcs(self):
        # Test Cartesian conversions without WCS

        value = 30 # degrees, Cartesian
        angle = OrientedAngle(f'{value} deg', orientation='cartesian')

        # Cartesian to Cartesian
        self.assertAlmostEqual(value, angle.cartesian.deg)

        # Cartesian to Sky
        # NOTE: Equivalent to Cartesian for no WCS
        self.assertAlmostEqual(angle.sky.deg, angle.cartesian.deg)

        # Cartesian to East-of-North
        cart2EoN = Angle(angle.cartesian.deg-90, unit='deg')
        cart2EoN = cart2EoN.wrap_at('360 deg')
        self.assertAlmostEqual(angle.east_of_north.deg, cart2EoN.deg)

        return

    def test_sky_conversions_no_wcs(self):
        # Test Sky conversions without WCS

        value = 30 # degrees, Sky
        angle = OrientedAngle(f'{value} deg', orientation='sky')

        # Sky to Sky
        self.assertAlmostEqual(value, angle.sky.deg)

        # Sky to Cartesian
        # NOTE: Equivalent to Sky for no WCS
        self.assertAlmostEqual(angle.cartesian.deg, angle.sky.deg)

        # Sky to East-of-North
        sky2EoN = Angle(angle.sky.deg-90, unit='deg')
        sky2EoN = sky2EoN.wrap_at('360 deg')
        self.assertAlmostEqual(angle.east_of_north.deg, sky2EoN.deg)

        return

    def test_EoN_conversions_no_wcs(self):
        # Test east-of-north conversions without WCS

        value = 30 # degrees, EoN
        angle = OrientedAngle(f'{value} deg', orientation='east-of-north')

        # EoN to EoN
        self.assertAlmostEqual(value, angle.east_of_north.deg)

        # EoN to Cartesian
        EoN2cart = Angle(angle.east_of_north.deg+90, unit='deg')
        EoN2cart = EoN2cart.wrap_at('360 deg')
        self.assertAlmostEqual(angle.cartesian.deg, EoN2cart.deg)

        # EoN to Sky
        # NOTE: Comparison same as Cartesian for no WCS
        self.assertAlmostEqual(angle.sky.deg, EoN2cart.deg)

        return

    def test_cartesian_conversions(self):
        # Test Cartesian conversions with WCS

        value = 30 # degrees, Cartesian
        wcs_angle = 60 # degrees, relative to Sky
        wcs = create_wcs_with_rotation(f'{wcs_angle} deg')
        angle = OrientedAngle(f'{value} deg', orientation='cartesian', wcs=wcs)

        # Cartesian to Cartesian
        self.assertAlmostEqual(value, angle.cartesian.deg)

        # Cartesian to Sky
        cart2sky = Angle(angle.cartesian.deg+wcs_angle, unit='deg')
        cart2sky = cart2sky.wrap_at('360 deg')
        self.assertAlmostEqual(angle.sky.deg, cart2sky.deg)

        # Cartesian to East-of-North
        cart2EoN = Angle(angle.sky.deg-90, unit='deg').wrap_at('360 deg')
        self.assertAlmostEqual(angle.east_of_north.deg, cart2EoN.deg)

        return

    def test_sky_conversions(self):
        # Test Sky conversions with WCS

        value = 30 # degrees, Sky
        wcs_angle = 60 # degrees, relative to Sky
        wcs = create_wcs_with_rotation(f'{wcs_angle} deg')
        angle = OrientedAngle(f'{value} deg', orientation='sky', wcs=wcs)

        # Sky to Sky
        self.assertAlmostEqual(value, angle.sky.deg)

        # Sky to Cartesian
        angle_cartesian = angle.to_orientation('cartesian')
        sky2cart = Angle(value-wcs_angle, unit='deg').wrap_at('360 deg')
        self.assertAlmostEqual(angle_cartesian.deg, sky2cart.deg)

        # Sky to East-of-North
        angle_EoN = angle.to_orientation('east-of-north')
        sky2EoN = Angle(value-90, unit='deg').wrap_at('360 deg')
        self.assertAlmostEqual(angle_EoN.deg, sky2EoN.deg)

        return

    def test_EoN_conversions(self):
        # Test east-of-north conversions with WCS

        value = 30 # degrees, EoN
        wcs_angle = 60 # degrees, relative to Sky
        wcs = create_wcs_with_rotation(f'{wcs_angle} deg')
        angle = OrientedAngle(
            f'{value} deg', orientation='east-of-north', wcs=wcs
            )

        # EoN to EoN
        angle_EoN = angle.to_orientation('east-of-north')
        self.assertAlmostEqual(value, angle_EoN.deg)

        # EoN to Sky
        angle_sky = angle.to_orientation('sky')
        self.assertAlmostEqual(angle_sky.deg, value+90)

        # EoN to Cartesian
        angle_cartesian = angle.to_orientation('cartesian')
        self.assertAlmostEqual(angle_cartesian.deg, value+90-wcs_angle)

        return

if __name__ == '__main__':
    unittest.main()