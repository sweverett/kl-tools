import numpy as np
from astropy.coordinates import Angle
from astropy.wcs import WCS
from astropy.units import Unit as u

'''
Some classes and methods for defining, generating, and transforming various
coordinate and angle representations.

Particularly useful for:
- Generating 2D position grids (in image or world coordinates) needed for
  vmaps and imaps
- Dealing with angles in various units and representations
'''

class OrientedAngle(Angle):
    '''
    An extension of an astropy Angle object that provides some additional
    functionality common to our use cases, namely:
      - The ability to specify the orientation of the angle, possibly relative
        to an image WCS
      - Methods for transforming the angle between different orientations

    Parameters
    ----------
    angle: float | str | astropy.coordinates.Angle
        The value of the angle, either as a float, str, or an astropy Angle.
        Must provide a unit if the value is a float.
    unit : str or astropy.units.Unit, optional
        The unit of the angle. Default is None, but is required if the
        passed value is not an astropy Angle object.
    orientation: str, optional
        The orientation of the angle. Default is 'cartesian', which is
        the standard orientation of angles being measured counter-clockwise
        from the x-axis, regardless of any image WCS. All options include:
        - 'cartesian': Angle is measured ccw from the x-axis (default)
        - 'sky': Angle is measured ccw from the right when looking up from the
            sky, which is image/Cartesian-like but physically corresponds to
            north-of-west (equvalent to 'image' if wcs=None)
        - 'east-of-north': Angle is measured east of the north-aligned
            DEC-axis. This measures ccw from the DEC-axis in most astronomical images, but will measure ccw from 90 deg if no WCS is passed
    wcs : astropy.wcs.WCS, optional
        The WCS of the image the angle is being measured in. Default is
        None. Only affects angle transformations if orientation is not
        'cartesian'
    wrap_angle: str or astropy Angle, optional
        The angle at which the angle wraps around. Default is '360 deg'.
        This is used to ensure that the angle is always within the range
        [0, wrap_angle). If a string is passed, it must be a valid astropy
        Angle initialization string.
    '''

    _allowed_orientations = [
        'cartesian',
        'sky',
        'east-of-north'
        ]

    def __new__(cls, angle, unit=None, orientation='cartesian', wcs=None,
                wrap_angle='360 deg', **kwargs):
        '''
        NOTE: The parent astropy classes *do not have an __init__ method*. Thus
        the least awkward way to do this is to follow their pattern and have
        __new__ do both the construction & initialization of the instance
        '''

        # handle the case in which the passed angle already needs to be
        # wrapped by the wrap_angle
        temp_angle = Angle(angle, unit=unit)
        temp_angle = temp_angle.wrap_at(wrap_angle)

        instance = super().__new__(cls, temp_angle, **kwargs)

        if orientation not in cls._allowed_orientations:
            raise ValueError(f'Invalid orientation: {orientation}')
        instance.orientation = orientation

        if wcs is not None:
            if not isinstance(wcs, WCS):
                raise ValueError('wcs must be an astropy.wcs.WCS object')
        instance.wcs = wcs

        # set a uniform wrap angle for all later transformations
        instance.wrap_angle = Angle(wrap_angle)

        # determine the rotation angle of the image given the WCS, if provided
        instance._compute_image_rotation()

        return instance

    def _compute_image_rotation(self):
        '''
        Determine the rotation angle of the image given the WCS, if provided
        '''

        if self.wcs is None:
            self.image_rotation = Angle('0 deg')
            return

        wcs = self.wcs

        # extract the CD matrix or PC matrix components from the WCS
        # to build the rotation matrix
        # NOTE: While the CD matrix contains the pixel scale, the factor
        # cancels out in the arctan calculation
        if wcs.wcs.has_pc() is True:
            R = wcs.wcs.get_pc()
        elif wcs.wcs.has_cd() is True:
            R = wcs.wcs.get_cd()
        else:
            raise ValueError('WCS must have a CD or PC matrix')

        # calculate the rotation angle of the image
        cos = R[0, 0]
        sin = R[1, 0]
        theta = np.arctan2(sin, cos)

        # determine if there is any mirroring in the WCS
        det = np.linalg.det(R)
        if det < 0:
            theta += np.pi

        wrap = self.wrap_angle
        self.image_rotation = Angle(theta, unit='radian').wrap_at(wrap)

        return

    @property
    def cartesian(self):
        '''
        Return the angle in the standard cartesian orientation
        '''

        return self.to_orientation('cartesian')

    @property
    def sky(self):
        '''
        Return the angle in the 'sky' orientation
        '''

        return self.to_orientation('sky')

    @property
    def east_of_north(self):
        '''
        Return the angle in the 'east-of-north' orientation
        '''

        return self.to_orientation('east-of-north')

    def to_orientation(self, new_orientation):
        '''
        Convert the angle to a new orientation. Similar to the existing
        Angle.to() method for units.

        Must be one of the allowed orientations:
        - 'cartesian': Angle is measured north from the x-axis (default)
        - 'sky': Angle is measured north of west when looking up at the sky
            (equvalent to 'cartesian' if wcs=None)
        - 'east-of-north': Angle is measured east of the north-aligned DEC-axis
        '''

        if new_orientation not in self._allowed_orientations:
            raise ValueError(f'Invalid orientation: {new_orientation}')

        if new_orientation == self.orientation:
            return self

        if self.orientation == 'cartesian':
            if new_orientation == 'sky':
                return self._cartesian2sky()
            elif new_orientation == 'east-of-north':
                return self._cartesian2EoN()
        elif self.orientation == 'sky':
            if new_orientation == 'cartesian':
                return self._sky2cartesian()
            elif new_orientation == 'east-of-north':
                return self._sky2EoN()
        elif self.orientation == 'east-of-north':
            if new_orientation == 'cartesian':
                return self._EoN2cartesian()
            elif new_orientation == 'sky':
                return self._EoN2sky()
        else:
            # shouldn't happen given above
            raise NotImplementedError(
                'Conversion not implemented for orientation pair: ' 
                f'{self.orientation} -> {new_orientation}'
                )

    def _cartesian2sky(self):
        '''
        Convert an angle measured in the standard cartesian orientation to
        the 'sky' orientation
        '''

        return OrientedAngle(
            self + self.image_rotation,
            orientation='sky',
            wrap_angle=self.wrap_angle,
            wcs=self.wcs
            )

    def _sky2cartesian(self):
        '''
        Convert an angle measured in the 'sky' orientation to
        the standard cartesian orientation
        '''

        return OrientedAngle(
            self - self.image_rotation,
            orientation='cartesian',
            wrap_angle=self.wrap_angle,
            wcs=self.wcs
            )

    def _sky2EoN(self):
        '''
        Convert an angle measured in the 'sky' orientation to
        the 'east-of-north' orientation
        '''

        return OrientedAngle(
            self - Angle('90 deg'),
            orientation='east-of-north',
            wrap_angle=self.wrap_angle,
            wcs=self.wcs
            )

    def _EoN2sky(self):
        '''
        Convert an angle measured in the 'east-of-north' orientation to
        the 'sky' orientation
        '''

        return OrientedAngle(
            Angle('90 deg') + self,
            orientation='sky',
            wrap_angle=self.wrap_angle,
            wcs=self.wcs
            )

    def _cartesian2EoN(self):
        '''
        Convert an angle measured in the standard cartesian orientation to
        the 'east-of-north' orientation
        '''

        theta_sky = self._cartesian2sky()

        return theta_sky._sky2EoN()

    def _EoN2cartesian(self):
        '''
        Convert an angle measured in the 'east-of-north' orientation to
        the standard cartesian orientation
        '''

        theta_sky = self._EoN2sky()

        return theta_sky._sky2cartesian()

    def __str__(self):
        orig = super().__str__()[0:-1]

        return f'{orig}, {self.orientation}>'

    def __repr__(self):
        orig = super().__repr__()[0:-1]

        return f'{orig}, {self.orientation}>'

    # TODO: Implement additional dunder methods for comparison & equality!