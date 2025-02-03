'''
Helper functions for creating objects useful for testing
'''

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import Angle

def create_wcs_with_rotation(angle):
    '''
    Create an astropy WCS object with a specified rotation angle.

    Parameters:
    - angle: astropy.coordinates.Angle object specifying the rotation angle.

    Returns:
    - wcs: astropy.wcs.WCS object with the specified rotation.
    '''

    # just in case
    angle = Angle(angle)

    # Create a FITS header
    header = fits.Header()

    # Set the WCS parameters
    header['CTYPE1'] = 'RA---TAN'
    header['CTYPE2'] = 'DEC--TAN'
    header['CUNIT1'] = 'deg'
    header['CUNIT2'] = 'deg'
    header['CRPIX1'] = 0.5  # Reference pixel along x-axis
    header['CRPIX2'] = 0.5  # Reference pixel along y-axis
    header['CRVAL1'] = 0.0  # RA at the reference pixel
    header['CRVAL2'] = 0.0  # DEC at the reference pixel
    header['CDELT1'] = (0.5 / 3600)  # Pixel scale in degrees/pixel
    header['CDELT2'] = (0.5 / 3600)   # Pixel scale in degrees/pixel

    # Convert the rotation angle to radians
    angle_rad = angle.to('rad').value

    # Compute the rotation matrix elements
    cos = np.cos(angle_rad)
    sin = np.sin(angle_rad)

    # Apply the rotation to the CD matrix
    header['PC1_1'] = cos
    header['PC1_2'] = -sin
    header['PC2_1'] = sin
    header['PC2_2'] = cos

    # Create the WCS object
    wcs = WCS(header)

    return wcs