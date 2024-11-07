'''
Various helper methods for the KROSS measurement. Should be refined & refactored
in the future
'''

import numpy as np
from astropy.units import Unit as u
import matplotlib.pyplot as plt

def theta2pars(theta):
    '''
    Map a fixed array of parameters to a dict of vmap parameters.

    We have more general tools to handle this, but this is a simple
    way to handle the fixed parameters in the KROSS model.
    '''

    pars = {
        'v0': theta[0],
        'vcirc': theta[1],
        'rscale': theta[2],
        'sini': theta[3],
        'theta_int': theta[4],
        'g1': theta[5],
        'g2': theta[6],
        'x0': theta[7],
        'y0': theta[8],
        'r_unit': u('arcsec'),
        'v_unit': u('km/s'),
    }

    return pars

def pars2theta(pars):
    '''
    Map a dict of vmap parameters to a fixed array of parameters for the fitter

    We have more general tools to handle this, but this is a simple
    way to handle the fixed parameters in the KROSS model.
    '''

    theta = np.array([
        pars['v0'],
        pars['vcirc'],
        pars['rscale'],
        pars['sini'],
        pars['theta_int'],
        pars['g1'],
        pars['g2'],
        pars['x0'],
        pars['y0'],
    ])

    return theta

def plot_line_on_image(im, cen, angle, ax, c='k', ls='--', label=None):
    """
    Plots a line on an image at a given angle using WCS projection.
    
    Parameters:
    im : Image data
    cen: Center of the image in pixel coordinates (x0, y0)
    angle : OrientedAngle of PA
    ax : Matplotlib Axes object
    c: color
    """

    # Get the dimensions of the image
    Nrow, Ncol = im.shape

    # gal center
    xgal, ygal = cen

    xc = Ncol / 2. + xgal
    yc = Nrow / 2. + ygal
    
    # Calculate the angle in radians
    angle_rad = angle.cartesian.rad

    if angle_rad > np.pi/2:
        angle_rad -= np.pi

    x0, x1 = 0, Ncol - 1

    y0 = np.tan(angle_rad) * (x0 - xc) + yc
    y1 = np.tan(angle_rad) * (x1 - xc) + yc

    # Create the line coordinates in pixel space
    line_x = np.array([x0, x1])
    line_y = np.array([y0, y1])
    
    ax.plot(
        line_x,
        line_y,
        color=c,
        linewidth=2,
        ls=ls,
        label=label
        )
    plt.xlim(0, Ncol)
    plt.ylim(0, Nrow)

    return
