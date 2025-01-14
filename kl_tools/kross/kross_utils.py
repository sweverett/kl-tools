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
