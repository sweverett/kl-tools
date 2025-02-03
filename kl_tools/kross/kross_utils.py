'''
Various helper methods for the KROSS measurement. Should be refined & refactored
in the future
'''

from pathlib import Path
import numpy as np
from astropy.units import Unit as u
import matplotlib.pyplot as plt
import fitsio
from glob import glob
from astropy.coordinates import SkyCoord
from astropy.units import deg

from kl_tools.utils import get_base_dir

def theta2pars(theta, r_unit=u('arcsec'), v_unit=u('km/s')):
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
        'r_unit': r_unit,
        'v_unit': v_unit
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

def get_kross_obj_data(kid, vb=False):
    '''
    Get all of the data for a KROSS object, given the KID

    Parameters
    ----------
    kid : int
        The KROSS ID of the object
    vb : bool, optional
        If True, print out the files that are not found
    '''

    kross_dir = get_base_dir() / 'data/kross'
    cosmo_dir = get_base_dir() / 'data/cosmos'
    kross_data = fitsio.read(kross_dir / 'kross_release_v2.fits', ext=1)

    # need all the coords for a match to the COSMOS cutouts
    kross_ra = kross_data['RA']
    kross_dec = kross_data['DEC']
    kross_names = kross_data['NAME']
    kross_coords = SkyCoord(ra=kross_ra*deg, dec=kross_dec*deg)

    # clean up the names
    for i, name in enumerate(kross_names):
        new = name.strip()
        kross_names[i] = new

    # get the KROSS sky coordinates to match to the COSMOS cutouts
    cosmo_files = glob(str(cosmo_dir / 'cutouts/*.fits'))
    cosmo_cutouts = {}
    for f in cosmo_files:
        fname = Path(f).name

        # match to the KROSS sky coordinates to get the obj name
        ra  = float(fname.split('_')[1])
        dec = float(fname.split('_')[2])
        coord = SkyCoord(ra=ra*deg, dec=dec*deg)

        indx, sep, _ = coord.match_to_catalog_sky(kross_coords)
        cosmo_cutouts[kross_names[indx]] = f

    row = kross_data[kross_data['KID'] == kid] 
    name = row['NAME'][0].strip()
    ra = kross_data['RA']
    dec = kross_data['DEC']

    cube_file = kross_dir / 'cubes' / f'{name}.fits'
    velocity_file = kross_dir / 'vmaps' / f'{name}.fits' 
    sigma_file = kross_dir / 'disp' / f'{name}.fits' 
    halpha_file = kross_dir / 'halpha' / f'{name}.fits'
    hst_file = cosmo_cutouts[name]

    # grab all of the object data
    try:
        cube, cube_hdr = fitsio.read(cube_file, header=True)
    except FileNotFoundError:
        if vb is True:
            print(f'Cube file not found: {cube_file}')
        cube, cube_hdr = None
    try:
        velocity, velocity_hdr = fitsio.read(velocity_file, header=True)
    except FileNotFoundError:
        if vb is True:
            print(f'Velocity map file not found: {velocity_file}')
        velocity, velocity_hdr = None, None
    try:
        sigma, sigma_hdr = fitsio.read(sigma_file, header=True)
    except FileNotFoundError:
        if vb is True:
            print(f'Sigma file not found: {sigma_file}')
        sigma, sigma_hdr = None, None
    try:
        halpha, halpha_hdr = fitsio.read(halpha_file, header=True)
    except FileNotFoundError:
        if vb is True:
            print(f'Halpha file not found: {halpha_file}')
        halpha, halpha_hdr = None, None
    try:
        hst, hst_hdr = fitsio.read(hst_file, header=True)
    except FileNotFoundError:
        if vb is True:
            print(f'HST file not found: {hst_file}')
        hst, hst_hdr = None, None

    obj_data = {
        'catalog': row,
        'cube': cube,
        'cube_hdr': cube_hdr,
        'cube_file': cube_file,
        'velocity': velocity,
        'velocity_hdr': velocity_hdr,
        'velocity_file': velocity_file,
        'sigma': sigma,
        'sigma_hdr': sigma_hdr,
        'halpha': halpha,
        'halpha_hdr': halpha_hdr,
        'halpha_file': halpha_file,
        'hst': hst,
        'hst_hdr': hst_hdr,
        'hst_file': hst_file
    }

    return obj_data
