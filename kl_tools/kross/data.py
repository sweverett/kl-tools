import fitsio
from pathlib import Path
from glob import glob
from astropy.coordinates import SkyCoord
from astropy.io.fits import Header
import astropy.units as u

from kl_tools.utils import get_base_dir

def get_kross_obj_data(kid, load_data=True, vb=False):
    '''
    Get all of the data for a single KROSS object, given the KID

    Parameters
    ----------
    kid : int
        The KROSS ID of the object
    load_data : bool, optional. Default=True
        If True, load the data into memory. Otherwise, just return the file 
        paths
    vb : bool, optional. Default=False
        If True, print out the files that are not found
    '''

    kross_dir = get_base_dir() / 'data/kross'
    cosmo_dir = get_base_dir() / 'data/cosmos'
    kross_data = fitsio.read(kross_dir / 'kross_release_v2.fits', ext=1)

    # need all the coords for a match to the COSMOS cutouts
    kross_ra = kross_data['RA']
    kross_dec = kross_data['DEC']
    kross_names = kross_data['NAME']
    kross_coords = SkyCoord(ra=kross_ra*u.deg, dec=kross_dec*u.deg)

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
        coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)

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
    if load_data is True:
        try:
            cube, cube_hdr = fitsio.read(cube_file, header=True)
            # cube_hdr = Header(cube_hdr)
        except FileNotFoundError:
            if vb is True:
                print(f'Cube file not found: {cube_file}')
            cube, cube_hdr = None
        try:
            velocity, velocity_hdr = fitsio.read(velocity_file, header=True)
            # velocity_hdr = Header(velocity_hdr)
        except FileNotFoundError:
            if vb is True:
                print(f'Velocity map file not found: {velocity_file}')
            velocity, velocity_hdr = None, None
        try:
            sigma, sigma_hdr = fitsio.read(sigma_file, header=True)
            # sigma_hdr = Header(sigma_hdr)
        except FileNotFoundError:
            if vb is True:
                print(f'Sigma file not found: {sigma_file}')
            sigma, sigma_hdr = None, None
        try:
            halpha, halpha_hdr = fitsio.read(halpha_file, header=True)
            # halpha_hdr = Header(halpha_hdr)
        except FileNotFoundError:
            if vb is True:
                print(f'Halpha file not found: {halpha_file}')
            halpha, halpha_hdr = None, None
        try:
            hst, hst_hdr = fitsio.read(hst_file, header=True)
            # hst_hdr = Header(hst_hdr)

            # NOTE: the HST image cutouts we've been using have some errors
            # in them; handle them now
            for i, key in enumerate(hst_hdr):
                if (key == None) or (key == 'None'):
                    hst_hdr.delete(key)
        except FileNotFoundError:
            if vb is True:
                print(f'HST file not found: {hst_file}')
            hst, hst_hdr = None, None
    else:
        cube, cube_hdr = None, None
        velocity, velocity_hdr = None, None
        sigma, sigma_hdr = None, None
        halpha, halpha_hdr = None, None
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
        'sigma_file': sigma_file,
        'halpha': halpha,
        'halpha_hdr': halpha_hdr,
        'halpha_file': halpha_file,
        'hst': hst,
        'hst_hdr': hst_hdr,
        'hst_file': hst_file
    }

    return obj_data