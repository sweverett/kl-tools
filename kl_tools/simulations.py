import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

verbose = False
baseURL = 'http://www.tng-project.org/api/'
headers = {"api-key":"b703779c1f099efed6f47b91607b1bb1"}


class Simulation():
    def __init__(self, pars):
        pass


class TNGsimulation():
    def __init__(self):
        pass

    @classmethod
    def from_datacube(cls, datacube, **kwargs):

        '''
        Initialize the simulats to make mocks with observation parameters
        set by the provided datacube.
        '''
        pars = {}
        sim = Simulation(pars)
        return sim

    def _getFITS(self,params=None):

        '''
        We expect the output to be a fits file.

        '''
        r = requests.get(path, params=params)

        # raise exception if response code is not HTTP SUCCESS (200)
        r.raise_for_status()
        
        # Parse this into an astropy FITS.. thing.
        hdulist = fits.open(file_stream, memmap=True, do_not_scale_image_data=True)
        # From here:
        #  https://github.com/astropy/astropy/issues/7980

        
    def _getIllustrisData(self,haloid, snap, partType = 'gas', field = 'sb_H--1-1215.67A',fov = 2.):
        #print(f"field:{field}")
        base_visurl = snap['url']+f'subhalos/{haloid}/vis.hdf5'
        # Each visualization has an associated a particle type and display field.
        imgSize = fov
        imgSizeUnit = 'arcmin'
        url = base_visurl+f'?partType={partType}&partField={field}&size={imgSize}&sizeType={imgSizeUnit}'
        # Requests writes this to a file and returns the filename. So 
        #print(f"query url: {url}")
        r = requests.get(url,headers=headers)
        f = io.BytesIO(r.content)
        h = h5py.File(f,'r')
        data = h['grid'][:] # this is where Illustris likes to put the returned data.
        # Note: pixels with no subhalo information in them will get a 'nan' in flux, but will be 99 in FLAM. Be sure to get that later.
        # Add units to this, depending on what was requested.
        if 'sb_H' in field:
            flux = (10.**data) * u.photon / u.s /u.cm**2 / u.arcsec**2
        if 'stellarBand' in field:
            # work out the source flux, in photon/s/cm2/arcsec2
            dmod = cosmo.distmod(snap['redshift'])
            mag = data*u.ABmag + dmod
        if 'nuv' in field:
            lambda_eff =2304.74*u.AA  # from the SVO filter profile service, GALEX NUV
            lambda_width = 768.31*u.AA # from the SVO filter profile service, GALEX NUV
        if 'fuv' in field: 
            lambda_eff = 1555.87*u.AA    # from the SVO filter profile service, GALEX FUV
            lambda_width = 265.57 * u.AA # from the SVO filter profile service, GALEX FUV
        if 'sdss_g' in field:
            lambda_eff = 4671.78*u.AA
            lambda_width = 1064.68*u.AA
        flux = mag.to(u.photon/u.s/u.cm**2/u.AA,u.spectral_density(lambda_eff))*lambda_width
        
    bad = ~np.isfinite(flux)
    flux[bad] = 0.
    return flux
        
            

