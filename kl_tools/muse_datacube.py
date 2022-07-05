import cube
import numpy as np
from astropy.io import fits
import galsim
import os
import pickle
import fitsio
from astropy.table import Table
import matplotlib.pyplot as plt
import galsim as gs
import pathlib
import astropy.wcs as wcs
import astropy.units as u
import ipdb
# Sublass DataCube for MUSE observations.

class MuseDataCube(cube.DataCube):

    def __init__(self,filename3d = None, filename1d = None, catalogFile=None, pars=None):
        '''
        Initialize a DataCube object from the contents of a MUSE fits datacube.
        filename3d: string, or anything parsable by pathlib.Path 
            Filename containing MUSE datacube
        filename1d: string, or anything parsable by pathlib.Path 
            Filename containing MUSE 1d calibrated spectrum.
        catalog: 
        pars: dict

        '''
        # See if we can read the 
        
        cubePath = pathlib.Path(filename3d)
        if not cubePath.exists():
            print(f"{filename3d} doesn't exist")
            ipdb.set_trace()
        specPath = pathlib.Path(filename1d)
        if not specPath.exists():
            print(f"{filename1d} doesn't exist")
            ipdb.set_trace()
        catalogPath = pathlib.Path(catalogFile)
        if not catalogPath.exists():
            print(f"{catalogFile} doesn't exist")
            ipdb.set_trace()


        spec3d = fitsio.read(cubePath)
        hdr3d = fitsio.read_header(cubePath)
        self._cubewcs = wcs.WCS(hdr3d)
        self._spec1d = fitsio.read(specPath)
        catalog = fitsio.read(catalogPath)
        objid = re.search('(\d+)_*',cubePath.name).groups()[0]
        fullCatalog = fitsio.read(catalogPath)
        self._catalogEntry = fullCatalog[fullCatalog['UNIQUE_ID'] == objid]
        
        self.Nx = spec3d.shape[1]
        self.Ny = spec3d.shape[2]
        self.Nspec = spec3d.shape[0]
        scales = self._cubewcs.proj_plane_pixel_scales()
        self.pixel_scale = np.sqrt(scales[0].to(u.arcsec)*scales[1].to(u.arcsec)).value
        self._set_parameters()
        # Looks like the likelihood code expects the datacube to have a slightly different axis order.
        datacube = np.moveaxis(spec3d,0,-1)
        # Make the list of bandpasses.
        dlam = (self._spec1d['WAVE_VAC'][1:] - self._spec1d['WAVE_VAC'][:-1])
        dlam = np.append(dlam,dlam[-1])
        # Now dlam should be an array of spaxel widths. 
        bandpasses = [gs.Bandpass(1.0, 'A', blue_limit = il-dl, red_limit = il+dl, zeropoint = 22.5) for il,dl in zip(self._spec1d['WAVE_VAC'],dlam)]

        super().__init__(data=datacube,shape=datacube.shape, bandpasses=bandpasses,pix_scale = self.pars['pix_scale'] )

        # Once everything is set up for MUSE, populate the parameters dictionary that the runner will need.
        

        
    def _set_parameters(self):
        '''
        return a parameter dictionary populated with (some) of the fields that will be needed in the likelihood parameters dictionary, based on the data.
        '''
        
        self.pars = {}
        self.pars['pix_scale'] = self.pixel_scale # Use a geometric average to keep the right pixel area, in case the pixels aren't square.
        self.pars['Nx'] = self.Nx
        self.pars['Ny'] = self.Ny
        self.pars['bandpass_throughput'] = 0.2 # A guess, based on throughput here: https://www.eso.org/sci/facilities/paranal/instruments/muse/inst.html
        self.pars['z'] = self._catalogEntry['Z']

        # These MUSE observations have a limited spectral range, between 
        
        self.line_value = 

if __name__ == '__main__':
    testpath = pathlib.Path("../tests/testdata")
    spec1dPath = testpath / pathlib.Path("spectrum_102021103.fits.gz")
    spec3dPath = testpath / pathlib.Path("102021103_objcube.fits.gz")
    catalogPath = testpath / pathlib.Path("MW_1-24_main_table.fits")
    
    # Try initializing a datacube object with these paths.
    thisCube = MuseDataCube(filename3d = spec3dPath, filename1d = spec1dPath, catalogFile=catalogPath)
    ipdb.set_trace()
