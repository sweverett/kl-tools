import cube
from astropy.io import fits
import galsim
import os
import pickle
from astropy.table import Table
import matplotlib.pyplot as plt
import galsim as gs
import pathlib

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

        spec1d = fitsio.read(specPath)
        cube = fitsio.read(cubePath)
        entry = fitsio.read(catalogPath)
        self.Nx = cube.shape[1]
        self.Ny = cube.shape[2]
        self.Nspec = cube.shape[0]
        # Looks like the likelihood code expects the datacube to be in a slightly different format.
        newcube = np.moveaxis(cube,0,-1)
        self._check_shape_params()
        self.shape = newcube.shape
        self._data = newcube
        self._lambda_vac = spec1d(["WAV_VAC"]

        def _set_bandpasses(self):
            
