import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import astropy.units as u
import astropy.constants as const
from astropy import cosmology
from scipy.spatial.transform import Rotation
import h5py

import requests
from io import BytesIO
import os
import pathlib

import utils
from cube import DataCube, CubePars

import ipdb

def get(path, params=None):
    # make HTTP GET request to path
    headers = {"api-key":"b703779c1f099efed6f47b91607b1bb1"}
    r = requests.get(path, params=params, headers=headers)

    # raise exception if response code is not HTTP SUCCESS (200)
    r.raise_for_status()

    if r.headers['content-type'] == 'application/json':
        return r.json() # parse json responses automatically

    if 'content-disposition' in r.headers:
        filename = r.headers['content-disposition'].split("filename=")[1]
        with open(filename, 'wb') as f:
            f.write(r.content)
        return filename # return the filename string

    return r

class TNGsimulation(object):
    def __init__(self):
        self.base_url = 'http://www.tng-project.org/api/'
        self.cosmo = cosmology.Planck18_arXiv_v2

        return

    def set_subhalo(self, subhaloid, redshift=0.1, simname = 'TNG100-1'):
        # Set the _subhalo attribute by querying the TNG catalogs.
        # Then, pull the corresponding particle data.
        self.redshift = redshift
        rbase = get(self.base_url)
        self._sim_name = simname
        names = [sim['name'] for sim in rbase['simulations']]
        i = names.index(simname)
        sim = get( rbase['simulations'][i]['url'] )
        snaps = get( sim['snapshots'] )
        snap_redshifts = np.array([snap['redshift'] for snap in snaps])

        snapurl = snaps[np.argmin(np.abs(snap_redshifts - redshift))]['url']
        suburl = snapurl+f'subhalos/{subhaloid}'
        print(f"closest snapshot to desired redshift {redshift:.04} is at {snapurl} ")
        self._subhalo = get(suburl)
        self._snapshot = snap
        self._getIllustrisTNGData()

        return

    def _calculate_gas_temperature(self,h5data):
        u           = h5data['PartType0']['InternalEnergy'][:]    #  the Internal Energy
        Xe          = h5data['PartType0']['ElectronAbundance'][:]  # xe (=ne/nH)  the electron abundance
        XH          = 0.76             # the hydrogen mass fraction
        gamma        = 5.0/3.0          # the adiabatic index
        KB          = 1.3807e-16       # the Boltzmann constant in CGS units  [cm^2 g s^-2 K^-1]
        mp          = 1.6726e-24       # the proton mass  [g]
        little_h    = 0.704                 # NOTE: 0.6775 for all TNG simulations
        mu          = (4*mp)/(1+3*XH+4*XH*Xe)
        # Estimate temperature
        temperature = (gamma-1)* (u/KB)* mu* 1e10

        return temperature

    def _gas_line_flux(self, h5data):
        T = self._calculate_gas_temperature(h5data)
        h = self.cosmo.h # hubble parameter
        alpha = 2.6e-13 * (T/1e4)**(-0.7)  * u.cm**3 / u.s
        Xe = h5data['PartType0']['ElectronAbundance'][:]
        XH = 0.76
        nH =  (h5data['PartType0']['Density'][:]*1e10 * u.M_sun / u.kpc**3 * h**2 / const.m_e).to(1/u.cm**3)
        ne = Xe * nH
        V = (h5data['PartType0']['Masses'][:]* 1e10 * u.M_sun/h) / ( h5data['PartType0']['Density'][:] * 1e10 * u.M_sun/h / (u.kpc / h)**3)
        # Number of recombinations in this volume element
        nr = alpha * ne * nH * V

        # What's the flux from this particle at the observer?
        dL = self.cosmo.luminosity_distance(self.redshift)
        photon_flux = ( nr / (4 * np.pi * dL**2) ).to(1/u.cm**2/u.s)

        return photon_flux

    def _star_particle_flux(self, h5data):
        # This should just return continuum proportional to stellar mass.
        # Maybe an age correction.
        # Normalize a solar spectrum:
        norm = 1 * u.Watt / u.m**2 / u.nm / const.M_sun
        spec_norm = norm * (h5data['PartType4']['Masses'][:]*u.M_sun)

        return spec_norm

    def _getIllustrisTNGData(self):
        '''
        For a chosen haloid and snapshot, get the ingredients necessary to build a simulated datacube.
        This means stellar continuum (flat across our SED) and a line that traces the gas.

        This sets internal attributes that hold the TNG data. If a cachefile is provided, 
          it will look there first, and use data if that file exists.
        The most cachefile most recently used by this object is stored in the '_cachefile' attribute.
        '''
        sub  = self._subhalo
        cachepath = pathlib.Path(f"./TNGcache_{self._sim_name}_subhalo_{sub['id']}.hdf5")
        if not cachepath.exists():
            url = f"http://www.tng-project.org/api/{self._sim_name}/snapshots/{sub['snap']}/subhalos/{sub['id']}/cutout.hdf5"
            r = requests.get(url,headers=headers)
            f = BytesIO(r.content)
            h = h5py.File(f,mode='r')
        else:
            h = h5py.File(cachefile,mode='r')
            self._cachefile = cachefile

        self._particleData = h
        self._particleTemp = self._calculate_gas_temperature(h)
        self._starFlux = self._star_particle_flux(h)
        self._line_flux = self._gas_line_flux(h)


        return

    def _generateCube(self, pars):
        '''
        pars: cube.CubePars
            A CubePars instance that holds all relevant metadata about the
            desired instrument and DataCube parameters needed to render the
            TNG object
        '''

        #----------------------------------------------------------------------
        # unpacking parmeters step
        # needed parameters:
        #   - pixel scale [arcsec/pixel, scalar]
        #   - vector of wavelength spaxel centers [angstroms]
        #   - spectral resolution [ scalar float, lambda/delta_lambda]
        #   - redshift (default to _requested_ subhalo redshift) [scalar float]
        #   - dimensions of the datacube [nlam, nx, ny]
        # What are the dimensions of the datacube?
        #
        # Here's how to grab the following from CubePars; can be deleted by
        # Eric later

        pixel_scale = pars['pix_scale']
        shape = pars['shape']

        # each element of the following list is an EmissionLine object
        lines = pars['emission_lines'] # may be an empty list

        # list of tuples (lambda_blue, lambda_red) w/ associated astropy unit for
        # each slice
        lambda_bounds = pars['wavelengths']

        # get list of slice wavelength midpoints
        lambdas = [
            np.mean([l[0], l[1]] for l in lambda_bounds)
        ]

        if 'psf' in pars:
            psf = pars['psf']
        else:
            psf = None

        # you might want to do something like the following:
        # for line in line:
        #     line_sed = line.sed
        #     ...

        #----------------------------------------------------------------------

        # What is the position of the sources relative to the field center?
        dx = (self._particleData['PartType0']['Coordinates'][:,0] - np.mean(self._particleData['PartType0']['Coordinates'][:,0]))/self.cosmo.h
        dy = (self._particleData['PartType0']['Coordinates'][:,1] - np.mean(self._particleData['PartType0']['Coordinates'][:,1]))/self.cosmo.h
        dz = (self._particleData['PartType0']['Coordinates'][:,2] - np.mean(self._particleData['PartType0']['Coordinates'][:,2]))/self.cosmo.h

        # TODO: add an optional rotation matrix operation.
        RR = Rotation.from_euler('z',45,degrees=True)
        
        # Calculate the velocity offset of each particle.
        dv = self._particleData['PartType0']['Velocities'][:,2] * np.sqrt(1/1+self.redshift) * u.km/u.s
        # get physical velocities from TNG by multiplying by sqrt(a)
        # https://www.tng-project.org/data/docs/specifications/#parttype0
        dlam = line_center *  dv / const.c
        line_spectra = self._line_flux[:,np.newaxis]/np.sqrt(2*np.pi*sigma**2) * np.exp((-(self._lambda1d - (line_center + dlam)[:,np.newaxis])**2/sigma**2/2.).value)
        # Now put these on the pixel grid.
        du = (dx*u.kpc / self.cosmo.angular_diameter_distance(self.redshift)).to(u.dimensionless_unscaled).value * 180/np.pi * 3600 / self._datacube.pars['pix_scale']
        dv = (dy*u.kpc / self.cosmo.angular_diameter_distance(self.redshift)).to(u.dimensionless_unscaled).value * 180/np.pi * 3600 / self._datacube.pars['pix_scale']
        # TODO: This is where we should apply a shear.
        # Round each one to the pixel center
        du_int = (np.round(du)).astype(int) # + x grid size
        dv_int = (np.round(dv)).astype(int) # + y grid size
        self.simcube = np.zeros_like(self._datacube.data)
        for i in range(self.simcube.shape[1]):
            for j in range(self.simcube.shape[2]):
                these = (du_int == i) & (dv_int == j)
                self.simcube[:,i,j] = np.sum(line_spectra[these,:],axis=0)

        return

    def getSpectrum(self, subhaloid):
        self._getIllustrisTNGData(subhaloid)
        spectrum = self._generateSpectra(self._particleData)

        return

    def from_slit(self):
        pass

    def to_slit(self):
        pass

    def to_cube(self, pars, shape=None):
        '''
        This method is used to render the TNG object onto a fresh datacube,
        given a CubePars instance

        pars: cube.CubePars
            A CubePars instance that sets the paramters of the instrument &
            desired DataCube
        shape: (Nspec, Nx, Ny) tuple
            A tuple describing the desired DataCube shape, if not already
            present in pars
        '''

        if not isinstance(pars, CubePars):
            raise TypeError('Must pass a CubePars object!')

        if shape is None:
            if 'shape' not in pars:
                raise KeyError('Must set the datacube shape in CubePars ' +\
                               'if not passed explicitly!')
            shape = pars['shape']
        else:
            if 'shape' in pars:
                raise AttributeError('Cannot pass a shape if there is ' +\
                                     'already one stored in CubePars!')
            # parsing done in function
            pars.set_shape(shape)

        # The `emission_lines` field is not requrired for a CubePars object.
        # If it is not present, then we will build an empty list and have the
        # datacube generation simply set a continuum
        if 'emission_lines' not in pars:
            pars['emission_lines']  = []

        # generate cube data given passed pars & emission lines
        data = self._generateCube(pars)

        return DataCube(data, pars=pars)

    def from_cube(self, datacube):
        '''
        This method is used to render the TNG object onto an existing,
        empty datacube (which has CubePars already set)

        datacube: cube.DataCube
            An DataCube instance on which to render the TNG object
        '''

        if not isinstance(datacube, DataCube):
            raise TypeError('Must pass a DataCube object!')

        pars = datacube.pars

        # The `emission_lines` field is not requrired for a CubePars object.
        # If it is not present, then we will build an empty list and have the
        # datacube generation simply set a continuum
        if 'emission_lines' not in pars:
            pars['emission_lines']  = []

        # generate cube data given passed pars & emission lines
        data = self._generateCube(pars)

        # override any existing data in the cube, while keeping all metadata
        datacube.set_data(data)

        return datacube

if __name__ == '__main__':
    sim = TNGsimulation()
    sim.set_subhalo(0)
    ipdb.set_trace()
