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
from muse import MuseDataCube

import ipdb

cosmo = cosmology.Planck18_arXiv_v2

verbose = False
baseURL = 'http://www.tng-project.org/api/'
headers = {"api-key":"b703779c1f099efed6f47b91607b1bb1"}



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

class TNGsimulation():
    def __init__(self):
        pass


    def set_subhalo(self, subhaloid, redshift=0.1, simname = 'TNG100-1'):
        # Set the _subhalo attribute by querying the TNG catalogs.
        # Then, pull the corresponding particle data.
        self.redshift = redshift
        rbase = get(baseURL)
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
        self._getIllustrisTNGData()
        
        
        
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
        h = cosmo.h # hubble parameter
        alpha = 2.6e-13 * (T/1e4)**(-0.7)  * u.cm**3 / u.s 
        Xe = h5data['PartType0']['ElectronAbundance'][:]
        XH = 0.76
        nH =  (h5data['PartType0']['Density'][:]*1e10 * u.M_sun / u.kpc**3 * h**2 / const.m_e).to(1/u.cm**3)
        ne = Xe * nH
        V = (h5data['PartType0']['Masses'][:]* 1e10 * u.M_sun/h) / ( h5data['PartType0']['Density'][:] * 1e10 * u.M_sun/h / (u.kpc / h)**3)
        # Number of recombinations in this volume element
        nr = alpha * ne * nH * V

        # What's the flux from this particle at the observer?
        dL = cosmo.luminosity_distance(self.redshift)
        photon_flux = ( nr / (4 * np.pi * dL**2) ).to(1/u.cm**2/u.s)
        return photon_flux    

    def _star_particle_flux(self, h5data):
        # This should just return continuum proportional to stellar mass.
        # Maybe an age correction.
        # Normalize a solar spectrum:
        norm = 1 * u.Watt / u.m**2 / u.nm / const.M_sun
        spec_norm = norm * (h5data['PartType4']['Masses'][:]*u.M_sun)
        return spec_norm        

    def _getIllustrisTNGData(self, cachefile = None):
        '''
        For a chosen haloid and snapshot, get the ingredients necessary to build a simulated datacube.
        This means stellar continuum (flat across our SED) and a line that traces the gas.

        This sets internal attributes that hold the TNG data. If a cachefile is provided, 
          it will look there first, and use data if that file exists.
        The most cachefile most recently used by this object is stored in the '_cachefile' attribute.
        '''
        if cachefile == None:
            sub = self._subhalo# get( subs['results'][subhaloid]['url'] )
            url = f"http://www.tng-project.org/api/{self._sim_name}/snapshots/{sub['snap']}/subhalos/{sub['id']}/cutout.hdf5"
            r = requests.get(url,headers=headers)
            f = BytesIO(r.content)
            h = h5py.File(f,mode='r')
        
            # Assign an emission-line flux and star particle continuum level to each star and gas particle

        else:
            h = h5py.File(cachefile,mode='r')
            self._cachefile = cachefile
            
        self._particleData = h
        self._particleTemp = self._calculate_gas_temperature(h)
        self._starFlux = self._star_particle_flux(h)
        self._line_flux = self._gas_line_flux(h)

        
    def _generateCube(self, some_parameters:cubepars instance):


        # unpacking parmeters step
        # needed parameters:
        #   - pixel scale [arcsec/pixel, scalar]
        #   - vector of wavelength spaxel centers [angstroms]
        #   - spectral resolution [ scalar float, lambda/delta_lambda]
        #   - redshift (default to _requested_ subhalo redshift) [scalar float]
        #   - dimensions of the datacube [nlam, nx, ny]
        

        # What are the dimensions of the datacube?

        # What is the position of the sources relative to the field center?
        dx = (self._particleData['PartType0']['Coordinates'][:,0] - np.mean(self._particleData['PartType0']['Coordinates'][:,0]))/cosmo.h
        dy = (self._particleData['PartType0']['Coordinates'][:,1] - np.mean(self._particleData['PartType0']['Coordinates'][:,1]))/cosmo.h
        dz = (self._particleData['PartType0']['Coordinates'][:,2] - np.mean(self._particleData['PartType0']['Coordinates'][:,2]))/cosmo.h

        # TODO: add an optional rotation matrix operation.
        RR = Rotation.from_euler('z',45,degrees=True)

        sigma = line_center / resolution
        # Calculate the velocity offset of each particle.
        dv = self._particleData['PartType0']['Velocities'][:,2] * np.sqrt(1/1+self.redshift) * u.km/u.s
        # get physical velocities from TNG by multiplying by sqrt(a)
        # https://www.tng-project.org/data/docs/specifications/#parttype0
        dlam = line_center *  dv / const.c
        delta_lambda = self._lambda1d[1:] - self._lambda1d[:-1]
        line_spectra = self._line_flux[:,np.newaxis]/np.sqrt(2*np.pi*sigma**2) * np.exp((-(self._lambda1d - (line_center + dlam)[:,np.newaxis])**2/sigma**2/2.).value)
        # Now put these on the pixel grid.
        du = (dx*u.kpc / cosmo.angular_diameter_distance(self.redshift)).to(u.dimensionless_unscaled).value * 180/np.pi * 3600 / self._datacube.pars['pix_scale']
        dv = (dy*u.kpc / cosmo.angular_diameter_distance(self.redshift)).to(u.dimensionless_unscaled).value * 180/np.pi * 3600 / self._datacube.pars['pix_scale']

        # TODO: This is where we should apply a shear.
        
        # Round each one to the pixel center
        du_int = (np.round(du)).astype(int)
        dv_int = (np.round(dv)).astype(int)
        self.simcube = np.zeros_like(self._datacube.data)
        for i in range(self.simcube.shape[1]):
            for j in range(self.simcube.shape[2]):
                these = (du_int == i) & (dv_int == j)
                self.simcube[:,i,j] = np.sum(line_spectra[these,:],axis=0)
        ipdb.set_trace()

                
    def getSpectrum(self,subhaloid):
        self._getIllustrisTNGData(subhaloid)
        spectrum = self._generateSpectra(self._particleData)        
        
        
    
    def from_slit(self):
        pass

    def to_slit(self):
        pass

    def to_cube(self, parameters):
        pass

    def from_cube(self, DataCube):
        pass
    


if __name__ == '__main__':
    sim = TNGsimulation()
    sim.set_subhalo(0)
    ipdb.set_trace()
