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



rbase = get(baseURL)
use_sim = 'TNG50-1'
names = [sim['name'] for sim in rbase['simulations']]
i = names.index(use_sim)
sim = get( rbase['simulations'][i]['url'] )
snaps = get( sim['snapshots'] )
snap = get( snaps[-1]['url'] )

subs = get( snap['subhalos'], {'limit':500, 'order_by':'-mass_stars'} )




class Simulation():
    def __init__(self, pars):
        pass


class TNGsimulation(Simulation):
    def __init__(self, datacube = None):

        pass
        
    
    @classmethod
    def from_datacube(cls, datacube, **kwargs):

        '''
        Initialize the simulats to make mocks with observation parameters
        set by the provided datacube.
        '''
        pars = {}
        sim = Simulation(pars)
        
        lambda1d = np.array( [(bp.red_limit + bp.blue_limit)/2. for bp in datacube.bandpasses] )
        dlambda1d = np.array( [(bp.red_limit - bp.blue_limit) for bp in datacube.bandpasses] )
        
        sim = TNGsimulation()
        sim._datacube = datacube
        sim._lambda1d = lambda1d * u.nm
        sim.redshift = datacube.pars['z'].value[0]
        
        return sim

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
        norm = 1 * u.Watt / u.m**2 / u.nm / const.M_sun * u.AU**2
        spec_norm = norm * (h5data['PartType4']['Masses'][:]*u.M_sun)

        # Then, using our line center, convert to photons per spaxel.
        Ephot = const.h*const.c/(6563.*u.Angstrom) * 1/u.photon
        photon_density = ( spec_norm.to(u.erg/u.s/u.nm) / Ephot ).to( u.photon/u.s/u.nm)

        photons = photon_density * np.array( [(bp.red_limit - bp.blue_limit) for bp in datacube.bandpasses] )
        
        return photon_density.to(u.photon/u.s).value
        
    def _getIllustrisTNGData(self,subhaloid=100, line_center = 6563*u.Angstrom, cachefile = None):
        '''
        For a chosen haloid and snapshot, get the ingredients necessary to build a simulated datacube.
        This means stellar continuum (flat across our SED) and a line that traces the gas.

        This sets internal attributes that hold the TNG data. If a cachefile is provided, it will look there first, and use data if that file exists.
        The most cachefile most recently used by this object is stored in the '_cachefile' attribute.
        '''
        
        if cachefile == None:
            print(f"downloading data from TNG remote server. Will store results locally at {cachefile} ")
            sub = get( subs['results'][subhaloid]['url'] )        
            url = f"http://www.tng-project.org/api/{use_sim}/snapshots/{sub['snap']}/subhalos/{sub['id']}/cutout.hdf5"
            r = requests.get(url,headers=headers)
            f = BytesIO(r.content)
            h = h5py.File(f,mode='r')
            with open(cachefie,'wb') as g:
                g.write(r.content)
            
            # Assign an emission-line flux and star particle continuum level to each star and gas particle

        else:
            print(f"using locally cached file {cachefile}")
            h = h5py.File(cachefile,mode='r')
            self._cachefile = cachefile
            
        self._particleData = h
        self._particleTemp = self._calculate_gas_temperature(h)
        self._starFlux = self._star_particle_flux(h)
        self._line_flux = self._gas_line_flux(h)
        
        
        
    def _generateSpectra(self,line_center = 6563*u.Angstrom, resolution = 5000,rescale_positions = 3.):

        # What are the dimensions of the datacube?

        # What is the position of the sources relative to the field center?
        # Center on the star particles, since that's how we'll really define galaxies.
        dx = (self._particleData['PartType0']['Coordinates'][:,0] - np.median(self._particleData['PartType4']['Coordinates'][:,0]))/cosmo.h / rescale_positions
        dy = (self._particleData['PartType0']['Coordinates'][:,1] - np.median(self._particleData['PartType4']['Coordinates'][:,1]))/cosmo.h / rescale_positions
        dz = (self._particleData['PartType0']['Coordinates'][:,2] - np.median(self._particleData['PartType4']['Coordinates'][:,2]))/cosmo.h / rescale_positions

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


        # Loop over slices. In each slice, make a 2D histogram of the spectra.

        # Choose a pixelization:
        self.simcube = np.zeros_like(self._datacube.data)
        xbins = np.linspace(-self.simcube.shape[1]/2,self.simcube.shape[1]/2.,self.simcube.shape[1]+1)
        ybins = np.linspace(-self.simcube.shape[2]/2,self.simcube.shape[2]/2.,self.simcube.shape[2]+1)
        
        xcen,ycen = np.mean(xbins),np.mean(ybins)
        for islice in range(self.simcube.shape[0]):
            gas_slice2D,_,_ = np.histogram2d(du-xcen,dv-ycen,bins=(xbins,ybins),weights=line_spectra[:,islice])
            star_slice2D,_,_ = np.histogram2d(du-xcen,dv-ycen,bins=(xbins,ybins),weights=self._starFlux)
            slice2D = gas_slice2D + star_slice2D
            self.simcube[islice,:,:] = slice2D
        
        ipdb.set_trace()
                
    def getSpectrum(self,subhaloid):
        self._getIllustrisTNGData(subhaloid)
        spectrum = self._generateSpectra(self._particleData)
        

if __name__ == '__main__':
    
    
    cube_dir = os.path.join(utils.TEST_DIR, 'test_data')    
    cubefile = os.path.join(cube_dir, '102021103_objcube.fits')
    specfile = os.path.join(cube_dir, 'spectrum_102021103.fits')
    catfile = os.path.join(cube_dir, 'MW_1-24_main_table.fits')
    linefile = os.path.join(cube_dir, 'MW_1-24_emline_table.fits')
    
    print(f'Setting up MUSE datacube from file {cubefile}')
    datacube = MuseDataCube(cubefile, specfile, catfile, linefile)
    
    # default, but we'll make it explicit:
    datacube.set_line(line_choice='strongest')
    # Then make a simulation object from this datacube.
    TNGsim = TNGsimulation.from_datacube(datacube)
    # Finally, go and get the data!
    cachepath = pathlib.Path("tng_cutout.hdf5")
    # Choose the line center from the provided datacube.
    empars = datacube.pars['emission_lines'][0]
    line_center = empars.line_pars['value'] * (1 + empars.line_pars['z']) * u.Angstrom #empars.line_pars['unit']
    if cachepath.exists():
        TNGsim._getIllustrisTNGData(cachefile=cachepath, line_center = line_center)
    else:
        TNGsim._getIllustrisTNGData(line_center = line_center)
    TNGsim._generateSpectra(line_center = line_center)
    ipdb.set_trace()
    