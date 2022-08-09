import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import astropy.units as u
import astropy.constants as const
from astropy import cosmology
cosmo = cosmology.Planck18_arXiv_v2


verbose = False
baseURL = 'http://www.tng-project.org/api/'
headers = {"api-key":"b703779c1f099efed6f47b91607b1bb1"}

rbase = get(baseUrl)
use_sim = 'TNG50-1'
names = [sim['name'] for sim in rbase['simulations']]
i = names.index(use_sim)
sim = get( rbase['simulations'][i]['url'] )
snaps = get( sim['snapshots'] )
snap = get( snaps[-1]['url'] )

subs = get( snap['subhalos'], {'limit':50000, 'order_by':'-mass_stars'} )


class Simulation():
    def __init__(self, pars):
        pass


class TNGsimulation(Simulation):
    def __init__(self):
        pass

    def set_redshift(self,z=0.1):
        self.z = z
    
    @classmethod
    def from_datacube(cls, datacube, **kwargs):

        '''
        Initialize the simulats to make mocks with observation parameters
        set by the provided datacube.
        '''
        pars = {}
        sim = Simulation(pars)
        self._datacube = datacube
        self._lambda1d = np.array( [(bp.red_limit + bp.blue_limit)/2. for bp in datacube.bandpasses] )
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
        nH =  (h5data['PartType0']['Density']*1e10 * u.M_sun / u.kpc**3 * h**2 / const.m_e).to(1/u.cm**3)
        ne = Xe * nH
        V = (h5data['PartType0']['Mass']* 1e10 * u.M_sun/h) / ( h5data['PartType0']['Density'] * 1e10 * u.M_sun/h / (u.kpc / h)***3)
        # Number of recombinations in this volume element
        nr = alpha * ne * nH * V

        # What's the flux from this particle at the observer?
        dL = cosmo.luminosity_distance(z)
        photon_flux = ( nr / (4 * np.pi * dL**2) ).to(1/u.cm**2/u.s)

        return photon_flux
    
    def _star_particle_flux(self, h5data):

        # This should just return continuum proportional to stellar mass.
        # Maybe an age correction.
        # Normalize a solar spectrum:
        norm = 1 * u.Watt / u.m**2 / u.nm / const.M_sun
        spec_norm = norm * (h5data['PartType4']['Masses']*u.M_sun)
        return spec_norm

    def _linespec(self,h5data,line_flux, line_center = 6563. * u.Angstrom, resolution = 5000):
        '''
        resolution -- spectral resolution at line center; unitless, means Lambda/Delta Lambda
        '''

        sigma * line_center / resolution
        # Calculate the velocity offset of each particle.
        dv = h5data['PartType0']['Velocities'][:,0] * np.sqrt(1/1+self.z) * u.km/u.s
        # get physical velocities from TNG by multiplying by sqrt(a)
        # https://www.tng-project.org/data/docs/specifications/#parttype0
        dlam = line_center *  dv / const.c
        
        line_spectra = line_flux/np.sqrt(2*np.pi*sigma**2) * np.exp(-(self._lambda1 - (line_center + dlam))**2/sigma/2.)
        return line_spectra
        
        
    def _getIllustrisTNGData(self,subhaloid=100, line_center = 6563*u.Angstrom):
        '''
        For a chosen haloid and snapshot, get the ingredients necessary to build a simulated datacube.
        This means stellar continuum (flat across our SED) and a line that traces the gas.
        '''

        sub = get( subs['results'][subhaloid]['url'] )        
        url = f"http://www.tng-project.org/api/{use_sim}/snapshots/{sub['snap']}/subhalos/{sub['id']}/cutout.hdf5"
        r = requests.get(url,headers=headers)
        f = BytesIO(r.content)
        h = h5py.File(f,mode='r')
        Temp = 
        # Assign an emission-line flux and star particle continuum level to each star and gas particle
        
        self._particleData = h
        self._particleTemp = self._calculate_gas_temperature(h)
        self._starFlux = self._star_particle_flux(h)
        self._line_flux = self._gas_line_flux(h)
        
        # Once we have these data, we then need to generate an SED for each star and gas particle!
        particle_spec = self._linespec(h,self._line_flux, line_center = line_center)
        
        
