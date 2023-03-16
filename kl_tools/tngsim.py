import numpy as np
import matplotlib.pyplot as plt
import pickle
from astropy.io import fits
import astropy.units as u
import astropy.constants as const
from astropy import cosmology
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation
from scipy.stats import binned_statistic_2d, binned_statistic
from collections import defaultdict
import h5py
import fsps

import requests
from io import BytesIO
import os
import pathlib
import sys
sys.path.insert(0, './grism_modules')
import utils
from cube import DataCube, CubePars
import muse
import grism
import emission
from emission import LINE_LAMBDAS

from tqdm import tqdm
import ipdb


def gethdr():
    return {"api-key":"b703779c1f099efed6f47b91607b1bb1"}

def get(path, params=None):
    # make HTTP GET request to path

    
    headers = gethdr()#{"api-key":"b703779c1f099efed6f47b91607b1bb1"}
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
        self.cosmo = cosmology.Planck18
        return

    @property
    def gasPrtl(self):
        return self._particleData['PartType0']

    @property
    def starPrtl(self):
        return self._particleData['PartType4']

    @property
    def dmPrtl(self):
        return self._particleData['PartType1']

    def set_subhalo(self, subhaloid, redshift=0.5, simname = 'TNG50-1',
                    move_to_redshift = None):
        # Set the _subhalo attribute by querying the TNG catalogs.
        # Then, pull the corresponding particle data.
        self.redshift = redshift
        if move_to_redshift is not None:
            self.move_to_z = move_to_redshift
        else:
            self.move_to_z = self.redshift
        self.DA = self.cosmo.angular_diameter_distance(self.move_to_z)
        self.dL = self.cosmo.luminosity_distance(self.move_to_z)
        rbase = get(self.base_url)
        self._sim_name = simname
        names = [sim['name'] for sim in rbase['simulations']]
        i = names.index(simname)
        sim = get( rbase['simulations'][i]['url'] )
        snaps = get( sim['snapshots'] )
        snap_redshifts = np.array([snap['redshift'] for snap in snaps])
        self._snapshot = snaps[np.argmin(np.abs(snap_redshifts - redshift))]
        snapurl = self._snapshot['url']
        suburl = snapurl+f'subhalos/{subhaloid}'
        print(f"closest snapshot to desired redshift {redshift:.04} is at {snapurl} ")
        self._subhalo = get(suburl)
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
        photon_flux = ( nr / (4 * np.pi * self.dL**2) ).to(1/u.cm**2/u.s)
        return photon_flux

    def ObsSED(self, pars, h5data, method=2, gas_weight=1):
        _crt = (self._line_flux.value > 1e-5) & \
                (np.isfinite(self._line_flux.value))
        inds = np.arange(self.gasPrtl['Coordinates'][:,0].size)[_crt]
        
        lambdas = pars['wavelengths']*pars._lambda_unit
        waves = np.mean(lambdas, axis=1).to('nm').value
        _sed = np.zeros(waves.shape)
        _gas_interp = self.gasObsSED(pars, h5data, method)
        _star_interp = self.starObsSED(pars, h5data)
        _sed += _gas_interp(waves) * np.sum(self._line_flux[inds].value) * gas_weight
        _sed += _star_interp(waves) * np.sum(self.starPrtl['Masses'][self._starIDs]*1e10/self.cosmo.h)
        return interp1d(waves, _sed)
        
    
    def gasObsSED(self, pars, h5data, method=2):
        ''' Calculate obs-frame SED of gas particles
        Return: GrismSED object, FLAM [photons/s/cm2/nm]
        '''
        if method==1:
            return grism.GrismSED(pars['sed'])
        
        elif method==2:
            lambdas = pars['wavelengths']*pars._lambda_unit
            lr, lb = lambdas[-1,1].to('nm').value, lambdas[0,0].to('nm').value
            dl = np.mean(lambdas[:,1] - lambdas[:,0]).to('nm').value
            waves = np.mean(lambdas, axis=1).to('nm').value
            print(lb, lr)
            _sed_pars = {
                'lblue': lb,
                'lred': lr,
                'resolution': dl,
                'unit': u.nm,
            }
            elines = []
            _sed = np.zeros(waves.shape)
            for k,v in pars['sed']['lines'].items():
                _line_pars = {
                    'value': LINE_LAMBDAS[k].to('nm').value,
                    'R': LINE_LAMBDAS[k].to('nm').value/pars['sed']['line_sigma_int'][k],
                    'z': self.move_to_z,
                    'unit': u.nm,
                } 
                _line = emission.EmissionLine(_line_pars, _sed_pars) # /nm
                elines.append(_line)
                _sed += _line.sed(waves) # 1/nm
            return interp1d(waves, _sed, bounds_error=False, fill_value = 0)
        
    def starObsSED(self, pars, h5data):
        ''' Calculate obs-frame SED of star particles (not including winds)
        Return: scipy.interpolate.interp1d object, FLAM/Msol [photons/s/cm2/nm/Msol]
        '''
        self._starIDs = np.where(self.starPrtl['GFM_StellarFormationTime'][:]>0)[0]
        # get the mean star formation time, metalicity, and stellar mass
        GFM_SFT = self.starPrtl['GFM_StellarFormationTime'][:]
        GFM_SM  = self.starPrtl['GFM_InitialMass'][:]
        GFM_Z   = np.log10(self.starPrtl['GFM_Metallicity'][:]/0.0127)
        mean_SFT = np.average(GFM_SFT[self._starIDs], 
            weights=GFM_SM[self._starIDs])
        mean_Z = np.average(GFM_Z[self._starIDs],
            weights=GFM_SM[self._starIDs])
        # set up FSPS & get continuum per solar mass
        rest_spec = fsps.StellarPopulation(compute_vega_mags=False, zcontinuous=1,
                            sfh=0, logzsol=mean_Z, nebemlineinspec=False, 
                            add_agb_dust_model=False,add_dust_emission=False,
                            smooth_velocity=False,
                            dust_type=2, dust2=0.2, imf_type=1)
        _wave, _spec = rest_spec.get_spectrum(tage=mean_SFT, peraa=True)
        _photon_rate = _spec * (const.L_sun) / (4*np.pi*self.dL**2) / (const.h*const.c/_wave) 
        #Fsun = (const.L_sun/(4*np.pi*self.dL**2)).to('erg s-1 cm-2')
        # F_lambda per Msun (photons/s/cm2/nm/Msol), as a function of wavelength in nm
        sed_continuum = interp1d(_wave*(1+self.move_to_z)/10, _photon_rate.to('cm-2 s-1 nm-1'))

        return sed_continuum
    def _star_particle_flux(self, h5data):
        # This should just return continuum proportional to stellar mass.
        # Maybe an age correction.
        # Normalize a solar spectrum:
        # select star particles
        self._starIDs = np.where(self.starPrtl['GFM_StellarFormationTime'][:]>0)[0]
        # get the mean star formation time, metalicity, and stellar mass
        GFM_SFT = self.starPrtl['GFM_StellarFormationTime'][:]
        GFM_SM  = self.starPrtl['GFM_InitialMass'][:]
        GFM_Z   = np.log10(self.starPrtl['GFM_Metallicity'][:]/0.0127)
        mean_SFT = np.average(GFM_SFT[self._starIDs], 
            weights=GFM_SM[self._starIDs])
        mean_Z = np.average(GFM_Z[self._starIDs],
            weights=GFM_SM[self._starIDs])
        # set up FSPS & get continuum per solar mass
        sp = fsps.StellarPopulation(compute_vega_mags=False, zcontinuous=1,
                            sfh=0, logzsol=mean_Z, nebemlineinspec=False, 
                            add_agb_dust_model=False,add_dust_emission=False,
                            smooth_velocity=False,
                            dust_type=2, dust2=0.2, imf_type=1)
        _wave, _spec = sp.get_spectrum(tage=mean_SFT, peraa=True)
        Fsun = ((const.L_sun/(4*np.pi*self.dL**2)).to('erg s-1 cm-2')).value
        # F_lambda per M_sun
        sed_continuum = interp1d(_wave, _spec * Fsun)

        #norm = 1 * u.Watt / u.m**2 / u.nm / const.M_sun
        #spec_norm = norm * (h5data['PartType4']['Masses'][:]*u.M_sun)

        return sed_continuum

    def _getIllustrisTNGData(self):
        '''
        For a chosen haloid and snapshot, get the ingredients necessary to build a simulated datacube.
        This means stellar continuum (flat across our SED) and a line that traces the gas.

        This sets internal attributes that hold the TNG data. If a cachefile is provided,
          it will look there first, and use data if that file exists.
        The most cachefile most recently used by this object is stored in the '_cachefile' attribute.
        '''
        sub  = self._subhalo
        cachepath = pathlib.Path(
            f'{utils.CACHE_DIR}/{self._sim_name}_subhalo_{sub["id"]}_{self._snapshot["number"]}.hdf5'
            )
        if not cachepath.exists():
            url = f'http://www.tng-project.org/api/{self._sim_name}/snapshots/{sub["snap"]}/subhalos/{sub["id"]}/cutout.hdf5'
            hdr = gethdr()

            r = requests.get(url,headers=hdr,params = {'stars':'all','gas':'all', 'dm':'all', })
            f = BytesIO(r.content)
            h = h5py.File(f,mode='r')
            with open(cachepath, 'wb') as ff:
                ff.write(r.content)
        else:
            h = h5py.File(cachepath,mode='r')
            self._cachefile = cachepath

        self.header = dict(h['Header'].attrs.items())
        self._particleData = h
        # ckpc/h
        self.CoM = np.array([sub['cm_x'],sub['cm_y'], sub['cm_z']])
        self.CoMdm = np.mean(self.dmPrtl['Coordinates'], axis=0)
        # (kpc/h)/(km/s)
        self.spin = np.array([sub['spin_x'], sub['spin_y'], sub['spin_z']])
        # kpc
        self.hmr = sub['halfmassrad']
        self.hmr_stars = sub['halfmassrad_stars']
        self.hmr_gas = sub['halfmassrad_gas']
        # peculiar velocity of the group, km/s
        self.vel = np.array([sub['vel_x'], sub['vel_y'], sub['vel_z']])
        self.veldm = np.mean(self.dmPrtl['Velocities'], axis=0)
        # 3D peculiar velocity dispersion divided by sqrt{3}
        self.veldisp = sub['veldisp']
        # Maximum value of the spherically-averaged rotation curve, km/s
        # for all particles
        self.vmax = sub['vmax']
        # Comoving radius of rotation curve maximum, ckpc/h
        self.vmaxrad = sub['vmaxrad']
        self._particleTemp = self._calculate_gas_temperature(h)
        #self._starFlux = self._star_particle_flux(h)
        #mags = h['PartType4']['GFM_StellarPhotometrics'][:]
        #starflux = 10**(-mags[:,4]/2.5)
        self._line_flux = self._gas_line_flux(h) # photons rate per particle

        return

    def _generateCube(self, pars, rescale=.25, center=True):
        '''
        pars: cube.CubePars
            A CubePars instance that holds all relevant metadata about the
            desired instrument and DataCube parameters needed to render the
            TNG object
        rescale: float
            TODO: ask Eric
        center: bool
            TODO: ask Eric
        '''

        pixel_scale = pars['pix_scale']
        shape = pars['shape']
        pars['truth'] = {'subhalo':self._subhalo,'simulation':self._snapshot}
        # each element of the following list is an EmissionLine object
        lines = pars['emission_lines'] # may be an empty list

        # list of tuples (lambda_blue, lambda_red) w/ associated astropy unit for
        # each slice
        lambda_bounds = pars['wavelengths']

        # get list of slice wavelength midpoints
        lambdas = np.array([np.mean([l[0], l[1]]) for l in lambda_bounds])
        if 'psf' in pars:
            psf = pars['psf']
        else:
            psf = None

        print('Choosing  indices')
        inds = np.arange(self._particleData['PartType0']['Coordinates'][:,0].size)[
            (self._line_flux.value > 1e-5) & (np.isfinite(self._line_flux.value))
            ]
        #inds = np.arange(self._particleData['PartType0']['Coordinates'][:,0].size)[(self._line_flux.value > 1e3) & (np.isfinite(self._line_flux.value))]

        # What is the position of the sources relative to the field center?
        print('Reading particle data.')
        dx = rescale * (self._particleData['PartType0']['Coordinates'][:,0] -\
                        np.mean(self._particleData['PartType0']['Coordinates'][:,0]))/self.cosmo.h
        dy = rescale * (self._particleData['PartType0']['Coordinates'][:,1] -\
                        np.mean(self._particleData['PartType0']['Coordinates'][:,1]))/self.cosmo.h
        dz = rescale * (self._particleData['PartType0']['Coordinates'][:,2] -\
                        np.mean(self._particleData['PartType0']['Coordinates'][:,2]))/self.cosmo.h

        print('Subsampling particle data.')
        dx = dx[inds]
        dy = dy[inds]
        dz = dz[inds]

        # TODO: add an optional rotation matrix operation.
        RR = Rotation.from_euler('z',45,degrees=True)

        print(f'Calculating velocity offsets')
        # Calculate the velocity offset of each particle.
        deltav = self._particleData['PartType0']['Velocities'][:,2] * np.sqrt(1./(1+self.redshift)) * u.km/u.s
        deltav = deltav[inds]
        # get physical velocities from TNG by multiplying by sqrt(a)
        # https://www.tng-project.org/data/docs/specifications/#parttype0

        # Loop over the lines.
        line_spectra = np.zeros_like(lambdas) *  (self._line_flux[0]* pars['emission_lines'][0].sed(lambdas[0])).unit

        #for iline in pars['emission_lines']:
        #    line_center = iline.line_pars['value'] * (1 + pars['z'])
        #    dlam = line_center *  (dv / const.c).to(u.dimensionless_unscaled)
        #    line_spectra = self._line_flux[:,np.newaxis]* iline.sed( lambdas / (1+pars['z']) - dlam[:,np.newaxis] )

        # Now put these on the pixel grid.
        print('Calculating position offsets')
        du = (dx*u.kpc / self.cosmo.angular_diameter_distance(
            pars['emission_lines'][0].line_pars['z'])
              ).to(u.dimensionless_unscaled).value * 180/np.pi * 3600 / pixel_scale
        dv = (dy*u.kpc / self.cosmo.angular_diameter_distance(
            pars['emission_lines'][0].line_pars['z'])
              ).to(u.dimensionless_unscaled).value * 180/np.pi * 3600 / pixel_scale

        # TODO: This is where we should apply a shear.

        # Round each one to the pixel center
        if center:
            print('Centering')
            hist2d,xbins,ybins = np.histogram2d(du,dv,bins=[np.arange(-100,100),np.arange(-100,100)])
            xmax = xbins[np.argmax(np.sum(hist2d,axis=1))]
            ymax = ybins[np.argmax(np.sum(hist2d,axis=0))]
        else:
            xmax = 0.
            ymax = 0.

        print('Discretizing positions')
        du_int = (np.round(du - xmax)).astype(int)  + int(shape[1]/2)
        dv_int = (np.round(dv - ymax)).astype(int)  + int(shape[2]/2)

        simcube = np.zeros(shape)
        print('Populating datacube')
        pbar = tqdm(total=shape[1]*shape[2])
        for i in range(shape[1]):
            for j in range(shape[2]):
                these = (du_int == i) & (dv_int == j)

                
                for iline in pars['emission_lines']:
                    line_center = iline.line_pars['value'] * (1 + iline.line_pars['z'])
                    #if (line_center > np.min(lambdas/(1+iline.line_pars['z']))) & (line_center < np.max(lambdas/(1+iline.line_pars['z']))):
                    if (line_center > lambdas[0]) & (line_center < lambdas[-1]):
                        dlam = line_center *  (deltav[these] / const.c).to(
                            u.dimensionless_unscaled
                            ).value
                        #line_spectra = self._line_flux[inds[these],np.newaxis]* iline.sed( lambdas / (1+iline.line_pars['z']) - dlam[:,np.newaxis] )
                        line_spectra = self._line_flux[inds[these],np.newaxis] *\
                            iline.sed(lambdas - dlam[:,np.newaxis])

                pbar.update(1)
                simcube[:,i,j] = simcube[:,i,j] + np.sum(line_spectra.value, axis=0)

        pbar.close()

        if psf is not None:
            for islice in range(shape[0]):
                channelIm = galsim.Image(simcube[islice,:,:],scale=pixel_scale)
                channelIm_conv = galsim.Convolve(
                    [psf,galsim.InterpolatedImage(channelIm)]).drawImage(
                        image= channelIm, method='nopixel'
                        )
                simcube[:,i,j] = channelIm_conv.array

        return simcube

    def generateVelocityMap(self, pars, rescale=.25):
        '''
        pars: grism.GrismPars
            A GrismPars instance that holds all relevant metadata about the
            desired instrument and GrismModelCube parameters needed to 
            render the TNG object
        rescale: float
            TODO: ask Eric
        center: bool
            TODO: ask Eric
        '''
        pixel_scale = pars['pix_scale']
        shape = pars['shape']
        pars['truth'] = {'subhalo':self._subhalo,'simulation':self._snapshot}
        x_range = [-shape[1]/2., shape[1]/2.]
        y_range = [-shape[2]/2., shape[2]/2.]

        # choose gas particles with flux above some threshold
        print('Selecting gas particles')
        _crt = (self._line_flux.value > 1e-5) & \
                (np.isfinite(self._line_flux.value))
        inds = np.arange(self.gasPrtl['Coordinates'][:,0].size)[_crt]

        # get the field center, weighted by line flux
        # we'll be lazy and do not mask NaN here, see if we'll meet any
        gas_cen = np.average(self.gasPrtl['Coordinates'], axis=0, 
                            weights=self._line_flux)
        dpos = rescale * (self.gasPrtl['Coordinates'] - gas_cen)/self.cosmo.h

        print('Subsampling particle data.')
        dx = dpos[inds, 0]
        dy = dpos[inds, 1]
        print('Calculating position offsets')
        # angular position
        du = (dx*u.kpc/self.DA*u.rad).to('arcsec').value/pixel_scale
        dv = (dy*u.kpc/self.DA*u.rad).to('arcsec').value/pixel_scale
        print('Discretizing positions')
        du_int = (np.round(du)).astype(int)  + int(shape[1]/2)
        dv_int = (np.round(dv)).astype(int)  + int(shape[2]/2)

        # TODO: add an optional rotation matrix operation.
        RR = Rotation.from_euler('z',45,degrees=True)

        print(f'Calculating velocity offsets')
        # Calculate the velocity offset of each particle.
        deltav = self.gasPrtl['Velocities'][:,2][inds] * \
                    np.sqrt(1./(1+self.redshift))
        v_sys = np.average(deltav, axis=0, weights=self._line_flux[inds])
        print(f'Systemic velocity = {v_sys} km/s')
        # get physical velocities from TNG by multiplying by sqrt(a)
        # https://www.tng-project.org/data/docs/specifications/#parttype0

        # TODO: This is where we should apply a shear.
        v_ary, x_edge, y_edge, binID = binned_statistic_2d(du, dv,
            deltav - v_sys, statistic=np.nanmedian, bins=shape[1:], 
            range=[x_range, y_range]
            )

        return v_ary
    def _generateGrismCube_gas(self, pars, rescale=.25, method=2):
        ''' Generate grism cube for gas particles
        '''
        # setup grids
        pixel_scale = pars['pix_scale']
        shape = pars['shape']
        pars['truth'] = {'subhalo':self._subhalo,'simulation':self._snapshot}

        # get list of slice wavelength midpoints (observer-frame)
        lambda_bounds = pars['wavelengths'] * pars._lambda_unit
        lambdas = np.mean(lambda_bounds, axis=1)
        
        # get observer-frame sed
        _obs_sed = self.gasObsSED(pars, None, method=method)

        # choose gas particles with flux above some threshold
        print('Selecting gas particles')
        _crt = (self._line_flux.value > 1e-5) & \
                (np.isfinite(self._line_flux.value))
        inds = np.arange(self.gasPrtl['Coordinates'][:,0].size)[_crt]

        # get the field center
        dpos = rescale * (self.gasPrtl['Coordinates'] - self.CoMdm)/self.cosmo.h

        print('Subsampling particle data.')
        dx = dpos[inds, 0]
        dy = dpos[inds, 1]
        dz = dpos[inds, 2]
        total_flux = np.sum(self._line_flux[inds])
        flux_frac = self._line_flux[inds]/np.sum(self._line_flux[inds])
        print('Calculating position offsets')
        # angular position
        du = (dx*u.kpc/self.DA*u.rad).to('arcsec').value/pixel_scale
        dv = (dy*u.kpc/self.DA*u.rad).to('arcsec').value/pixel_scale
        print('Discretizing positions')
        du_int = (np.round(du)).astype(int)  + int(shape[1]/2)
        dv_int = (np.round(dv)).astype(int)  + int(shape[2]/2)

        # TODO: add an optional rotation matrix operation.
        RR = Rotation.from_euler('z',45,degrees=True)

        print(f'Calculating velocity offsets')
        # Calculate the velocity offset of each particle.
        deltav = self.gasPrtl['Velocities'][:,2][inds] * \
                    np.sqrt(1./(1+self.redshift)) * u.km/u.s
        # get physical velocities from TNG by multiplying by sqrt(a)
        # https://www.tng-project.org/data/docs/specifications/#parttype0

        # TODO: This is where we should apply a shear.
        
        simcube = np.zeros(shape)
        print('Populating datacube with emission lines flux')
        pbar = tqdm(total=shape[1]*shape[2])
        for i in range(shape[1]):
            for j in range(shape[2]):
                these = (du_int == i) & (dv_int == j)
                if any(these):
                    # get the LoS-shifted SED
                    Doppler = 1.0 - (deltav[these]/const.c).to('1').value
                    #ipdb.set_trace()
                    DopLam = lambdas*Doppler[:,np.newaxis] # [these, Nlam]
                    if method==1:
                        line_spectra = flux_frac[these, np.newaxis] * _obs_sed(DopLam)
                    elif method==2:
                        line_spectra = flux_frac[these, np.newaxis] * _obs_sed(DopLam) * total_flux
                    simcube[:,i,j] = simcube[:,i,j] + np.sum(line_spectra.value, axis=0)
                pbar.update(1)
        return simcube
    
    def _generateGrismCube_star(self, pars, rescale=.25):
        ''' Generate grism cube for star particles
        '''
        # setup grids
        pixel_scale = pars['pix_scale']
        shape = pars['shape']
        pars['truth'] = {'subhalo':self._subhalo,'simulation':self._snapshot}

        # get list of slice wavelength midpoints (observer-frame)
        lambda_bounds = pars['wavelengths'] * pars._lambda_unit
        lambdas = np.mean(lambda_bounds, axis=1)
        
        # get observer-frame sed
        # stellar continuum 
        _obs_sed = self.starObsSED(pars, None)

        # choose gas particles with flux above some threshold
        print('Selecting star particles')
        inds = self._starIDs

        # get the field center
        dpos = rescale * (self.starPrtl['Coordinates'] - self.CoMdm)/self.cosmo.h
        SMs = self.starPrtl['Masses'][self._starIDs]*1e10/self.cosmo.h # Msol
        print('Subsampling particle data.')
        dx = dpos[inds, 0]
        dy = dpos[inds, 1]
        dz = dpos[inds, 2]
        print('Calculating position offsets')
        # angular position
        du = (dx*u.kpc/self.DA*u.rad).to('arcsec').value/pixel_scale
        dv = (dy*u.kpc/self.DA*u.rad).to('arcsec').value/pixel_scale
        print('Discretizing positions')
        du_int = (np.round(du)).astype(int)  + int(shape[1]/2)
        dv_int = (np.round(dv)).astype(int)  + int(shape[2]/2)

        # TODO: add an optional rotation matrix operation.
        RR = Rotation.from_euler('z',45,degrees=True)

        print(f'Calculating velocity offsets')
        # Calculate the velocity offset of each particle.
        deltav = self.starPrtl['Velocities'][:,2][inds] * \
                    np.sqrt(1./(1+self.redshift)) * u.km/u.s
        # get physical velocities from TNG by multiplying by sqrt(a)
        # https://www.tng-project.org/data/docs/specifications/#parttype0

        # TODO: This is where we should apply a shear.
        
        simcube = np.zeros(shape)
        print('Populating datacube with stellar continuum flux')
        pbar = tqdm(total=shape[1]*shape[2])
        for i in range(shape[1]):
            for j in range(shape[2]):
                these = (du_int == i) & (dv_int == j)
                if any(these):
                    # get the LoS-shifted SED
                    Doppler = 1.0 - (deltav[these]/const.c).to('1').value
                    #ipdb.set_trace()
                    DopLam = lambdas*Doppler[:,np.newaxis] # [these, Nlam]
                    continuum_spectra = SMs[these, np.newaxis] * _obs_sed(DopLam)
                    simcube[:,i,j] = simcube[:,i,j] + np.sum(continuum_spectra, axis=0)
                pbar.update(1)
        return simcube
        
    def _generateGrismCube(self, pars, rescale=.25, method=2, gas_weight=1, cached=False):
        '''
        pars: grism.GrismPars
            A GrismPars instance that holds all relevant metadata about the
            desired instrument and GrismModelCube parameters needed to 
            render the TNG object
        rescale: float
            TODO: ask Eric
        center: bool
            TODO: ask Eric
        '''
        _cached_fname_pars = '.cache/pars.pkl'
        _cached_fname_gascube = '.cache/gascube.pkl'
        _cached_fname_starcube = '.cache/starcube.pkl'
        _cache_exists = os.path.exists(_cached_fname_pars) and \
                       os.path.exists(_cached_fname_gascube) and \
                       os.path.exists(_cached_fname_starcube)
        if cached:
            if _cache_exists:
                with open(_cached_fname_pars, 'rb') as fp:
                    _cached_pars = pickle.load(fp)
                with open(_cached_fname_gascube, 'rb') as fp:
                    _cached_gascube = pickle.load(fp)
                with open(_cached_fname_starcube, 'rb') as fp:
                    _cached_starcube = pickle.load(fp)
                self.cube_gas = _cached_gascube
                self.cube_star = _cached_starcube
                #if _cached_pars == pars:
                #    self.cube_gas = _cached_gascube
                #    self.cube_star = _cached_starcube
            else:
                self.cube_gas = self._generateGrismCube_gas(pars, rescale, method=method)
                self.cube_star = self._generateGrismCube_star(pars, rescale)
            with open(_cached_fname_pars, 'wb') as fp:
                pickle.dump(pars, fp)
            with open(_cached_fname_gascube, 'wb') as fp:
                pickle.dump(self.cube_gas, fp)
            with open(_cached_fname_starcube, 'wb') as fp:
                pickle.dump(self.cube_star, fp)
        else:
            self.cube_gas = self._generateGrismCube_gas(pars, rescale, method=method)
            self.cube_star = self._generateGrismCube_star(pars, rescale)

        return self.cube_gas*gas_weight + self.cube_star
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

    def from_slit(self):
        '''
        To Do:
        Method for generating slit spectrum from mock data
        '''
        pass

    def to_slit(self, pars):
        '''
        Generates slit spectrum given meta data

        pars: cube.CubePars
            A CubePars instance that holds all relevant metadata about the
            desired instrument and DataCube parameters needed to render the
            TNG object

        To Do:
        Implement an abstract class for the slit spectrum and return instance instead of slit spectrum
        '''
        data_cube = self.to_cube(pars)
        simcube = data_cube._data
        slit_mask = self._get_slit_mask(pars)

        slit_spectrum = np.sum(slit_mask[np.newaxis, :, :]*simcube, axis=2)

        return slit_spectrum

    def to_grism(self, pars, rescale=0.25, method=2, gas_weight=1, cached=False):
        ''' 
        Generate grism spectrum given meta data

        pars: any classes that subclass from cube.CubePars
            A CubePars or its derived class that holds all relevant metadata
            about the desired instrument and DataCube parameters needed to
            render the TNG object.
            The this particular function, it is grism.GrismPars object. 
        '''
        # build data cube
        print('Generating 3d data cube')
        _simcube = self._generateGrismCube(pars, rescale = rescale, method=method, gas_weight=gas_weight, cached=cached)
        _simcube = np.transpose(_simcube, (0, 2, 1)) # follow the convention of Spencer, Nlam, Nx, Ny
        self.simGrismCube = grism.GrismModelCube(_simcube, pars=pars)
        print('Generating simulated grism image')
        image, noise = self.simGrismCube.observe(force_noise_free=False)
        # return the grism data
        return image, noise

    def _get_slit_mask(self, pars):
        '''
        Creates slit mask given a list of slit parameters

        pars: cube.CubePars
            A CubePars instance that holds all relevant slit metadata
        '''

        ###
        # slit_width: float
        #     Slit width (in arcsec)

        # slit_angle: float
        #     Slit angle w.r.t. to the x-axis of the observed/image plane

        # shape: (ngrid_x, ngrid_y) tuple
        #     Number of grid points in the pixelized mask

        # pix_scale: float
        #     Pixel scale (in arcsec/pix)

        # offset_x: float
        #     x-offset of the slit mask from grid center (in arcsec)

        # offset_y: float
        #     y-offset of the slit mask from grid center (in arcsec)

        slit_width, slit_angle
        shape = (pars['shape'][1], pars['shape'][2])
        pix_scale = pars['pixscale']
        offset_x, offset_y = pars['offset_x'], pars['offset_y']

        slit_mask = np.ones((ngrid, ngrid))
        grid_x = self.generate_grid(0, pix_scale, shape[0])
        grid_y = self.generate_grid(0, pix_scale, shape[1])

        xx, yy = np.meshgrid(grid_x, grid_y)

        xx_new = (xx - offset_x) * np.cos(slit_angle) - (yy - offset_y) * np.sin(slit_angle)
        yy_new = (xx - offset_x) * np.sin(slit_angle) + (yy - offset_y) * np.cos(slit_angle)

        slit_mask[np.abs(yy_new) > slit_width/2.] = 0.

        return slit_mask

    def _generate_grid(self, center, pix_scale, ngrid):
        low, high =  center - pix_scale*ngrid, center + pix_scale*ngrid
        edges = np.linspace(low, high, ngrid+1)

        centers = (edges[1:] + edges[:-1])/2

        return centers

    def getzdisk(self, rmax_pkpc=30):
        a = 1.0/(1.0+self.redshift)
        h0 = self.cosmo.h
        mgas = self.gasPrtl['Masses'][:]*1e10*const.M_sun/h0 # 1e10 Msol/h
        rgas = (self.gasPrtl['Coordinates']-self.CoMdm)*a/h0*u.kpc # kpc
        vgas = (self.gasPrtl['Velocities']-self.veldm)*a**0.5*u.Unit('km s-1')
        Rgas = np.sum(rgas*rgas, axis=1)**0.5
        gasIDs = np.where(np.logical_and(self._line_flux.value>1e-5, 
            Rgas<rmax_pkpc*u.kpc))[0]
        ### Angular momentum of gas particles
        Lgas = (np.cross(rgas, vgas).T * mgas).T
        Ldisk = np.sum(Lgas[gasIDs], axis=0)
        # definition of z-direction: direction of the total angular momentum
        return Ldisk/np.sqrt(Ldisk[0]**2 + Ldisk[1]**2 + Ldisk[2]**2)

    def getRotationCurve(self, rmin_pkpc=0, rmax_pkpc=30, Nbins=20):
        ''' Generate the gas rotation velocity curve given radial binning
        '''
        a = 1/(1+self.redshift)
        h0 = self.cosmo.h
        R_bins = np.linspace(rmin_pkpc, rmax_pkpc, Nbins+1)
        mgas = self.gasPrtl['Masses'][:]*1e10*const.M_sun/h0 # 1e10 Msol/h
        rgas = (self.gasPrtl['Coordinates']-self.CoMdm)*a/h0*u.kpc # kpc
        vgas = (self.gasPrtl['Velocities']-self.veldm)*a**0.5*u.Unit('km s-1')
        Rgas = np.sum(rgas*rgas, axis=1)**0.5
        gasIDs = np.where(np.logical_and(self._line_flux.value>1e-5, 
            Rgas<rmax_pkpc*u.kpc))[0]
        ### Angular momentum of gas particles
        Lgas = (np.cross(rgas, vgas).T * mgas).T
        Ldisk = np.sum(Lgas[gasIDs], axis=0)
        # definition of z-direction: direction of the total angular momentum
        zdisk = Ldisk/np.sqrt(np.sum(Ldisk*Ldisk))
        Ldisk = np.sqrt(np.sum(Ldisk*Ldisk))
        dgas = np.cross(rgas, zdisk)
        Dgas = np.sqrt(np.sum(dgas.T*dgas.T, axis=0))
        Lzgas = np.sum(Lgas * zdisk, axis=1)
        SFR = self.gasPrtl['StarFormationRate'][:]
        vtan = Lzgas/(mgas*Dgas)
        gas2Dhist, skip, gas2Dbin_id = binned_statistic(
            Dgas.value, Dgas.value, statistic='count', bins=R_bins)
        gas2Dmap = defaultdict(list)
        for i,bini in enumerate(gas2Dbin_id):
            gas2Dmap[bini].append(i)
        vcirc_mean = np.zeros(Nbins)
        vcirc_std = np.zeros(Nbins)
        for i in range(Nbins):
            pid = gas2Dmap[i+1]
            _vcirc = np.average(vtan[pid], weights=SFR[pid])
            _vstd = np.average((vtan[pid]-_vcirc)**2, weights=SFR[pid])**0.5
            vcirc_mean[i] = _vcirc.to('km s-1').value
            vcirc_std[i] = _vstd.to('km s-1').value
        self.vrot = vcirc_mean
        self.vrot_err = vcirc_std
        return (R_bins[1:]+R_bins[:-1])/2., self.vrot, self.vrot_err

    def getCircularVelocity(self, rmin_pkpc=0, rmax_pkpc=30, Nbins=20):
        ''' Generate the gas circular velocity curve given the radial binning
        '''
        a = 1/(1+self.redshift)
        h0 = self.cosmo.h
        R_bins = np.linspace(rmin_pkpc, rmax_pkpc, Nbins+1)
        R_cen = (R_bins[1:]+R_bins[:-1])/2.

        mgas = self.gasPrtl['Masses'][:]*1e10*const.M_sun/h0 # 1e10 Msol/h
        rgas = (self.gasPrtl['Coordinates']-self.CoMdm)*a/h0*u.kpc # kpc
        Rgas = np.sum(rgas*rgas, axis=1)**0.5

        mstar = self.starPrtl['Masses'][:]*1e10*const.M_sun/h0 # 1e10 Msol/h
        rstar = (self.starPrtl['Coordinates']-self.CoMdm)*a/h0*u.kpc # kpc
        Rstar = np.sum(rstar*rstar, axis=1)**0.5

        mdm = self.header['MassTable'][1] * 1e10*const.M_sun/h0
        rdm = (self.dmPrtl['Coordinates']-self.CoMdm)*a/h0*u.kpc # kpc
        Rdm = np.sum(rdm*rdm, axis=1)**0.5

        dm3Dhist, skip, dm3Dbin_id = binned_statistic(
            Rdm.to('kpc').value, Rdm.value, statistic='count', bins=R_bins)
        dm3Dhist = dm3Dhist*mdm
        gas3Dhist, skip, gas3Dbin_id = binned_statistic(
            Rgas.to('kpc').value, mgas.to('kg').value, 
            statistic='sum', bins=R_bins)
        gas3Dhist = gas3Dhist*u.kg
        star3Dhist, skip, star3Dbin_id = binned_statistic(
            Rstar.to('kpc').value, mstar.to('kg').value, 
            statistic='sum', bins=R_bins)
        star3Dhist = star3Dhist*u.kg
        BH3Dhist = np.zeros(Nbins)
        BH3Dhist[0] = self._particleData['PartType5']['Masses'][0]
        BH3Dhist = BH3Dhist * 1e10*const.M_sun/h0
        Menclose = dm3Dhist + gas3Dhist + star3Dhist + BH3Dhist
        for i in range(1, Nbins):
            Menclose[i] += Menclose[i-1]
        self.vcirc = np.sqrt(Menclose*const.G/(R_cen*u.kpc)).to('km s-1').value
        return (R_bins[1:]+R_bins[:-1])/2., self.vcirc

    def getFaceOnDirection(self):
        a = 1.0/(1.0+self.redshift)
        h0 = self.cosmo.h
        SFR = self.gasPrtl['StarFormationRate'][:]
        rgas = (self.gasPrtl['Coordinates']-self.CoMdm)*a/h0*u.kpc # kpc
        Rgas = np.sum(rgas*rgas, axis=1)**0.5
        _rmax = 2*self.hmr_stars/(1+self.redshift)/self.cosmo.h*u.kpc
        inds = np.where(Rgas < _rmax)[0]
        if len(inds)>=50:
            Iij = np.sum(
                [np.outer(r, r) for r,sfr in zip(rgas[inds], SFR[inds])], 
                axis=0)
        else:
            mstar = self.starPrtl['Masses'][:]*1e10*const.M_sun/h0
            rstar = (self.starPrtl['Coordinates']-self.CoMdm)*a/h0*u.kpc
            Rstar = np.sum(rstar*rstar, axis=1)**0.5
            inds = np.where(Rstar < _rmax/2.)[0]
            Iij = np.sum(
                [np.outer(r, r) for r,m in zip(rstar[inds], mstar[inds])], 
                axis=0)
        w,v = np.linalg.eig(Iij)
        print(f'Eigenvals of SFR moment of inertia {w}')
        zFaceOn = v.T[0]
        return zFaceOn

    def getStarMomentInertiaEigvals(self):
        a = 1.0/(1.0+self.redshift)
        h0 = self.cosmo.h
        mstar = self.starPrtl['Masses'][:]
        rstar = (self.starPrtl['Coordinates']-self.CoMdm)*a/h0
        Iij = np.sum(
            [np.outer(r, r)*m for r,m in \
            zip(rstar[self._starIDs], mstar[self._starIDs])], axis=0)
        Iij /= np.sum(mstar[self._starIDs])
        w,v = np.linalg.eig(Iij)
        print(f'Eigenvals of Stellar mass moment of inertia {w}')
        return w

    def getGasMomentInertiaEigvals(self):
        a = 1.0/(1.0+self.redshift)
        h0 = self.cosmo.h
        inds = np.where( (self._line_flux.value > 1e-5) & \
                (np.isfinite(self._line_flux.value)) )[0]
        rgas = (self.gasPrtl['Coordinates']-self.CoMdm)*a/h0
        sfrgas = self.gasPrtl['StarFormationRate'][:]
        Iij = np.sum(
            [np.outer(r, r)*s for r,s in \
            zip(rgas[inds], sfrgas[inds])], axis=0)
        Iij /= np.sum(sfrgas[inds])
        w,v = np.linalg.eig(Iij)
        print(f'Eigenvals of gas SFR moment of inertia {w}')
        return w

    def getStarHMR(self, rescale = 1.0, arcsec=True):
        shmr = self.hmr_stars/(1+self.redshift)/self.cosmo.h*u.kpc*rescale
        if arcsec:
            return ((shmr/self.DA)*u.rad).to('arcsec').value
        else:
            return shmr.to('kpc').value

    def getGasHMR(self, rescale = 1.0, arcsec=True):
        ghmr = self.hmr_gas/(1+self.redshift)/self.cosmo.h*u.kpc*rescale
        if arcsec:
            return ((ghmr/self.DA)*u.rad).to('arcsec').value
        else:
            return ghmr.to('kpc').value

    def getHMR(self, rescale=1.0, arcsec=True):
        hmr = self.hmr/(1+self.redshift)/self.cosmo.h*u.kpc*rescale
        if arcsec:
            return ((hmr/self.DA)*u.rad).to('arcsec').value
        else:
            return hmr.to('kpc').value

def main():

    sim = TNGsimulation()
    sim.set_subhalo(2)

    # Make a mock of a MUSE data cube.
    testdir = utils.get_test_dir()
    testpath = pathlib.Path(os.path.join(testdir, 'test_data'))
    spec1dPath = testpath / pathlib.Path("spectrum_102021103.fits")
    spec3dPath = testpath / pathlib.Path("102021103_objcube.fits")
    catPath = testpath / pathlib.Path("MW_1-24_main_table.fits")
    emlinePath = testpath / pathlib.Path("MW_1-24_emline_table.fits")

    # Try initializing a datacube object with these paths.
    museCube = muse.MuseDataCube(
        cubefile=spec3dPath,
        specfile=spec1dPath,
        catfile=catPath,
        linefile=emlinePath
        )
    # Now generate a MUSE-like mock from the datacube.
    musemock = sim.from_cube(museCube)

    lambdas = np.array([np.mean([l[0], l[1]]) for l in museCube.pars['wavelengths']])
    spec1d = np.sum(np.sum(musemock.data,axis=-1),axis=-1)
    lam_vel = np.sum(lambdas[:,np.newaxis,np.newaxis] * musemock.data,axis=0) / np.sum(musemock.data,axis=0)
    lam_cen = np.nanmedian(lam_vel)
    vel = (lam_vel/lam_cen-1.) * 300000.

    fig,(ax1,ax2) = plt.subplots(ncols=2,figsize=(14,7))
    ax1.imshow(np.log10(np.sum(musemock.data,axis=0)),origin='lower')
    cax1 = ax1.set_title("log summed intensity")
    cax = ax2.imshow(vel,cmap=plt.cm.seismic,vmin=-1000,vmax=1000,origin='lower')
    ax2.set_title("line-of-sight velocity")
    fig.colorbar(cax,ax=ax2,fraction=0.046,pad=0.04)
    plt.show()

    return

if __name__ == '__main__':
    main()
