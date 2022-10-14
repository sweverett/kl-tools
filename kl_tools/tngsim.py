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
import muse

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

    def set_subhalo(self, subhaloid, redshift=0.5, simname = 'TNG50-1'):
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
        cachepath = pathlib.Path(
            f'{utils.CACHE_DIR}/{self._sim_name}_subhalo_{sub["id"]}_{self._snapshot["number"]}.hdf5'
            )
        if not cachepath.exists():
            url = f'http://www.tng-project.org/api/{self._sim_name}/snapshots/{sub["snap"]}/subhalos/{sub["id"]}/cutout.hdf5'
            hdr = gethdr()

            r = requests.get(url,headers=hdr,params = {'stars':'all','gas':'all'})
            f = BytesIO(r.content)
            h = h5py.File(f,mode='r')
            with open(cachepath, 'wb') as ff:
                ff.write(r.content)
        else:
            h = h5py.File(cachepath,mode='r')
            self._cachefile = cachepath

        self._particleData = h
        self._particleTemp = self._calculate_gas_temperature(h)
        self._starFlux = self._star_particle_flux(h)
        #mags = h['PartType4']['GFM_StellarPhotometrics'][:]
        #starflux = 10**(-mags[:,4]/2.5)
        self._line_flux = self._gas_line_flux(h)

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
