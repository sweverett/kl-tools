import os
import warnings
import copy
import ipdb
import numpy as np
import galsim as gs
# astropy
import astropy.units as u
import astropy.constants as constants
# interpolations
from scipy.interpolate import interp1d
from numpy import interp
# kltools
import utils

### some constants
_h = constants.h.to('erg s').value
_c = constants.c.to('nm/s').value
_c_kms = constants.c.to('km/s').value
_c_nms = constants.c.to('nm/s').value

### Emission line table
# NOTE: These are the emission line wavelengths
# in a vacuum
# NOTE: These use the MUSE convention, but we can map to these
# for other experiments
LINE_LAMBDAS = {
    'O6_2': (1031.93, 1037.62) * u.Unit('Angstrom'),
    'Lya': 1215.670 * u.Unit('Angstrom'),
    'N5': (1238.821, 1242.804) * u.Unit('Angstrom'),
    'C4': (1548.203, 1550.777) * u.Unit('Angstrom'),
    'Mg2': 2796.290 * u.Unit('Angstrom'),
    'O2': (3727.048, 3729.832) * u.Unit('Angstrom'),
    'Ne3': 3870.115 * u.Unit('Angstrom'),
    'Ne32': 3968.872 * u.Unit('Angstrom'),
    'Hzet': 3890.109 * u.Unit('Angstrom'),
    'Heps': 3971.154 * u.Unit('Angstrom'),
    'Hd': 4102.852 * u.Unit('Angstrom'),
    'Hg': 4341.647 * u.Unit('Angstrom'),
    'O3_3': 4364.400 * u.Unit('Angstrom'),
    'Hb': 4862.650 * u.Unit('Angstrom'),
    'O3_1': 4960.263 * u.Unit('Angstrom'),
    'O3_2': 5008.208 * u.Unit('Angstrom'),
    'He1': 5877.217 * u.Unit('Angstrom'),
    'O1': 6302.022 * u.Unit('Angstrom'),
    'N2_1': 6549.825 * u.Unit('Angstrom'),
    'Ha': 6564.589 * u.Unit('Angstrom'),
    'N2_2': 6585.255 * u.Unit('Angstrom'),
    'S2_1': 6718.271 * u.Unit('Angstrom'),
    'S2_2': 6732.645 * u.Unit('Angstrom'),
}

### default sed parameters, as an example
_default_sed_pars = {
    # redshift
    'z': 0.0,
    ### **obs-frame** SED wavelength grid limits, nm
    'lblue': 50,
    'lred': 2000,
    # spectral resolution at 1 micron, assuming dispersion per pixel
    'resolution': 100000,
    'thin': -1,
    ### **rest-frame** continuum
    'continuum_type': 'func', # func, template, spec?
    'continuum_func': '10 - (wave-400)/400',
    'restframe_temp': '../data/Simulation/GSB2.spec',
    'temp_wave_type': 'Ang',
    'temp_flux_type': 'flambda',
    ### **obs-frame** continuum normalization, either
    'cont_norm_method': 'flux', # flux or mag
    # 1. normalized to specific flam (erg/s/cm2/nm) at specific wavelength (nm)
    'obs_cont_norm_wave': 400,
    'obs_cont_norm_flam': 0.0,
    # 2. normalized to specific magnitude at specific band 
    'obs_norm_band': "../data/Bandpass/HST/WFC3_IR_F105W.dat",
    'obs_norm_mag': 17,
    ### **obs-frame** emission line properties
    # flux (erg/s/cm2), line width (nm), doublet flux share, systemic velocity (km/s)
    # other emission line properties could include: CONT, EW, etc. See
    # https://desidatamodel.readthedocs.io/en/latest/DESI_SPECTRO_REDUX/SPECPROD/tiles/GROUPTYPE/
    #     TILEID/GROUPID/emline-SPECTROGRAPH-TILEID-GROUPID.html
    # for reference.
    'em_O2_flux': None,
    'em_O2_sigma': (0.1, 0.1),
    'em_O2_share': (0.3,0.7),
    'em_O2_vsys': 30,
    'em_Hb_flux': None,
    'em_Hb_sigma': 0.4,
    'em_O3_1_flux': None,
    'em_O3_1_sigma': 0.5,
    'em_O3_2_flux': None,
    'em_O3_2_sigma': 0.5,
    'em_Ha_flux': None,
    'em_Ha_sigma': 0.5,
}


class EmissionLine(object):
    '''
    Holds relevant information for a specific emission line
    '''

    _req_line_pars = ['value', 'R', 'z', 'unit']
    _req_sed_pars = ['lblue', 'lred', 'resolution', 'unit']

    def __init__(self, line_pars, sed_pars):
        '''
        line_pars: dict
            A dictionary that holds all relevant line params
        sed_pars: dict
            A dictionary that holds all relevant sed params

        ------------------------------------------------
        line_pars fields:

        value: float
            Line central value (restframe, vacuum)
        R: float
            Spectral resolution of instrument
        z: float
            Redshift of line
        unit: astropy.Unit
            Unit of line wavelength

        ------------------------------------------------
        sed_pars fields:

        lblue: float
            Starting wavelength in `unit`
        lred: float
            Ending wavelength in `unit`
        resolution: float
            Spectral resolution (assumed to be constant)
        unit: astropy.Unit
            The unit of the SED wavelength
        '''

        args = {
            'line_pars': (line_pars, dict),
            'sed_pars': (sed_pars, dict)
            }

        utils.check_types(args)

        utils.check_fields(line_pars, self._req_line_pars, None, 'line_pars')
        utils.check_fields(sed_pars, self._req_sed_pars, None, 'sed_pars')

        self.line_pars = line_pars
        self.sed_pars = sed_pars

        self.setup_sed()

        return

    def setup_sed(self):

        self.sed = self._build_sed(self.line_pars, self.sed_pars)

        return

    @staticmethod
    def _build_sed(line_pars, sed_pars):

        lblue = sed_pars['lblue']
        lred = sed_pars['lred']
        res = sed_pars['resolution']
        lam_unit = sed_pars['unit']

        lambdas = np.arange(lblue, lred+res, res) * lam_unit

        # Will compare the requested resolution w/ the slice delta lambda's
        # to make sure a reasonable interpolator step size is used
        wlam = np.mean(
            (np.array(lambdas)[1:] - np.array(lambdas)[:-1])/2.
            )

        # Model emission line SED as gaussian
        R = line_pars['R']
        z = line_pars['z']
        obs_val = line_pars['value'] * (1.+z)
        obs_std = obs_val / R

        line_unit = line_pars['unit']
        mu  = obs_val * line_unit
        std = np.sqrt(obs_std**2 + wlam**2) * line_unit

        norm = 1. / (std * np.sqrt(2.*np.pi))
        chi = ((lambdas - mu)/std).value
        gauss = norm * np.exp(-0.5*chi**2)

        # This was added by Eric to convert to a numpy interpolator, but this
        # causes pickling issues
        #def interpfunc(x):
        #    return np.interp(x,lambdas.to(lam_unit).value,gauss,left=0.,right=0.)
        #return interpfunc

        return interp1d(lambdas, gauss, fill_value=0., bounds_error=False)

class SED(object):
    '''
    Not currently being used, but we could
    '''
    def __init__(self, lblue, lred, resolution, unit):

        args = {
            'lblue': lblue,
            'lred': lred,
            'resolution': resolution
        }
        for name, val in args.items():
            if not isinstance(val, (int,float)):
                raise TypeError(f'{name} must be an int or float!')

        if lblue >= lred:
            raise ValueError('lred must be greater than lblue!')

        return

class ObsFrameSED(SED):
    def __init__(self, sed_meta=None):
        '''
        Initialize SED class object with parameters dictionary
        '''
        self.pars = copy.deepcopy(_default_sed_pars)
        super(ObsFrameSED, self).__init__(self.pars['lblue'], self.pars['lred'], self.pars['resolution'], 'nm')
        self.blue_limit, self.red_limit = self.pars['lblue']*u.nm, self.pars['lred']*u.nm
        if sed_meta is not None:
            self.updatePars(self.pars, sed_meta)
        #print(self.pars)
        self.calculate_obsframe_sed(self.pars)
        self.inventory = {
            'total':     self.obs_frame_sed, 
            'continuum': self.continuum, 
            'emissions': self.emissions,
        }

    def __call__(self, wave, new_pars=None, component='total', flux_type='fphotons'):
        if isinstance(wave, u.Quantity):
            wave = wave.to(self.spectrum.wave_type).value
            
        # check SED recomputation
        if (new_pars is not None) and self.updatePars(self.pars, new_pars):
            self.calculate_obsframe_sed(self.pars)
        
        # check returned flux type
        norm = {'fphotons': 1, '1': 1, 'flambda': (_h*_c)/wave, 'fnu': wave*_h}
        if flux_type not in norm.keys():
            raise ValueError(f'`flux_type` only supports ({norm.keys()})!')
        
        # check returned SED components
        if component not in self.inventory.keys():
            raise ValueError(f'`component` only supports ({self.inventory.keys()})!')
        return self.inventory[component](wave)*norm[flux_type]
            
    
    def calculate_obsframe_sed(self, pars):
        meta_obs_wave = np.arange(self.pars['lblue'], self.pars['lred'], 
            1000/self.pars['resolution'])
        self.continuum, _cont_tab = self.addContinuum(pars, meta_obs_wave)
        self.emissions, _emis_tab = self.addEmissionLines(pars, meta_obs_wave)
        #self.emission = np.sum(self.emissions_list)
        _total_tab = gs.LookupTable(meta_obs_wave, 
            _cont_tab(meta_obs_wave) + _emis_tab(meta_obs_wave), 
            interpolant='linear')
        #self.obs_frame_sed = self.continuum + self.emission
        self.obs_frame_sed = gs.SED(_total_tab, "nm", "flambda", fast=True, interpolant='linear')
        if pars['thin'] > 0:
            self.obs_frame_sed = self.obs_frame_sed.thin(rel_err=pars['thin'])
        
    @classmethod
    def updatePars(cls, old_pars, new_pars):
        '''
        Update parameters, return True is pars is updated.
        '''
        recompute = False
        for key, val in new_pars.items():
            if type(val) is dict:
                recompute = recompute | updatePars(old_pars[key], new_pars[key])
            else:
                if (old_pars[key] != val):
                    recompute = True
                    old_pars[key] = val
        return recompute
    
    @classmethod
    def wrap_emline_string(cls, center, sigma, share):
        eml_fmt = 'np.exp(-0.5*((wave-%le)/%le)**2)*%le/np.sqrt(2*np.pi*%le**2)'
        _cen, _sig, _shr = np.atleast_1d(center), np.atleast_1d(sigma), np.atleast_1d(share)
        assert _cen.shape == _sig.shape == _shr.shape, 'center, sigma, share have inconsistent shape!'
        eml_strings = ' + '.join([eml_fmt%(_c,_s,_f,_s) for _c,_s,_f in zip(_cen, _sig, _shr)])
        return eml_strings
            
    @classmethod
    def addEmissionLines(cls, pars, meta_obs_wave):
        # Gaussian emission line profile (no spectra resolution convolved)
        meta_obsframe_flux = np.zeros(meta_obs_wave.shape)
        # loop through emission lines in the meta parameters
        for emline,vacwave in LINE_LAMBDAS.items():
            if pars.get(f'em_{emline}_flux', None) is not None:
                # get obs-frame line width (nm), line flux (erg/s/cm2), and doublet share
                zsys = pars.get(f'em_{emline}_vsys', 0.0)/_c_kms # redshift due to systemic velocity
                obs_wcen  = vacwave.to('nm').value*(1.+zsys)*(1.+pars['z'])
                obs_sigma = pars[f'em_{emline}_sigma']
                obs_flux  = pars[f'em_{emline}_flux']
                share = pars.get(f'em_{emline}_share', 1.0)
                assert np.isclose(np.sum(share), 1.0), 'emission line share does not add up to 1!'
                # the redshift argument only changes the wavelength, keeping flux density fixed
                emlstr = cls.wrap_emline_string(obs_wcen, obs_sigma, share)
                emlsed = eval('lambda wave:'+emlstr) # erg/s/cm2/nm
                meta_obsframe_flux += emlsed(meta_obs_wave)*obs_flux
                #emline_list.append(gs.SED(emlstr, wave_type='nm', flux_type='flambda', redshift=z)*obs_flux)
        _table = gs.LookupTable(meta_obs_wave, meta_obsframe_flux, 
            interpolant='linear', )
        _sed = gs.SED(_table, "nm", "flambda", fast=True, interpolant='linear')
        return _sed, _table
    
    @classmethod
    def addContinuum(cls, pars, meta_obs_wave):
        # TODO: add support to other continuum methods: desi spectra
        # acceptable continuum types: lambda string, template file
        if pars['continuum_type'] == 'func':
            cont = gs.SED(pars['continuum_func'], wave_type='nm', flux_type='flambda', redshift=pars['z'])
            #cont = eval('lambda wave:'+pars['continuum_func']) # erg/s/cm2/nm
        elif pars['continuum_type'] == 'temp':
            template = pars['restframe_temp']
            if not os.path.isfile(template):
                raise OSError(f'Can not find template file {template}!')
            #_temp = np.genfromtxt(template)
            # convert wavelength and flux to nm and flambda
            #wave_factor = {'nm': 1, 
            #               'ang': 10, 
            #               'angstrom': 10, 
            #               'aa': 10}[pars['temp_wave_type'].lower()]
            #flux_factor = {'flambda': wave_factor,
            #                'fnu': _c_nms/(_temp[:,0]/wave_factor)**2,
            #                'fphotons': wave_factor**2*_h*_c/_temp[:,0],
            #                '1': wave_factor}[pars['temp_flux_type']]
            #cont = gs.LookupTable(
            #    _temp[:,0]/wave_factor*(1.+pars['z']), _temp[:,1]*flux_factor, 
            #    interpolant='linear')
            cont = gs.SED(template, pars['temp_wave_type'], 
                pars['temp_flux_type'], redshift=pars['z'])
        else:
            raise ValueError(f'Continuum type {pars["continuum_type"]} is not yet supported. '+\
                             f'Only (func, temp) are supported now.')
        #meta_obsframe_flux = cont(meta_obs_wave)
        #_table = gs.LookupTable(meta_obs_wave, meta_obsframe_flux, interpolant='linear')
        #_sed = gs.SED(_table, "nm", "flambda", fast=True, interpolant='linear')

        # normalization: either specify flux at specific wavelength (do not require emission line)
        # or specify band magnitude (need to subtract emission line flux)
        if pars.get('cont_norm_method', 'flux') == 'flux':
            # note for the units 
            # `withFluxDensity`:    phot/s/cm2/nm
            # `obs_cont_norm_flam`: erg/s/cm2/nm
            _sed = cont.withFluxDensity(
                pars['obs_cont_norm_flam']/(_h*_c/pars['obs_cont_norm_wave']), 
                pars['obs_cont_norm_wave'])
        elif pars.get('cont_norm_method', 'flux') == 'mag':
            # TODO: compensate for emission line fluxes here
            norm_band = gs.Bandpass(pars['obs_norm_band'], wave_type='nm').withZeropoint('AB')
            _sed = cont.withMagnitude(pars['obs_norm_mag'], norm_band)
        else:
            # do not normalize
            warnings.warn(f'cont_norm_method is set to {pars["cont_norm_method"]} which is neither '+\
                          f'flux nor mag. Skip continuum normalizing...')
        _tab = gs.LookupTable(meta_obs_wave,
                _sed(meta_obs_wave)*_h*_c/meta_obs_wave, interpolant='linear')
        return _sed, _tab
    
    def calculateMagnitude(self, bandpass, zp='AB', component='total'):
        bp = gs.Bandpass(bandpass, wave_type='nm').withZeropoint(zp)
        return self.inventory[component].calculateMagnitude(bp)
    
    def calculateFlux(self, bandpass, zp='AB', component='total'):
        bp = gs.Bandpass(bandpass, wave_type='nm').withZeropoint(zp)
        return self.inventory[component].calculateFlux(bp)

    def calculateEW(self, emline, frame='obs'):
        z = self.pars['z']
        if self.pars.get(f'em_{emline}_flux', None) is not None:
            S = self.pars[f'em_{emline}_flux']
            zsys = self.pars.get(f'em_{emline}_vsys', 0.0)/_c_kms
            wcen = np.atleast_1d(LINE_LAMBDAS[emline].to('nm').value*(1.+z)*(1+zsys)).mean()
            C = self.__call__(wcen, component='continuum', flux_type='flambda')
            return S/C if frame=='obs' else (S/C)/((1.+z)*(1+zsys))
        else:
            print(f'Emission line {emline} not detected in the SED!')
            return np.nan