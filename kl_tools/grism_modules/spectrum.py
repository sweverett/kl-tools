import os
import numpy as np
from astropy.units import Unit
import astropy.units as units
import astropy.constants as constants
import sys
import galsim

sys.path.insert(0, '../')
import kl_tools.parameters as parameters
import kl_tools.emission as emission
from kl_tools.emission import LINE_LAMBDAS

class Spectrum(emission.SED):
    '''
    This class describe the obs-frame SED template of a source galaxy, includ-
    ing emission lines and continuum components.
    This is mostly a wrapper of the galsim.SED class object

    Note that this class should only capture the intrinsic properties of
    the galaxy SED, like
        - redshift
        - continuum and emission line flux density
        - emission line width
        - [dust?]
    Other extrinsic properties, like
        - sky background
        - system bandpass
        - spectral resolution and dispersion
        - exposure time
        - collecting area
        - pixelization
        - noise
    will be implemented in mock observation module
    '''
    ### settings for the whole class
    # default parameters
    _default_pars = {
        'template': '../../data/Simulation/GSB2.spec',
        'wave_type': 'Ang',
        'flux_type': 'flambda',
        'z': 0.0,
        'spectral_range': (50, 50000), # nm
        # obs-frame continuum normalization (nm, erg/s/cm2/nm)
        'obs_cont_norm': (400, 0.),
        # spectral resolution at 1 micron, assuming dispersion per pixel
        #'resolution': 3000,
        # a dict of line names and obs-frame flux values (erg/s/cm2)
        'lines': {'Ha': 1e-15},
        # intrinsic linewidth in nm
        'line_sigma_int': {'Ha': 0.5,},
        #'line_hlr': (0.5, Unit('arcsec')),
        'thin': -1,
    }
    # units conversion
    _h = constants.h.to('erg s').value
    _c = constants.c.to('nm/s').value
    # build-in emission line species and info
    _valid_lines = {k:v.to('nm').value for k,v in LINE_LAMBDAS.items()}

    def __init__(self, pars):
        '''
        Initialize SED class object with parameters dictionary
        '''
        self.pars = Spectrum._default_pars.copy()
        self.updatePars(pars)
        _con = self._addContinuum()
        _emi = self._addEmissionLines()
        self.spectrum = _con + _emi
        if self.pars['thin'] > 0:
            self.spectrum = self.spectrum.thin(rel_err=self.pars['thin'])
        super(Spectrum, self).__init__(self.pars['spectral_range'][0],
self.pars['spectral_range'][1], 3000, 'nm')

    def updatePars(self, pars):
        '''
        Update parameters
        '''
        for key, val in pars.items():
            self.pars[key] = val

    def _addContinuum(self):
        '''
        Build and return continuum GalSim SED object
        '''
        template = self.pars['template']
        if not os.path.isfile(template):
            raise OSError(f'Can not find template file {template}!')
        # build GalSim SED object out of template file
        _template = np.genfromtxt(template)
        _table = galsim.LookupTable(x=_template[:,0], f=_template[:,1],)
        SED = galsim.SED(_table,
                         wave_type=self.pars['wave_type'],
                         flux_type=self.pars['flux_type'],
                         redshift=self.pars['z'],
                         _blue_limit=self.pars['spectral_range'][0],
                         _red_limit=self.pars['spectral_range'][1])
        # normalize the SED object at observer frame
        # erg/s/cm2/nm -> photons/s/cm2/nm
        # TODO: add more flexible normalization parametrization
        norm = self.pars['obs_cont_norm'][1]*self.pars['obs_cont_norm'][0]/\
            (Spectrum._h*Spectrum._c)
        return SED.withFluxDensity(target_flux_density=norm,
                                   wavelength=self.pars['obs_cont_norm'][0])

    def _addEmissionLines(self):
        '''
        Build and return Gaussian emission lines GalSim SED object
        '''
        # init LookupTable for rest-frame SED
        lam_grid = np.arange(self.pars['spectral_range'][0]/(1+self.pars['z']),
                             self.pars['spectral_range'][1]/(1+self.pars['z']),
                             0.1)
        flux_grid = np.zeros(lam_grid.size)
        # Set emission lines: (specie, observer frame flux)
        all_lines = Spectrum._valid_lines.keys()
        norm = -1
        for line, flux in self.pars['lines'].items():
            # sanity check
            rest_lambda = np.atleast_1d(Spectrum._valid_lines[line])
            flux = np.atleast_1d(flux)
            line_sigma_int = np.atleast_1d(self.pars['line_sigma_int'][line])
            if rest_lambda is None:
                raise ValueError(f'{line} is not a valid emission line! '+\
                        f'For now, line must be one of {all_lines}')
            else:
                assert rest_lambda.size == flux.size, f'{line} has'+\
                f' {rest_lambda.size} lines but {flux.size} flux are provided!'
            # build rest-frame f_lambda SED [erg s-1 cm-2 nm-1]
            # then, redshift the SED. The line width will increase automatically
            for i,cen in enumerate(rest_lambda):
                _lw_sq = line_sigma_int[i]**2
                # erg/s/cm2 -> erg/s/cm2/nm
                _norm = flux[i]/np.sqrt(2*np.pi*_lw_sq)
                flux_grid += _norm * np.exp(-(lam_grid-cen)**2/(2*_lw_sq))
                # also, calculate normalization factor for obs-frame spectrum
                # convert flux units: erg/s/cm2/nm -> photons/s/cm2/nm
                # only one emission line needed
                if(norm<0):
                    norm_lam = cen*(1+self.pars['z'])
                    norm = flux[i]*norm_lam/(Spectrum._h*Spectrum._c)/\
                                np.sqrt(2*np.pi*_lw_sq*(1+self.pars['z'])**2)

        _table = galsim.LookupTable(x=lam_grid, f=flux_grid,)
        SED = galsim.SED(_table,
                         wave_type='nm',
                         flux_type='flambda',
                         redshift=self.pars['z'],)
        # normalize to observer-frame flux
        SED = SED.withFluxDensity(target_flux_density=norm,
                                  wavelength=norm_lam)
        return SED
