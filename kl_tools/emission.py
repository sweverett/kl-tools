import numpy as np
from scipy.interpolate import interp1d
from numpy import interp
import astropy.units as u

import utils

import ipdb

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
    def __init__(self, start, end, resolution, unit):

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
