import numpy as np
from scipy.interpolate import interp1d
import astropy.units as u

import utils

class EmissionLine(object):
    '''
    Holds relevant information for a specific emission line
    '''

    _req_line_pars = ['value', 'std', 'unit']
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
            Line central value
        std: float
            Width of line
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

        self._setup_sed()

        return

    def _setup_sed(self):

        sed_pars = self.sed_pars
        line_pars = self.line_pars

        lblue = sed_pars['lblue']
        lred = sed_pars['lred']
        res = sed_pars['resolution']
        lam_unit = sed_pars['unit']

        lambdas = np.arange(lblue, lred+res, res) * lam_unit

        # Model emission line SED as gaussian
        line_unit = line_pars['unit']
        mu  = line_pars['value'] * line_unit
        std = line_pars['std'] * line_unit

        norm = 1. / (std * np.sqrt(2.*np.pi))
        chi = ((lambdas - mu)/std).value
        gauss = norm * np.exp(-0.5*chi**2)

        self.sed = interp1d(lambdas, gauss)

        return

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
    'O6_2': (1031.93, 1037.62) * u.Unit('A'),
    'Lya': 1215.670 * u.Unit('A'),
    'N5': (1238.821, 1242.804) * u.Unit('A'),
    'C4': (1548.203, 1550.777) * u.Unit('A'),
    'Mg2': 2796.290 * u.Unit('A'),
    'O2': (3727.048, 3729.832) * u.Unit('A'),
    'Ne3': 3870.115 * u.Unit('A'),
    'Ne32': 3968.872 * u.Unit('A'),
    'Hzet': 3890.109 * u.Unit('A'),
    'Heps': 3971.154 * u.Unit('A'),
    'Hd': 4102.852 * u.Unit('A'),
    'Hg': 4341.647 * u.Unit('A'),
    'O3_3': 4364.400 * u.Unit('A'),
    'Hb': 4862.650 * u.Unit('A'),
    'O3_1': 4960.263 * u.Unit('A'),
    'O3_2': 5008.208 * u.Unit('A'),
    'He1': 5877.217 * u.Unit('A'),
    'O1': 6302.022 * u.Unit('A'),
    'N2_1': 6549.825 * u.Unit('A'),
    'Ha': 6564.589 * u.Unit('A'),
    'N2_2': 6585.255 * u.Unit('A'),
    'S2_1': 6718.271 * u.Unit('A'),
    'S2_2': 6732.645 * u.Unit('A'),
}
