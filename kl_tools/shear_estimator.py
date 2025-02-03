'''
This file aims to provide classes and methods used to do a quick kinematic lensing analysis given independent estimates of the photometric and kinematic quantities
'''

import numpy as np

class ShearEstimator(object):
    '''
    This class is used to estimate the shear of a galaxy given independent estimates of the photometric and kinematic quantities
    '''

    _req_photometric = [
        'eobs'
        ]

    _req_kinematic = [
        'vmajor', # asymtotic rotation velocity along major axis
        'vminor', # asymtotic rotation velocity along minor axis
        ]

    # TODO: eventually formalize this
    _opt_photometric = {
        'sini',
        'qz',
    }
    _opt_kinematic = {
        'vtf', # predicted asymptotic major axis velocity from Tully-Fisher
    }

    # these refer to the papers that provide sets of quanity estimators
    _estimators = ['jiachuan']

    def __init__(self, photometric: dict, kinematic: dict, estimator='jiachuan') -> None:
        '''
        Initialize the class with the photometric and kinematic quantities. The required entries are defined in the class attributes
        '''

        # might use terminus later
        for field in self._req_photometric:
            if field not in photometric:
                raise ValueError(f'{field} is required in the photometric dictionary')

        for field in self._req_kinematic:
            if field not in kinematic:
                raise ValueError(f'{field} is required in the kinematic dictionary')

        self.photometric = photometric
        self.kinematic = kinematic

        if estimator not in self._estimators:
            raise ValueError(f'{estimator} is not a valid estimator')
        self.estimator = estimator

        # derived quantities (usually)
        self.sini = None
        self.eint = None
        self.gtan = None
        self.gcross = None

        return

    def estimate_shear(self, require_qz: bool=False, vb: bool=True) -> None:
        '''
        Estimate the shear of the galaxy given the photometric and kinematic quantities

        The general steps are as follows:
        1) Compute or lookup a Tully-Fisher velocity (vtf)
        2) Estimate inclination angle sini from the ratio between observed v_major and vtf
        3) Estimate intrinsic ellipticity e_int using both sini and qz (if provided)
        4) Estimate (g1, g2) from the observed ellipticity and e_int

        The output is a tuple of the estimated ellipticity components (e1, e2)

        Parameters
        ----------
        require_qz : bool
            If True, raise an error if qz is not provided in the photometric dictionary
        vb : bool
            If True, print the estimated shear
        '''

        if require_qz and 'qz' not in self.photometric:
            raise ValueError('qz is required to estimate the intrinsic ellipticity if require_qz is True')

        # step 1: compute or lookup Tully-Fisher velocity
        self._set_vtf()

        # step 2: estimate inclination angle (sometimes provided)
        if 'sini' not in self.kinematic:
            self._estimate_sini()
        else:
            self.sini = self.kinematic['sini']

        # step 3: estimate intrinsic ellipticity
        self._estimate_intrinsic_ellipticity()

        # step 4: estimate shear
        self._estimate_shear()

        if vb is True:
            print(f'Estimated shear: ({self.gtan:.4f}, {self.gcross:.4f})')

        return

    def _set_vtf(self) -> None:
        '''
        Compute or lookup the Tully-Fisher velocity
        '''

        # if vtf is provided, use that
        if 'vtf' in self.kinematic:
            return

        # otherwise, compute it
        # TODO: ...

        return

    def _estimate_sini(self) -> None:
        '''
        Estimate the inclination angle sini from the ratio between observed v_major and vtf
        '''

        sini = - self.kinematic['v_major'] / self.kinematic['vtf']
        self.sini = sini

        return

    def _estimate_intrinsic_ellipticity(self, marginalize_qz: bool=False, qz0=0.25) -> None:
        '''
        Estimate the intrinsic ellipticity using the observed ellipticity and the inclination angle

        Parameters
        ----------
        marginalize_qz : bool
            If True, marginalize over the intrinsic axis ratio qz if not provided in initialization. Otherwise, use the default value
        qz0 : float
            Default value for the intrinsic axis ratio if marginalize_qz is False. Also used to center the prior for marginzalization
        '''

        if 'qz' in self.photometric:
            qz = self.photometric['qz']
        else:
            # use prior around 0.25 (Ubler et al. 2017)
            if marginalize_qz is True:
                # TODO: ...
                pass
            else:
                qz = qz0

        sini = self.photometric['sini']
        sini2 = sini**2

        sqrt = np.sqrt(1 - (1-qz**2)*sini2)
        eint = (1 - sqrt) / (1 + sqrt)

        self.eint = eint

        return

    def _estimate_shear(self) -> None:
        '''
        Estimate the shear given the observed ellipticity and the intrinsic ellipticity
        '''

        eobs = self.photometric['eobs']
        eobs2 = eobs**2

        eint = self.eint
        eint2 = eint**2

        cosi = np.sqrt(1 - self.sini**2)

        vminor = self.kinematic['vminor']
        vmajor = self.kinematic['vmajor']

        gtan = (eobs2 - eint2) / (2*eobs2*(1-eint2))

        gcross = abs(vminor / vmajor) * (2*eint) / (cosi*(2*eint + 1 + eobs2))

        self.gtan = gtan
        self.gcross = gcross

        return