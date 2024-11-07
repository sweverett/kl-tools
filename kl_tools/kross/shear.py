'''
Helper classes & methods for the KROSS KL measurement. Should eventually be 
refactored into the main kl_tools code
'''

import numpy as np

class PointShear(object):
    '''
    A simple class to estimate various shear-related quantities from the KROSS 
    data, using a point estimator approach.
    '''

    def estimate_sini(vmax, vtf):
        return vmax / vtf

    def estimate_eint(sini, qz=0.25):
        factor = np.sqrt(1 - (1-qz)**2 * sini**2)
        eint = (1 - factor) / (1 + factor)
        return eint

    def estimate_gplus(eobs, eint):
        gplus = (eobs**2 - eint**2) / (2 * eobs * (1-eint**2))
        return gplus

    def estimate_gcross(vmax, vmin, eobs, eint, sini):
        cosi = np.sqrt(1 - sini**2)
        gcross = abs(vmin / vmax) * (2 * eint) / (cosi * (2*eint + 1 + eobs**2))
        return gcross

class MLEShear(PointShear):
    '''
    A simple class to estimate various shear-related quantities from the KROSS 
    data, using a full MLE approach
    '''

    def estimate_gplus(vmax, vmin, eobs, sini):
        pass

    def estimate_gcross(vmax, vmin, eobs, sini):
        pass