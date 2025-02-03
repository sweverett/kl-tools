'''
Helper classes & methods for the KROSS KL measurement. Should eventually be 
refactored into the main kl_tools code
'''

import numpy as np

import kl_tools.kross.analytic_estimators as analytic_estimator

class PointShear(object):
    '''
    A simple class to estimate various shear-related quantities from the KROSS 
    data, using a point estimator approach.
    '''

    @staticmethod
    def estimate_sini(vmax, vtf):
        return vmax / vtf

    @staticmethod
    def estimate_eint(sini, qz=0.25):
        factor = np.sqrt(1 - (1-qz)**2 * sini**2)
        eint = (1 - factor) / (1 + factor)
        return eint

    @staticmethod
    def estimate_gplus(eobs, eint):
        gplus = (eobs**2 - eint**2) / (2 * eobs * (1-eint**2))
        return gplus

    @staticmethod
    def estimate_gcross(vmax, vmin, eobs, eint, sini):
        cosi = np.sqrt(1 - sini**2)
        gcross = abs(vmin / vmax) * (2 * eint) / (cosi * (2*eint + 1 + eobs**2))
        return gcross

class MAPShear(PointShear):
    '''
    A simple class to estimate various shear-related quantities from the KROSS 
    data, using a full MAP approach
    '''

    @staticmethod
    def estimate_sini(vmax, vtf, vmax_err, vtf_err=0.05):

        Nsamples = len(vmax)
        assert Nsamples == len(vtf), 'vmax and vtf must have the same length'

        if Nsamples == 1:
            sini = analytic_estimator.estimate_sini_from_vtf_single(
                vmax, vtf, vmax_err
            )
        else:
            sini = np.nan * np.empty(Nsamples)
            for i in range(Nsamples):
                sini[i] = analytic_estimator.estimate_sini_from_vtf_single(
                    vmax[i], vtf[i], vmax_err[i]
                )

        return sini
