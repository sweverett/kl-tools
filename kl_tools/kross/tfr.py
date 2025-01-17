'''
Helper methods related to the Tully-Fisher Relation (TFR).
'''

import numpy as np

def estimate_vtf(
        log_mstar,
        alpha=4.51,
        log_M100=9.49,
        alpha_err=0.26,
        return_error_bounds=False
        ):
    '''
    Values are for K-band. Logs are base 10. Taken from Bell et al. 2001:
    https://ui.adsabs.harvard.edu/abs/2001ApJ...550..212B/abstract
    '''

    log_vtf = (1. / alpha) * (log_mstar - log_M100)
    vtf = 100*10**log_vtf

    if return_error_bounds is True:
        err_low  = 100*10**(log_vtf-alpha_err)
        err_high = 100*10**(log_vtf+alpha_err)
        v_bounds = (err_low, err_high)
        return vtf, v_bounds
    else:
        return vtf
    
