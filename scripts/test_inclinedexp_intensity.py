# unit test for InclinedExponential intensity profile
import os
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
import sys, os
sys.path.insert(0, './grism_modules')

from kl_tools.utils import get_test_dir 
from kl_tools.intensity import build_intensity_map

if __name__=='__main__':
    # generate an inclined exponential intensity profile
    print("\nRunning unit test for InclinedExponential intensity profile...\n")
    unit_test_data_fn = get_test_dir() + '/unit_tests/inclinedexp_intensity.npy'
    kwargs = {
        'scale': 0.1,  # pixel scale in arcsec. 
        'flux': 1.0, 
        'hlr': 0.5, # half-light radius in arcsec
        'imap_return_gal': False,
        'theory_Nx': 64,
        'theory_Ny': 64,
    }
    pars = {
        'g1': 0.1,
        'g2': 0.1,
        'theta_int': 0.1,
        'sini': 0.5,
        'intensity': {
            'dx_disk': 0.1,
            'dy_disk': 0.1,
            'dx_spec': 0.0,
            'dy_spec': 0.0,
        },
        'run_options':{
            'imap_return_gal': False, # return galsim galaxy object
        },
    }
    prof = build_intensity_map('inclined_exp', None, kwargs)
    image = prof.render({}, None, pars)["phot"]

    original_image = np.load(unit_test_data_fn)
    assert np.allclose(image, original_image, atol=1e-6)
    print("InclinedExponential intensity profile unit test passed!")
