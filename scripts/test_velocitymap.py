# unit test for VelocityMap 
import os
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
import sys, os
import astropy.units as units
import matplotlib.pyplot as plt
sys.path.insert(0, './grism_modules')

from kl_tools.utils import get_test_dir 
from kl_tools.velocity import build_model, VelocityMap

if __name__=='__main__':
    # generate a velocity map
    print("\nRunning unit test for VelocityMap...\n")
    unit_test_data_fn = get_test_dir() + '/unit_tests/velocity_map.npy'
    X, Y = np.meshgrid(
            np.linspace(-1.5, 1.5, 128),
            np.linspace(-1.5, 1.5, 128),
            )
    model_pars = {
        'v0': 10,
        'vcirc': 200.0,
        'rscale': 0.05,
        'sini': 0.9,
        'theta_int': np.pi/6,
        'g1': 0.5,
        'g2': 0.2,
        'r_unit': units.Unit('pixel'),
        'v_unit': units.km / units.s,
    }
    vmap = VelocityMap('default', model_pars)
    varray = vmap(
            'obs', X, Y, normalized=True, use_numba=False,
            )
    print(f'Velocity map array shape: {varray.shape}')
    print(f'Velocity map array stats: min={varray.min()}, max={varray.max()}, mean={varray.mean()}')
    original_image = np.load(unit_test_data_fn)
    assert np.allclose(varray, original_image, atol=1e-6)
    print("VelocityMap unit test passed!")