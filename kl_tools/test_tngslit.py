from tngslit import TNGSlit

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np

from mocks import setup_simple_emission_line
import tngsim
from cube import CubePars

R = 5000
line_val = 6563
width = line_val / R
z = 0.3
dlambda = 0.02

l = np.linspace(line_val*(1+z)-10, line_val*(1+z)+10, 65)

meta_pars = {'bandpasses': {'lambda_blue': (l[0]+l[1])/2, 'lambda_red':(l[-2]+l[-1])/2, 'dlambda':l[1]-l[0], 'unit':'Angstrom'},
'emission_lines':[setup_simple_emission_line(line_val, u.Angstrom, R, z, width)],
'pix_scale': 0.1185/2, 
'shape':(len(l)-2, 64, 64), 
'resolution': R,
'psfFWHM': 0.01}

slit_pars = {'slit_width': 0.5,
'slit_angle': np.pi/2,
'offset_x': -0.7,
'offset_y': 0.0}

tngslit = TNGSlit(meta_pars, subhalo_id=2)

plt.imshow(np.sum(tngslit.datacube.data, axis=), origin='lower')
plt.show()

# spectrum = tng.to_slit(pars)
# plt.imshow(spectrum)
# plt.show()