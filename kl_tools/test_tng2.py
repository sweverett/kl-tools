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

slit_pars ={'slit_width': 0.5,
'slit_angle': np.pi/2,
'offset_x': -0.7,
'offset_y': 0.0}

tng = tngsim.TNGsimulation()
pars = CubePars(meta_pars)

tng.set_subhalo(2)
datacube = tng._generateCube(pars, rescale=0.1)

plt.imshow(np.sum(datacube, axis=0), origin='lower')
plt.savefig('datacube.png')
plt.colorbar()
plt.close()

plt.plot(l[1:-1], np.sum(np.sum(datacube, axis=1), axis=1))
plt.savefig('1D projection.png')
plt.close()

plt.imshow(np.sum(datacube, axis=1))
plt.savefig('2D projection.pdf')
plt.close()

spectrum = tng.to_slit(pars, slit_angle=np.pi/4)
spectrum.plot_datavector()

# plt.imshow(spectrum)
# plt.show()