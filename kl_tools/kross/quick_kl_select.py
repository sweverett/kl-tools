'''
TODO: Quick script version of the orig KL selection notebook; should be cleaned up and made into a proper module
'''

import numpy as np
from astropy.table import Table
import fitsio
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from astropy.units import Unit, deg
# import esutil.htm

from kl_tools.utils import get_base_dir

plt.rcParams['figure.facecolor'] = 'white'

data_dir = get_base_dir() / 'data/kross'
kross_file = data_dir / 'kross_release_v2.fits'
kross = Table.read(kross_file)
print(kross)
print()
print('columns:')
for col in kross.columns:
    print(col)


# selection

vc = kross['VC'] # The corrected "intrinsic" circular velocity **NOTE** already has inclination correction, plus beam smearing
sigma = kross['SIGMA_TOT'] # velocity dispersion from aperture spectrum
theta_im = kross['THETA_IM'] # TThe inferred inclination angle, θim, with error. If θim < 25 then excluded from the analyses.
theta_flag = kross['THETA_FLAG'] # If =1 then the inclination angle was fixed to 53o ± 18.
quality = kross['QUALITY_FLAG'] # Quality 1: Hα detected, spatially-resolved and both θim and R1/2 were measured from the broad-band
# Quality 2: Hα detected and spatially resolved but θim was fixed (see THETA_FLAG) and/or R1/2 was estimated from the kinematic (see R_FLAG)
# Quality 3: Hα detected and resolved in the IFU data but only an upper limit on R1/2
# Quality 4: Hα detected but unresolved in the IFU data.
agn_flag = kross['AGN_FLAG'] # If =1 AGN emission affecting the emission-line properties and are excluded from the final analyses.
irr_flag = kross['IRR_FLAG'] # If =1 unphysical measurements for the rotational velocities and/or the half-light radii and are excluded from the final analyses
extrap_flag = kross['EXTRAP_FLAG'] # If =1, vC extrapolated >2 pixels beyond the extent of the data, if =2 then vC was estimated by scaling σtot
kin_type = kross['KIN_TYPE'] # Kinematic classification: RT+ “gold” rotationally dominated; RT: rotationally dominated; DN: dispersion dominated; X: IFU data is spatially unresolved;

selection = np.where(
    ((vc / sigma) > 1) &
    (quality == 1) &
    (agn_flag == 0) &
    (irr_flag == 0) &
    (theta_flag == 0) &
    (extrap_flag == 0) &
    # (kin_type == 'RT+')
    ((kin_type == 'RT ') | (kin_type == 'RT+'))
)

# for *no* selection:
# selection = np.arange(0, len(kross))

Nkross = len(kross)
sample = kross[selection]
Nsample = len(sample)

print(f'sample before selection: {Nkross}')
print(f'sample after selection: {Nsample}')

# update cols with selection
vc = vc[selection]
sigma = sigma[selection]
theta_im = theta_im[selection]
theta_flag = theta_flag[selection]
quality = quality[selection]
agn_flag = agn_flag[selection]
irr_flag = irr_flag[selection]
extrap_flag = extrap_flag[selection]
kin_type = kin_type[selection]

# get more useful columns
ra = sample['RA']
dec = sample['DEC']
z = sample['Z']
vc_err_h = sample['VC_ERR_H']
vc_err_l = sample['VC_ERR_L']
n = sample['VDW12_N'] # sersic index
boa = sample['B_O_A'] # The observed axis ratio b/a from the broad-band image
rmag = sample['R_AB']
mstar = sample['MASS']
pa_im = sample['PA_IM']
name = sample['NAME']

# derived columns
log_mstar = np.log10(mstar)
vcirc = vc * np.sin(np.deg2rad(theta_im)) # "observed" vcirc with beam smearing correction but *NOT* inclination correction
vcirc_err = (vc_err_h - vc_err_l) * np.sin(np.deg2rad(theta_im)) # estimated error on 2D vcirc
eobs = (1. - np.sqrt(1. - boa**2)) / (1. + np.sqrt(1. - boa**2))

def estimate_vtf(log_mstar, alpha=4.51, log_M100=9.49):
    '''
    Values are for K-band. Logs are base 10
    '''
    log_vtf = (1. / alpha) * (log_mstar - log_M100)
#     print(10**log_vtf)
    return 100*10**log_vtf

vtf = estimate_vtf(log_mstar)

# OLD WAY:
def estimate_sini_old(vmax, vtf):
    return vmax / vtf

from kl_tools.kross.analytic_estimators import estimate_sini_from_vtf_single

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

# plotting options
Nbins = 30

# estimate sini from vcirc and TF velocity (OLD):
sini_old = estimate_sini_old(vcirc, vtf)
plt.hist(sini_old, bins=Nbins, ec='k')
plt.xlabel('sini (OLD)')
plt.show()

import ipdb; ipdb.set_trace()
sini = -1. * np.ones(len(vcirc))
for i in range(len(vcirc)):
    sini[i] = estimate_sini_from_vtf_single(vcirc[i], vtf[i], vcirc_err[i])
plt.hist(sini, bins=Nbins, ec='k')
plt.xlabel('sini (analytic)')
plt.show()

# compare the two sini methods
Nold = len(sini_old)
Nnew = len(sini)
plt.hist(sini_old, ec='r', histtype='step', alpha=0.7, bins=np.linspace(0, 1, 20), label=f'Old ({Nold})')
plt.hist(sini, ec='k', histtype='step', alpha=0.7, bins=np.linspace(0, 1, 20), label=f'Analytic ({Nnew})')
plt.xlabel('sin(i)')
plt.show()

# estimate intrinsic ellipticity from sini and qz (assumed):
eint = estimate_eint(sini)
plt.hist(eint, bins=Nbins, ec='k')
plt.xlabel('eint')
plt.show()

# look at difference in eobs vs estimated eint
edelt = eobs - eint
edelt2 = eobs**2-eint**2
plt.hist(edelt, bins=Nbins, ec='k', label='eobs-eint', alpha=0.75)
plt.hist(edelt2, bins=Nbins, ec='k', label='eobs^2-eint^2', alpha=0.75)
plt.axvline(0, c='k', ls='--', lw=2)
plt.xlabel('ellipticity difference')
plt.legend()
plt.show()

# show shear response
response = (2 * eobs * (1-eint**2))
plt.hist(response, bins=Nbins, ec='k')
plt.xlabel('response (2 * eobs**2 * (1-eint**2))')
plt.show()

# estimate shear values from estimated eint:
gplus = estimate_gplus(eobs, eint)
plt.hist(gplus, bins=np.linspace(-1, 1, Nbins), ec='k')
plt.axvline(0, c='k', ls='--', lw=2)
plt.xlabel('g+')
# plt.xlim(-1, 1)
plt.show()

# gcross = estimate_gcross(vcirc, vmin, eobs, eint, sini)
# plt.hist(gcross, bins=Nbins, ec='k')
# plt.xlabel('gx')