'''
This script reads in the KROSS source catalog and makes various selections
in order to produce a KL lensing catalog that we can use to compare to
archival lensing measurements, e.g. COSMOS.
'''

from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.units import Unit, deg

from kl_tools.utils import get_base_dir, make_dir, plot
from kl_tools.kross.tfr import estimate_vtf
from kl_tools.kross.shear import PointShear, MLEShear

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-e', '--estimator', type=str, default='point',
                        choices=['point', 'mle'],
                        help='Estimator to use for shear calculations')
    parser.add_argument('-s', '--show', action='store_true', default=False,
                        help='Show plots')
    parser.add_argument('-o', '--overwrite', action='store_true', default=False,
                        help='Overwrite existing files')
    parser.add_argument('-vb', '--verbose', action='store_true', default=False)

    return parser.parse_args()

def main():

    #---------------------------------------------------------------------------
    # initial setup

    args = parse_args()
    show = args.show
    estimator = args.estimator
    overwrite = args.overwrite
    vb = args.verbose

    plot_dir = get_base_dir() / 'plots/kl_sample'
    make_dir(plot_dir)

    kross_dir = get_base_dir() / 'data/kross'
    kross_file = kross_dir / 'kross_release_v2.fits'
    kross = Table.read(kross_file)

    #---------------------------------------------------------------------------
    # make initial sample plots

    # first, look at the distribution of kinematic types
    plt.hist(kross['KIN_TYPE'])
    plt.xlabel('Kinematic Type')
    plt.title('KROSS Distribution of Kinematic Types; All Galaxies')

    outfile = plot_dir / 'kinematic_types_all.png'
    plot(show, save=True, out_file=outfile)

    # ...

    #---------------------------------------------------------------------------
    # make selections

    # The corrected "intrinsic" circular velocity
    # NOTE: already has inclination correction, plus beam smearing
    vc = kross['VC']
    # velocity dispersion from aperture spectrum
    sigma = kross['SIGMA_TOT']
    # The inferred inclination angle, θim, with error. If θim < 25 then 
    # excluded from the analyses.
    theta_im = kross['THETA_IM']
    # If =1 then the inclination angle was fixed to 53 ± 18 deg
    theta_flag = kross['THETA_FLAG']
    # Quality 1: Hα detected, spatially-resolved and both θim and R1/2 were measured from the broad-band
    # Quality 2: Hα detected and spatially resolved but θim was fixed (see THETA_FLAG) and/or R1/2 was estimated from the kinematic (see R_FLAG)
    # Quality 3: Hα detected and resolved in the IFU data but only an upper limit on R1/2
    # Quality 4: Hα detected but unresolved in the IFU data.
    quality = kross['QUALITY_FLAG'] 
    # If =1 AGN emission affecting the emission-line properties and are 
    # excluded from the final analyses.
    agn_flag = kross['AGN_FLAG']
    # If ==1 unphysical measurements for the rotational velocities and/or the 
    # half-light radii and are excluded from the final analyses
    irr_flag = kross['IRR_FLAG'] 
    # If =1, vC extrapolated >2 pixels beyond the extent of the data, if =2 
    # then vC was estimated by scaling σtot
    extrap_flag = kross['EXTRAP_FLAG'] 
    # Kinematic classification: RT+ “gold” rotationally dominated; RT: 
    # rotationally dominated; DN: dispersion dominated; X: IFU data is 
    # spatially unresolved;
    kin_type = kross['KIN_TYPE']

    selection = np.where(
        ((vc / sigma) > 1) &
        (quality == 1) &
        (agn_flag == 0) &
        (irr_flag == 0) &
        (theta_flag == 0) &
        (extrap_flag == 0) &
        (kin_type == 'RT+')
    # TODO: decide if we want the less conservative selection:
    #     ((kin_type == 'RT ') | (kin_type == 'RT+'))
    )

    # NOTE: for *no* selection:
    # selection = np.arange(0, len(kross))

    Nkross = len(kross)
    sample = kross[selection]
    Nsample = len(sample)

    if vb is True:
        print(f'Sample size before selection: {Nkross}')
        print(f'Sample size after selection: {Nsample}')

    #---------------------------------------------------------------------------
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
    boa = sample['B_O_A'] # Observed axis ratio b/a from the broad-band image
    rmag = sample['R_AB']
    mstar = sample['MASS']

    pa_im = sample['PA_IM']
    name = sample['NAME']

    # derived columns
    log_mstar = np.log10(mstar)
    eobs = (1. - np.sqrt(1. - boa**2)) / (1. + np.sqrt(1. - boa**2))
    # "observed" vcirc with beam smearing correction but *NOT* inclination 
    # correction
    vcirc = vc * np.sin(np.deg2rad(theta_im)) 

    #---------------------------------------------------------------------------
    # estimate vcirc using logmstar and the Tully-Fisher relation
    vtf = estimate_vtf(log_mstar)

    # now make a plot of the distribution of vtf and vcirc
    bins = np.linspace(0, 300, 25)
    plt.hist(vtf, ec='k', bins=bins, label='vtf', alpha=0.5)
    plt.hist(vcirc, ec='k', bins=bins, label='vcirc', alpha=0.5)
    plt.legend()
    plt.xlabel('v (km/s)')

    out_file = plot_dir / 'vtf_vcirc_compare.png'
    plot(show, save=True, out_file=out_file)

    # plot the distribution of sini = vcirc / vtf
    sini = vcirc / vtf
    plt.hist(sini, ec='k', bins=20)
    plt.xlabel('sini (vcirc / vtf)')

    out_file = plot_dir / 'sini_tfr.png'
    plot(show, save=True, out_file=out_file)

    # plot the distribution of vcirc / sigma
    bins = np.linspace(0, 4, 40)
    plt.hist(vcirc / sigma, bins=bins, ec='k')
    plt.axvline(1, c='k', ls='--')
    plt.xlabel('vcirc / sig_v')

    out_file = plot_dir / 'vcirc_sigma.png'
    plot(show, save=True, out_file=out_file)

    # plot the distribution of vc / sigma
    plt.hist(vc / sigma, bins=bins, ec='k')
    plt.axvline(1, c='k', ls='--')
    plt.xlabel('vc / sig_v)')

    out_file = plot_dir / 'vc_sigma.png'
    plot(show, save=True, out_file=out_file)

    Nsini = len(sini)
    if vb is True:
        print(f'Sample size with valid sini given v_TF: {Nsini}')

    #---------------------------------------------------------------------------
    # estimate shear from the KROSS quantities

    if estimator == 'point':
        shear = PointShear()
    elif estimator == 'mle':
        shear = MLEShear()

    # plotting options
    Nbins = 30

    # estimate sini from vcirc and TF velocity:
    sini = shear.estimate_sini(vcirc, vtf)
    plt.hist(sini, bins=Nbins, ec='k')
    plt.xlabel('sini')

    out_file = plot_dir / 'sini_sample.png'
    plot(show, save=True, out_file=out_file)

    # estimate intrinsic ellipticity from sini and qz (assumed):
    eint = shear.estimate_eint(sini)
    plt.hist(eint, bins=Nbins, ec='k')
    plt.xlabel('eint')

    out_file = plot_dir / 'eint_sample.png'
    plot(show, save=True, out_file=out_file)

    # look at difference in eobs vs estimated eint
    edelt = eobs - eint
    edelt2 = eobs**2-eint**2
    plt.hist(edelt, bins=Nbins, ec='k', label='eobs-eint', alpha=0.75)
    plt.hist(edelt2, bins=Nbins, ec='k', label='eobs^2-eint^2', alpha=0.75)
    plt.axvline(0, c='k', ls='--', lw=2)
    plt.xlabel('ellipticity difference')
    plt.legend()

    out_file = plot_dir / 'obs_eint_diff.png'
    plot(show, save=True, out_file=out_file)

    # show shear response
    response = (2 * eobs * (1-eint**2))
    plt.hist(response, bins=Nbins, ec='k')
    plt.xlabel('response (2 * eobs**2 * (1-eint**2))')

    out_file = plot_dir / 'shear_response.png'
    plot(show, save=True, out_file=out_file)

    # estimate shear values from estimated eint:
    gplus = shear.estimate_gplus(eobs, eint)
    plt.hist(gplus, bins=np.linspace(-1, 1, Nbins), ec='k')
    plt.axvline(0, c='k', ls='--', lw=2)
    plt.xlabel('g+')
    # plt.xlim(-1, 1)
    plt.show()

    # gcross = estimate_gcross(vcirc, vmin, eobs, eint, sini)
    # plt.hist(gcross, bins=Nbins, ec='k')
    # plt.xlabel('gx')

    # plot s2n
    verr = vc_err_h - vc_err_l
    s2n = vcirc / verr
    plt.hist(s2n, ec='k', bins=50)
    plt.xlabel('vcirc / verr')
    # plt.xlim(0, 10)

    # TODO: finish!!

    return

if __name__ == '__main__':
    main()
