'''
This script reads in the KROSS source catalog and makes various selections
in order to produce a KL lensing catalog that we can use to compare to
archival lensing measurements, e.g. COSMOS.
'''

from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table, join
from astropy.coordinates import SkyCoord
from astropy.units import Unit, deg
from astropy.coordinates import SkyCoord, match_coordinates_sky

from kl_tools.utils import get_base_dir, make_dir, plot
from kl_tools.kross.tfr import estimate_vtf
from kl_tools.kross.shear import PointShear, MAPShear

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        '-e',
        '--estimator',
        type=str,
        default='both',
        choices=['point', 'map', 'both'],
        help='Estimator to use for shear calculations'
        )
    parser.add_argument(
        '-o', '--overwrite', action='store_true', default=False,
        help='Overwrite existing files'
        )
    parser.add_argument(
        '-c', '--cosmos_only', action='store_true', default=False,
        help='Save only sources in the COSMOS field'
    )
    parser.add_argument(
        '-v', '--verbose', action='store_true', default=False,
        help='Print more verbose information'
        )
    parser.add_argument(
        '-s', '--show', action='store_true', default=False,
        help='Show plots'
        )

    return parser.parse_args()

def _setup_kl_tools_v2d(kids, name_map):
    '''
    Set up the 2D velocity data from KL tools, if already pre-computed

    Parameters
    ----------
    kids : list
        List of KIDs for the KROSS sources
    name_map : dict
        Mapping between KROSS source names and KIDs
    '''

    kross_dir = get_base_dir() / 'kl_tools/kross'
    vmap_fit_dir = kross_dir / 'vmap_fits'
    vmap_fit_file = vmap_fit_dir / 'vmap_fits.fits'

    try:
        t = Table.read(vmap_fit_file)
    except FileNotFoundError:
        print(f'Could not find vmap fits file: {vmap_fit_file}')
        print('Skipping KL tools 2D velocity data')
        return None
    
    t_kids = -1 * np.ones(len(t), dtype=int)
    for i, name in enumerate(t['name']):
        name = name.strip()
        if name not in name_map:
            print(f'Could not find kid for name: {name}')
            continue
        t_kids[i] = name_map[name]
    t['kid'] = t_kids

    v2d = np.empty(len(kids))
    for indx, kid in enumerate(kids):
        if kid in t['kid']:
            i = np.where(t['kid'] == kid)[0]
            assert len(i) == 1
            i = i[0]
            v2d[indx] = t['vcirc'][i]
        else:
            v2d[indx] = np.nan

    return v2d

def main():

    #---------------------------------------------------------------------------
    # initial setup

    args = parse_args()
    show = args.show
    estimator = args.estimator
    cosmos_only = args.cosmos_only
    overwrite = args.overwrite
    vb = args.verbose

    analysis_dir = get_base_dir() / 'kl_tools/kross'

    plot_dir = analysis_dir / 'plots/kl_sample'
    make_dir(plot_dir)
    out_dir = analysis_dir / 'sample'
    out_file = out_dir / 'kl_sample.fits'

    if out_file.exists() and not overwrite:
        raise FileExistsError(
            f'{out_file} already exists; use -o or --overwrite to replace'
            )

    kross_dir = get_base_dir() / 'data/kross'
    kross_file = kross_dir / 'kross_release_v2.fits'
    kross = Table.read(kross_file)

    # compute the map between the KROSS source names and kids for matching
    # to the kl-tools measurements
    name_map = {}
    for i, n in enumerate(kross['NAME']):
        name_map[n.strip()] = kross['KID'][i]

    #---------------------------------------------------------------------------
    # make initial sample plots

    # first, look at the distribution of kinematic types
    plt.hist(kross['KIN_TYPE'])
    plt.xlabel('Kinematic Type')
    plt.title('KROSS Distribution of Kinematic Types; All Galaxies')

    plotfile = plot_dir / 'kinematic_types_all.png'
    plot(show, save=True, out_file=plotfile)

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
        # (quality == 1) &
        (agn_flag == 0) &
        (irr_flag == 0) &
        # (theta_flag == 0) &
        (extrap_flag == 0) &
        # (kin_type == 'RT+')
        # TODO: decide if we want the less conservative selection:
        ((kin_type == 'RT ') | (kin_type == 'RT+') | (kin_type == 'DN'))
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
    kid = sample['KID']
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
    # estimated error on 2D vcirc
    vcirc_err = (vc_err_h - vc_err_l) * np.sin(np.deg2rad(theta_im))

    #---------------------------------------------------------------------------
    # estimate vcirc using logmstar and the Tully-Fisher relation
    vtf = estimate_vtf(log_mstar)

    # now make a plot of the distribution of vtf and vcirc
    bins = np.linspace(0, 300, 25)
    plt.hist(vtf, ec='k', bins=bins, label='vtf', alpha=0.5)
    plt.hist(vcirc, ec='k', bins=bins, label='vcirc', alpha=0.5)
    plt.legend()
    plt.xlabel('v (km/s)')

    plotfile = plot_dir / 'vtf_vcirc_compare.png'
    plot(show, save=True, out_file=plotfile)

    # plot the distribution of sini = vcirc / vtf
    sini = vcirc / vtf
    plt.hist(sini, ec='k', bins=20)
    plt.xlabel('sini (vcirc / vtf)')

    plotfile = plot_dir / 'sini_tfr.png'
    plot(show, save=True, out_file=plotfile)

    # plot the distribution of vcirc / sigma
    bins = np.linspace(0, 4, 40)
    plt.hist(vcirc / sigma, bins=bins, ec='k')
    plt.axvline(1, c='k', ls='--')
    plt.xlabel('vcirc / sig_v')

    plotfile = plot_dir / 'vcirc_sigma.png'
    plot(show, save=True, out_file=plotfile)

    # plot the distribution of vc / sigma
    plt.hist(vc / sigma, bins=bins, ec='k')
    plt.axvline(1, c='k', ls='--')
    plt.xlabel('vc / sig_v)')

    plotfile = plot_dir / 'vc_sigma.png'
    plot(show, save=True, out_file=plotfile)

    Nsini = len(sini)
    if vb is True:
        print(f'Sample size with valid sini given v_TF: {Nsini}')

    #---------------------------------------------------------------------------
    # estimate shear from the KROSS quantities, using each requested estimator
    # NOTE: The estimators attempt to compute the sini estimate using two 
    # different sources for the 2D velocity::
    # 1. The inclination-corrected circular velocity from KROSS, vcirc
    # 2. Our own estimate of the 2D velocity from our own fits
    # If the script cannot find the latter, it will compute only the former

    table = Table()
    table['kid'] = kid
    table['ra'] = ra
    table['dec'] = dec
    table['z'] = z
    table['kross_v2d'] = vcirc
    table['kross_v2d_err'] = vcirc_err
    table['vc'] = vc
    table['vc_err_h'] = vc_err_h
    table['vc_err_l'] = vc_err_l
    table['kin_type'] = kin_type
    table['eobs'] = eobs
    table['pa_im'] = pa_im
    table['vtf'] = vtf
    table['log_mstar'] = log_mstar

    # grab our 2D velocity estimates, if available
    our_vcirc = _setup_kl_tools_v2d(kid, name_map)
    if our_vcirc is not None:
        table['our_v2d'] = our_vcirc

    shears = {}
    if estimator == 'point':
        shears['point'] = PointShear()
    elif estimator == 'map':
        shears['map'] = MAPShear()
    elif estimator == 'both':
        shears['point'] = PointShear()
        shears['map'] = MAPShear()
    vel_2d = {}
    vel_2d['kross'] = vcirc
    vel_2d['our'] = our_vcirc 

    xmax = np.nanmax(vel_2d['kross'])
    ymax = np.nanmax(vel_2d['our'])
    vmax = np.nanmax([xmax, ymax])

    if show is True:
        plt.scatter(vel_2d['kross'], vel_2d['our'], s=5)
        plt.plot([0, vmax], [0, vmax], c='k', lw=2)
        plt.xlabel('KROSS 2D Velocity (vcirc)')
        plt.ylabel('KL Tools 2D Velocity')
        plt.title('Comparison of 2D Velocity Estimates')
        plt.savefig(f'{plot_dir}/2d_velocity_comparison.png')
        plt.show()

    # plotting options
    Nbins = 30

    # estimate sini from 2D velocity and TF velocity:
    for vel_type, v2d in vel_2d.items():
        print(f'Estimating shear using {vel_type} 2D velocity...')
        for shear_name, shear in shears.items():
            print(f'Estimating shear using {shear_name} estimator...')

            combo = f'({shear_name}, {vel_type})'

            if shear_name == 'point':
                shear_args = (v2d, vtf)
            elif shear_name == 'map':
                shear_args = (v2d, vtf, vcirc_err)
            sini = shear.estimate_sini(*shear_args)

            plt.hist(sini, bins=Nbins, ec='k')
            plt.xlabel(f'sini ({shear_name}, {vel_type})')
            plt.title(combo)

            plotfile = plot_dir / f'sini_sample_{shear_name}_{vel_type}.png'
            plot(show, save=True, out_file=plotfile)

            # restrict any unphysical sini values
            sini[(sini < -1) | (sini > 1)] = np.nan

            # estimate intrinsic ellipticity from sini and qz (assumed):
            eint = shear.estimate_eint(sini)
            plt.hist(eint, bins=Nbins, ec='k')
            plt.xlabel('eint')
            plt.title(combo)

            plotfile = plot_dir / f'eint_sample_{shear_name}_{vel_type}.png'
            plot(show, save=True, out_file=plotfile)

            # look at difference in eobs vs estimated eint
            edelt = eobs - eint
            edelt2 = eobs**2-eint**2
            plt.hist(edelt, bins=Nbins, ec='k', label='eobs-eint', alpha=0.75)
            plt.hist(edelt2, bins=Nbins, ec='k', label='eobs^2-eint^2', alpha=0.75)
            plt.axvline(0, c='k', ls='--', lw=2)
            plt.xlabel('ellipticity difference')
            plt.legend()
            plt.title(combo)

            plotfile = plot_dir / f'obs_eint_diff_{shear_name}_{vel_type}.png'
            plot(show, save=True, out_file=plotfile)

            # show shear response
            response = (2 * eobs * (1-eint**2))
            plt.hist(response, bins=Nbins, ec='k')
            plt.xlabel('response (2 * eobs**2 * (1-eint**2))')
            plt.title(combo)

            plotfile = plot_dir / f'shear_response_{shear_name}_{vel_type}.png'
            plot(show, save=True, out_file=plotfile)

            # estimate shear values from estimated eint:
            gplus = shear.estimate_gplus(eobs, eint)
            plt.hist(gplus, bins=np.linspace(-1, 1, Nbins), ec='k')
            plt.axvline(0, c='k', ls='--', lw=2)
            plt.xlabel('g+')
            plt.title(combo)
            # plt.xlim(-1, 1)

            plotfile = plot_dir / f'gplus_{shear_name}_{vel_type}.png'
            plot(show, save=True, out_file=plotfile)

            # TODO: Turn on when able
            # gcross = estimate_gcross(vcirc, vmin, eobs, eint, sini)
            # plt.hist(gcross, bins=Nbins, ec='k')
            # plt.xlabel('gx')
            gcross = np.empty(Nsample)
            gcross[:] = np.nan

            # plot s2n
            verr = vc_err_h - vc_err_l
            s2n = v2d / verr
            plt.hist(s2n, ec='k', bins=50)
            plt.xlabel('vcirc / verr')
            plt.title(combo)

            plotfile= plot_dir / f's2n_{shear_name}_{vel_type}.png'
            plot(show, save=True, out_file=plotfile)

            # now save the estimated quantities to the table
            table[f'sini_{shear_name}_{vel_type}'] = sini
            table[f'eint_{shear_name}_{vel_type}'] = eint
            table[f'gplus_{shear_name}_{vel_type}'] = gplus
            table[f'gcross_{shear_name}_{vel_type}'] = gcross

    #---------------------------------------------------------------------------
    # split the sample into the 4 corresponding fields

    field1 = np.where(
        (ra > 32) & (ra < 37)
    )

    field2 = np.where(
        (ra > 49) & (ra < 56)
    )

    # COSMOS
    field3 = np.where(
        (ra > 147) & (ra < 153)
    )

    field4 = np.where(
        (ra > 332) & (ra < 337)
    )

    Nfield1 = len(sample[field1])
    Nfield2 = len(sample[field2])
    Nfield3 = len(sample[field3])
    Nfield4 = len(sample[field4])

    Nfields = Nfield1 + Nfield2 + Nfield3 + Nfield4

    print(f'Nsample: {Nsample}')
    print(f'Nfields: {Nfields}')
    print(f'Nfield1: {Nfield1}')
    print(f'Nfield2: {Nfield2}')
    print(f'Nfield3: {Nfield3}')
    print(f'Nfield4: {Nfield4}')

    assert Nsample == Nfields

    # fields = Nfield1 * ['UDS'] + Nfield2 * ['ECDFS'] +\
        # Nfield3 * ['COSMOS'] + Nfield4 * ['SA22']
    fields = Nsample * ['']
    for i in field1[0]:
        fields[i] = 'UDS'
    for i in field2[0]:
        fields[i] = 'ECDFS'
    for i in field3[0]:
        fields[i] = 'COSMOS'
    for i in field4[0]:
        fields[i] = 'SA22'
    table['field'] = fields

    # restrict to COSMOS-only sources, if requested
    if cosmos_only is True:
        table = table[field3]

    #---------------------------------------------------------------------------
    # match to Eric's selection table

    selection_file = analysis_dir / 'sample/kross_quality_selection.csv'

    try:
        selection_table = Table.read(selection_file)
    except FileNotFoundError:
        print(f'Could not find selection file: {selection_file}')
        print('Skipping matching to Eric\'s selection table')
        selection_table = None

    if selection_table is not None:
        table = join(table, selection_table, keys='kid', join_type='left')

    #---------------------------------------------------------------------------
    # save the table

    table.write(out_file, overwrite=overwrite)

    print(f'Wrote KL sample data to {out_file}')

    return

if __name__ == '__main__':
    main()
