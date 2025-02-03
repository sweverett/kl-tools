'''
This script will take the KL sample and add in the COSMOS shear estimates
created by making a COSMOS shear map from scratch, using specified spatial
bin sizes and redshift cuts. The COSMOS shear estimates will be added to the
KL sample as new columns.
'''

from argparse import ArgumentParser
from pathlib import Path
import numpy as np
from astropy.table import Table
import matplotlib.pyplot as plt

from kl_tools.utils import get_base_dir, MidpointNormalize

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        'kl_sample_file', type=str,
        help='The path of the KL sample file to add COSMOS shear estimates to'
    )
    parser.add_argument(
        '-o', '--out_file', type=str, default=None,
        help='The path to save the output file to (default: save to a '
        'modified version of the input file)'
    )
    parser.add_argument(
        '-c', '--overwrite', action='store_true',
        help='Overwrite the input file with the output file'
    )
    parser.add_argument(
        '-p', '--plot', action='store_true',
        help='Plot the shear comparison'
    )

    return parser.parse_args()

def transform_shears_with_errors(g1, g2, sigma_g1, sigma_g2, theta):
    '''
    g1, g2: float
        The input shear components
    sigma_g1, sigma_g2: float
        The uncertainties on the input shear components
    theta: float
        The angle of the shear in radians
    '''

    cos2 = np.cos(2. * theta)
    sin2 = np.sin(2. * theta)

    g1_prime =  g1 * cos2 + g2 * sin2
    g2_prime = -g1 * sin2 + g2 * cos2

    sigma_g1_prime = np.sqrt((cos2**2 * sigma_g1**2) + (sin2**2 * sigma_g2**2))
    sigma_g2_prime = np.sqrt((sin2**2 * sigma_g1**2) + (cos2**2 * sigma_g2**2))

    return g1_prime, g2_prime, sigma_g1_prime, sigma_g2_prime

def main() -> None:

    args = parse_args()
    kl_sample_file = args.kl_sample_file
    out_file = args.out_file
    overwrite = args.overwrite
    plot = args.plot

    if out_file is None:
        out_dir = Path(kl_sample_file).parent
        out_file = out_dir / Path(kl_sample_file).name.replace(
            '.fits', '_with_cosmos.fits'
            )
    else:
        out_file = Path(out_file)

    if out_file.exists() and (not overwrite):
        raise ValueError(
            f'Output file {out_file} already exists. Use --overwrite to '
            'overwrite the file.'
            )

    #---------------------------------------------------------------------------
    # useful params for the map making

    # TODO: These should be passed as arguments
    buffer = 1.5 / 60. # ra / dec buffer size in degrees
    bin_size_ra = 1.5 / 60. # deg
    bin_size_dec = 1.5 / 60. # deg

    #---------------------------------------------------------------------------
    # Load the COSMOS shear catalog
    
    data_dir = get_base_dir() / 'data'
    cosmos_file = data_dir / 'cosmos/cosmosACS_sourcecatalog_2018.fits'
    cosmos = Table.read(cosmos_file)

    #---------------------------------------------------------------------------
    # Make any necessary cuts (e.g. for redshift matching)

    zmin = 0.8
    zmax = 1.1

    cosmos = cosmos[
        (cosmos['Zphot'] > zmin) & (cosmos['Zphot'] < zmax)
    ]

    #---------------------------------------------------------------------------
    # Make the ra and dec bins

    ra  = cosmos['RA']
    dec = cosmos['DEC']
    gamma1 = cosmos['gamma1']
    gamma2 = cosmos['gamma2']
    weights = cosmos['weight']

    min_ra  = np.min(ra) - buffer
    max_ra  = np.max(ra) + buffer

    min_dec = np.min(dec) - buffer
    max_dec = np.max(dec) + buffer
        
    delta_ra  = max_ra - min_ra
    delta_dec = max_dec - min_dec

    # Calculate the number of bins
    Nbins_ra  = int(delta_ra  / bin_size_ra )
    Nbins_dec = int(delta_dec / bin_size_dec)

    # Create the bins
    ra_bins  = np.linspace(min_ra,  max_ra,  Nbins_ra  + 1)
    dec_bins = np.linspace(min_dec, max_dec, Nbins_dec + 1)

    # Assign galaxies to bins
    ra_bin_indices  = np.digitize(cosmos['RA'],  ra_bins ) - 1
    dec_bin_indices = np.digitize(cosmos['DEC'], dec_bins) - 1

    cosmos['ra_bin']  = ra_bin_indices
    cosmos['dec_bin'] = dec_bin_indices

    #---------------------------------------------------------------------------
    # Make the shear maps

    # Combine the bin indices into a single index
    combined_indices = ra_bin_indices * (Nbins_dec) + dec_bin_indices

    # Compute the counts for each combined index
    counts_1d = np.bincount(
        combined_indices,
        minlength=(Nbins_ra * Nbins_dec),
        )

    # Compute the g1/g2 shear components each combined index
    g1_wsum_1d = np.bincount(
        combined_indices,
        minlength=(Nbins_ra * Nbins_dec),
        weights=gamma1*weights
        )
    g2_wsum_1d = np.bincount(
        combined_indices,
        minlength=(Nbins_ra * Nbins_dec),
        weights=gamma2*weights
        )

    #---------------------------------------------------------------------------
    # The above weighted average now needs to be normalized by the total weight in each bin

    # Compute the sum of the weights for each combined index
    wsum_1d = np.bincount(
        combined_indices, minlength=(Nbins_ra * Nbins_dec), weights=weights
    )

    #---------------------------------------------------------------------------
    # Reshape the result back into a 2D map

    # NOTE: Need to track down why the transpose is necessary
    g1_wsum_2d = g1_wsum_1d.reshape(Nbins_ra, Nbins_dec).T
    g2_wsum_2d = g2_wsum_1d.reshape(Nbins_ra, Nbins_dec).T
    wsum_2d = wsum_1d.reshape(Nbins_ra, Nbins_dec).T
    counts_2d = counts_1d.reshape(Nbins_ra, Nbins_dec).T

    # Compute the weighted average for each bin
    g1_wsum_2d[g1_wsum_2d == 0] = np.nan
    g2_wsum_2d[g2_wsum_2d == 0] = np.nan
    wsum_2d[wsum_2d == 0] = np.nan

    g1_2d = g1_wsum_2d / wsum_2d
    g2_2d = g2_wsum_2d / wsum_2d

    #---------------------------------------------------------------------------
    # now do the same procedure, but for the variance

    # compute the weighted variance for each combined index
    g1_diff_squared_wsum_1d = np.bincount(
        combined_indices,
        minlength=(Nbins_ra * Nbins_dec),
        weights=((gamma1 - g1_2d.flatten()[combined_indices])**2) * weights
    )
    g2_diff_squared_wsum_1d = np.bincount(
        combined_indices,
        minlength=(Nbins_ra * Nbins_dec),
        weights=((gamma2 - g2_2d.flatten()[combined_indices])**2) * weights
    )

    # Reshape to 2D
    g1_diff_squared_wsum_2d = g1_diff_squared_wsum_1d.reshape(
        Nbins_ra, Nbins_dec
        ).T
    g2_diff_squared_wsum_2d = g2_diff_squared_wsum_1d.reshape(
        Nbins_ra, Nbins_dec
        ).T

    # Compute the weighted variance
    g1_variance_2d = g1_diff_squared_wsum_2d / (wsum_2d - 1)
    g2_variance_2d = g2_diff_squared_wsum_2d / (wsum_2d - 1)

    # Compute the standard error of the mean
    g1_std_error_2d = np.sqrt(g1_variance_2d) / np.sqrt(counts_2d)
    g2_std_error_2d = np.sqrt(g2_variance_2d) / np.sqrt(counts_2d)

    # Set NaNs where weights or counts are zero
    g1_std_error_2d[np.isnan(wsum_2d)] = np.nan
    g2_std_error_2d[np.isnan(wsum_2d)] = np.nan

    #---------------------------------------------------------------------------
    # Lookup the KL galaxies and assign a cosmos shear

    kl = Table.read(kl_sample_file)

    # new columns we will add
    kl['cosmos_g1'] = np.nan
    kl['cosmos_g2'] = np.nan
    kl['cosmos_g1_err'] = np.nan
    kl['cosmos_g2_err'] = np.nan
    kl['cosmos_gplus'] = np.nan
    kl['cosmos_gcross'] = np.nan
    kl['cosmos_gplus_err'] = np.nan
    kl['cosmos_gcross_err'] = np.nan

    kl_cosmos_indices = np.where(
        kl['field'] == 'COSMOS'
    )

    kl_ra  = kl['ra'][kl_cosmos_indices]
    kl_dec = kl['dec'][kl_cosmos_indices]
    # position angle of the photometric axis, in deg
    kl_pa_im = kl['pa_im'][kl_cosmos_indices]

    kl_ra_bin_indices   = np.digitize(kl_ra,  ra_bins ) - 1
    kl_dec_bin_indices  = np.digitize(kl_dec, dec_bins) - 1

    kl_cosmos_g1 = g1_2d[kl_ra_bin_indices, kl_dec_bin_indices]
    kl_cosmos_g2 = g2_2d[kl_ra_bin_indices, kl_dec_bin_indices]

    kl_cosmos_g1_err = g1_std_error_2d[kl_ra_bin_indices, kl_dec_bin_indices]
    kl_cosmos_g2_err = g2_std_error_2d[kl_ra_bin_indices, kl_dec_bin_indices]

    kl['cosmos_g1'][kl_cosmos_indices] = kl_cosmos_g1
    kl['cosmos_g2'][kl_cosmos_indices] = kl_cosmos_g2
    kl['cosmos_g1_err'][kl_cosmos_indices] = kl_cosmos_g1_err
    kl['cosmos_g2_err'][kl_cosmos_indices] = kl_cosmos_g2_err

    # Add in the COSMOS shear estimates in g+/gx coords for each KL galaxy
    transformed_shears = transform_shears_with_errors(
        kl_cosmos_g1,
        kl_cosmos_g2,
        kl_cosmos_g1_err,
        kl_cosmos_g2_err,
        np.deg2rad(kl_pa_im)
        )

    kl_cosmos_gplus = transformed_shears[0]
    kl_cosmos_gcross = transformed_shears[1]
    kl_cosmos_gplus_err = transformed_shears[2]
    kl_cosmos_gcross_err = transformed_shears[3]

    kl['cosmos_gplus'][kl_cosmos_indices] = kl_cosmos_gplus
    kl['cosmos_gcross'][kl_cosmos_indices] = kl_cosmos_gcross
    kl['cosmos_gplus_err'][kl_cosmos_indices] = kl_cosmos_gplus_err
    kl['cosmos_gcross_err'][kl_cosmos_indices] = kl_cosmos_gcross_err

    kl.write(out_file, overwrite=overwrite)

    #---------------------------------------------------------------------------
    # Plot the shear comparison

    if plot is True:
        # estimate from forecasts
        klerr = 0.03

        shear_names = ['point', 'map']
        for name in shear_names:
            try:
                kl_gplus = kl[f'gplus_{name}'][kl_cosmos_indices]
                plt.scatter(kl_gplus, kl_cosmos_gplus)
                plt.errorbar(
                    kl_gplus, kl_cosmos_gplus, yerr=kl_cosmos_gplus_err, ls=''
                    )
                plt.fill_between(
                    [-1, 1],
                    [-1-klerr, 1-klerr],
                    [-1+klerr, 1+klerr],
                    color='gray',
                    alpha=0.5
                )
                plt.plot(
                    # [np.min(kl_gplus), np.max(kl_gplus)],
                    # [np.min(kl_cosmos_gplus), np.max(kl_cosmos_gplus)],
                    [-1, 1],
                    [-1, 1],
                    c='k',
                    ls='--',
                    lw=2
                    )
                plt.xlim(-1, 1)
                plt.ylim(-1, 1)
                plt.xlabel('KL g+')
                plt.ylabel('COSMOS g+ (projected)')
                plt.title('Shear comparison')
                plt.gcf().set_size_inches(9,9)

                plt.savefig('shear_comparison.png', dpi=300)
                plt.show()

                plt.scatter(kl_gplus, kl_cosmos_gplus)
                plt.errorbar(
                    kl_gplus, kl_cosmos_gplus, yerr=kl_cosmos_gplus_err, ls=''
                    )
                plt.fill_between(
                    [-1, 1],
                    [-1-klerr, 1-klerr],
                    [-1+klerr, 1+klerr],
                    color='gray',
                    alpha=0.5
                )
                plt.plot(
                    # [np.min(kl_gplus), np.max(kl_gplus)],
                    # [np.min(kl_cosmos_gplus), np.max(kl_cosmos_gplus)],
                    [-1, 1],
                    [-1, 1],
                    c='k',
                    ls='--',
                    lw=2
                    )
                plt.xlim(-0.25, 0.25)
                plt.ylim(-0.25, 0.25)
                plt.xlabel('KL g+')
                plt.ylabel('COSMOS g+ (projected)')
                plt.title('Shear comparison')
                plt.gcf().set_size_inches(9,9)
                plt.savefig('shear_comparison-zoom.png', dpi=300)
                plt.show()

                #---------------------------------------------------------------
                # Plot the shear comparision residuals

                resid = kl_gplus - kl_cosmos_gplus

                plt.hist(resid, bins=50, ec='k')
                plt.xlabel('KL g+ - COSMOS g1')
                plt.gcf().set_size_inches(12,6)
                plt.show()

                plt.hist(resid[resid>-1], bins=30, ec='k')
                plt.xlabel('KL g+ - COSMOS g1')
                plt.gcf().set_size_inches(12,6)
                plt.show()

            except KeyError:
                print(f'No {name} shear estimates in KL sample; skipping')

        #---------------------------------------------------------------
        # Plot the shear histograms

        plt.hist(
            g1_2d.flatten(), bins=100, ec='k', alpha=0.6, label='g1'
            )
        plt.hist(
            g2_2d.flatten(), bins=100, ec='k', alpha=0.6, label='g2'
            )
        plt.legend()
        # plt.yscale('log')
        plt.xlabel('shear')
        plt.gcf().set_size_inches(8, 6)
        plt.show()

        #---------------------------------------------------------------
        # Plot the shear component (and count) maps

        plt.subplot(131)
        norm = MidpointNormalize(
            midpoint=0, vmin=np.min(g1_2d), vmax=np.max(g1_2d)
            )
        plt.imshow(
            g1_2d, origin='lower', aspect='auto', norm=norm, cmap='RdBu'
            )
        plt.colorbar()
        plt.title('g1')
        plt.subplot(132)
        norm = MidpointNormalize(
            midpoint=0, vmin=np.min(g2_2d), vmax=np.max(g2_2d)
            )
        plt.imshow(
            g2_2d, origin='lower', aspect='auto', norm=norm, cmap='RdBu'
            )
        plt.colorbar()
        plt.title('g2')
        plt.subplot(133)
        norm = MidpointNormalize(
            midpoint=0, vmin=np.min(counts_2d), vmax=np.max(counts_2d)
            )
        plt.imshow(
            counts_2d, origin='lower', aspect='auto', norm=norm, cmap='RdBu'
            )
        plt.colorbar()
        plt.title('counts')

        plt.gcf().set_size_inches(16, 4)
        plt.show()

    return

if __name__ == '__main__':
    main()