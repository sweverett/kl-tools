import numpy as np
from astropy.table import Table
import matplotlib.pyplot as plt

from kl_tools.utils import get_base_dir, MidpointNormalize

def transform_shears(g1, g2, theta):
    '''
    g1, g2: float
        The input shear components
    theta: float
        The angle of the shear in radians
    '''

    g1_prime =  g1 * np.cos(2. * theta) + g2 * np.sin(2. * theta)
    g2_prime = -g1 * np.sin(2. * theta) + g2 * np.cos(2. * theta)

    return g1_prime, g2_prime

def main() -> None:

    #---------------------------------------------------------------------------
    # useful params for the map making

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
    # Lookup the KL galaxies and assign a cosmos shear

    kl_file = get_base_dir() / 'notebooks/kross/shear_estimates.fits'
    kl = Table.read(kl_file)

    kl_ra  = kl['ra']
    kl_dec = kl['dec']
    kl_gplus = kl['gplus']
    # kl_g2 = kl['gcross']
    kl_pa_im = kl['pa_im'] # position angle of the photometric axis, in deg

    kl_ra_bin_indices   = np.digitize(kl_ra,  ra_bins ) - 1
    kl_dec_bin_indices  = np.digitize(kl_dec, dec_bins) - 1

    kl_cosmos_g1 = g1_2d[kl_ra_bin_indices, kl_dec_bin_indices]
    kl_cosmos_g2 = g2_2d[kl_ra_bin_indices, kl_dec_bin_indices]

    kl_cosmos_g1_err = kl_cosmos_g1 / np.sqrt(counts_2d[kl_ra_bin_indices, kl_dec_bin_indices])

    kl['cosmos_g1'] = kl_cosmos_g1
    kl['cosmos_g2'] = kl_cosmos_g2

    # Add in the COSMOS shear estimates in g+/gx coords for each KL galaxy
    kl_cosmos_gplus, kl_cosmos_gcross = transform_shears(
        kl_cosmos_g1, kl_cosmos_g2, np.deg2rad(kl_pa_im)
        )

    kl_cosmos_gplus_err = kl_cosmos_gplus / np.sqrt(counts_2d[kl_ra_bin_indices, kl_dec_bin_indices])
    kl_cosmos_gcross_err = kl_cosmos_gcross / np.sqrt(counts_2d[kl_ra_bin_indices, kl_dec_bin_indices])

    kl['cosmos_gplus'] = kl_cosmos_gplus
    kl['cosmos_gcross'] = kl_cosmos_gcross
    kl['cosmos_gplus_err'] = kl_cosmos_gplus_err
    kl['cosmos_gcross_err'] = kl_cosmos_gcross_err

    kl.write(kl_file, overwrite=True)

    #---------------------------------------------------------------------------
    # Plot the shear comparison

    # estimate from forecasts
    klerr = 0.03

    plt.scatter(kl_gplus, kl_cosmos_gplus)
    plt.errorbar(kl_gplus, kl_cosmos_gplus, yerr=kl_cosmos_gplus_err, ls='')
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
    plt.errorbar(kl_gplus, kl_cosmos_gplus, yerr=kl_cosmos_gplus_err, ls='')
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

    #---------------------------------------------------------------------------
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

    #---------------------------------------------------------------------------
    # Plot the shear histograms

    plt.hist(g1_2d.flatten(), bins=100, ec='k', alpha=0.6, label='g1')
    plt.hist(g2_2d.flatten(), bins=100, ec='k', alpha=0.6, label='g2')
    plt.legend()
    # plt.yscale('log')
    plt.xlabel('shear')
    plt.gcf().set_size_inches(8, 6)
    plt.show()

    #---------------------------------------------------------------------------
    # Plot the shear component (and count) maps

    plt.subplot(131)
    norm = MidpointNormalize(
        midpoint=0, vmin=np.min(g1_2d), vmax=np.max(g1_2d)
        )
    plt.imshow(g1_2d, origin='lower', aspect='auto', norm=norm, cmap='RdBu')
    plt.colorbar()
    plt.title('g1')
    plt.subplot(132)
    norm = MidpointNormalize(
        midpoint=0, vmin=np.min(g2_2d), vmax=np.max(g2_2d)
        )
    plt.imshow(g2_2d, origin='lower', aspect='auto', norm=norm, cmap='RdBu')
    plt.colorbar()
    plt.title('g2')
    plt.subplot(133)
    norm = MidpointNormalize(
        midpoint=0, vmin=np.min(counts_2d), vmax=np.max(counts_2d)
        )
    plt.imshow(counts_2d, origin='lower', aspect='auto', norm=norm, cmap='RdBu')
    plt.colorbar()
    plt.title('counts')

    plt.gcf().set_size_inches(16, 4)
    plt.show()

    return

if __name__ == '__main__':
    main()