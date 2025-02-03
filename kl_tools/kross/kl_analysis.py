from argparse import ArgumentParser
import numpy as np
from astropy.table import Table
import matplotlib.pyplot as plt

from kl_tools.utils import get_base_dir, make_dir, plot

def parse_args():
    parser = ArgumentParser()

    parser.add_argument(
        'kl_sample_file', type=str, help='Input KL table to analyze'
        )
    parser.add_argument(
        '-i', '--sini_max', type=float, default=0.9,
        help='Maximum sin(i) value to consider'
        )
    parser.add_argument(
        '-u', '--unblind', action='store_true',
        help='Unblind the KID of each source considered in the analysis'
    )
    parser.add_argument(
        '-o', '--overwrite', action='store_true', default=False,
        help='Overwrite existing sample files'
    )
    parser.add_argument(
        '-s', '--show', action='store_true',
        help='Show the plots'
        )
    parser.add_argument(
        '-z', '--zoom', action='store_true',
        help='Make zoom-in plots for the g+ comparison'
    )

    return parser.parse_args()

def plot_dist(
        values,
        xlabel,
        plot_outfile=None,
        hist_kwargs={},
        title=None,
        show=False,
        ):

    plt.hist(
        values, **hist_kwargs
        )
    plt.xlabel(xlabel)
    plt.ylabel('Counts')

    if title is not None:
        plt.title(title)

    plot(show, True, out_file=plot_outfile)

    return

def plot_shear_comparison(
    kl_gplus,
    cosmos_gplus,
    # kl_gplus_err, TODO
    cosmos_gplus_err,
    plot_outfile,
    kid=None,
    xlim=None,
    ylim=None,
    title=None,
    show=False,
    size=(8, 8),
    buffer=.1,
    kl_err=0.03 # expected
    ):

    if xlim is None:
        x_low = np.min(
            [np.min(kl_gplus), np.min(cosmos_gplus)]
        )
        x_high = np.max(
            [np.max(kl_gplus), np.max(cosmos_gplus)]
        )
        xlim = (x_low-buffer, x_high+buffer)
    if ylim is None:
        y_low = np.min(
            [np.min(cosmos_gplus), np.min(kl_gplus)]
        )
        y_high = np.max(
            [np.max(cosmos_gplus), np.max(kl_gplus)]
        )
        ylim = (y_low-buffer, y_high+buffer)

    plt.scatter(kl_gplus, cosmos_gplus, alpha=0.75)
    plt.errorbar(
        kl_gplus, cosmos_gplus, yerr=cosmos_gplus_err, ls=''
        )
    plt.fill_between(
        [xlim[0], xlim[1]],
        [xlim[0]-kl_err, xlim[1]-kl_err],
        [xlim[0]+kl_err, xlim[1]+kl_err],
        color='gray',
        alpha=0.5
    )
    plt.axhline(0, c='k', ls=':', lw=2)
    plt.axvline(0, c='k', ls=':', lw=2)
    plt.plot(
        # [np.min(kl_gplus), np.max(kl_gplus)],
        # [np.min(kl_cosmos_gplus), np.max(kl_cosmos_gplus)],
        xlim,
        xlim,
        c='k',
        ls='--',
        lw=2
        )

    # Add the KID labels
    if kid is not None:
        offset = 0.01
        for i, (x, y) in enumerate(zip(kl_gplus, cosmos_gplus)):
            plt.text(
                x, y+offset, str(kid[i]), fontsize=8, ha='right', va='bottom', alpha=0.75
                )

    # plt.xlim(*xlim)
    # plt.ylim(*ylim)
    plt.xlabel('KL g+')
    plt.ylabel('COSMOS g+ (projected)')
    if title is not None:
        plt.title(title)
    plt.gcf().set_size_inches(size)

    plot(show, True, out_file=plot_outfile)

    return

def main():

    args = parse_args()
    kl_sample_file = args.kl_sample_file
    sini_max = args.sini_max
    unblind = args.unblind
    overwrite = args.overwrite
    show = args.show
    zoom = args.zoom

    kross_dir = get_base_dir() / 'kl_tools/kross'
    plot_dir = kross_dir / 'plots/analysis'
    sample_dir = kross_dir / 'sample'
    out_dir = sample_dir / 'analysis'
    make_dir(plot_dir)
    make_dir(out_dir)

    #---------------------------------------------------------------------------
    # define all selections

    estimator_sel = ['point', 'map']
    vmap_sel = ['kross', 'our']
    # vmap_sel = ['our']
    # sini_sel = [1.0, sini_max]
    sini_sel = [1.0]
    quality_sel = [3, 4]
    # allowed_kin_types = ['RT+']

    #---------------------------------------------------------------------------
    # load the KL sample

    kl = Table.read(kl_sample_file)
    Ntotal = len(kl)
    print(f'Loaded {Ntotal} galaxies from {kl_sample_file}')

    #---------------------------------------------------------------------------
    # loop over all selections

    for vmap_type in vmap_sel:
        for est in estimator_sel:
            for max_sini in sini_sel:
                for quality in quality_sel:

                    # select the sample
                    selection = np.where(
                        (kl[f'sini_{est}_{vmap_type}'] <= max_sini) &
                        (kl['vmap_quality'] >= quality) &
                        # TODO: figure out
                        # (kl['kin_type'] in allowed_kin_types) &
                        # (kl['kin_type'] == 'RT+') &
                        (kl['field'] == 'COSMOS')
                    )
                    sample = kl[selection]
                    Nsample = len(sample)

                    # only print the KIDs if we're unblinding
                    if unblind:
                        kids = sample['kid']
                    else:
                        kids = None

                    # grab some useful cols
                    sini = sample[f'sini_{est}_{vmap_type}']
                    eint = sample[f'eint_{est}_{vmap_type}']
                    gplus = sample[f'gplus_{est}_{vmap_type}']
                    gcross = sample[f'gcross_{est}_{vmap_type}']
                    cosmos_gplus = sample['cosmos_gplus']
                    cosmos_gplus_err = sample['cosmos_gplus_err']

                    # print some info
                    sample_info = f'Estimator: {est}; Vmap: {vmap_type}; sin(i) < {max_sini}; quality >= {quality}; {Nsample} galaxies'
                    print(sample_info)

                    combo = f'{est}_{vmap_type}'

                    # plot the sin(i) distribution
                    hist_kwargs = {'bins': 20, 'histtype': 'step'}
                    plot_dist(
                        sini,
                        'sini',
                        title=sample_info,
                        hist_kwargs=hist_kwargs,
                        plot_outfile = plot_dir / f'dist_sini_{combo}_{max_sini}_{quality}.png',
                        )

                    # plot the intrinsic ellipticity distribution
                    hist_kwargs = {'bins': 20, 'histtype': 'step'}
                    plot_dist(
                        eint,
                        'eint',
                        title=sample_info,
                        hist_kwargs=hist_kwargs,
                        plot_outfile = plot_dir / f'dist_eint_{combo}_{max_sini}_{quality}.png',
                        )

                    # plot the gplus distribution
                    hist_kwargs = {'bins': 20, 'histtype': 'step'}
                    plot_dist(
                        gplus,
                        'gplus',
                        title=sample_info,
                        hist_kwargs=hist_kwargs,
                        plot_outfile = plot_dir / f'dist_gplus_{combo}_{max_sini}_{quality}.png',
                        )

                    # plot the full gplus comparison
                    # if (vmap_type == 'our') & (est == 'point'):
                    #     import ipdb; ipdb.set_trace()
                    plot_shear_comparison(
                        gplus,
                        cosmos_gplus,
                        cosmos_gplus_err,
                        plot_outfile = plot_dir / f'gplus_comparison_{combo}_{max_sini}_{quality}.png',
                        title=sample_info,
                        kid=kids,
                        show=show
                        )

                    if zoom is True:
                        # plot the zoomed-in gplus comparison
                        plot_shear_comparison(
                            gplus,
                            cosmos_gplus,
                            cosmos_gplus_err,
                            plot_outfile = plot_dir / f'gplus_comparison_{combo}_{max_sini}_{quality}_zoomed.png',
                            title=f'{sample_info}; zoomed',
                            xlim=(-0.25, 0.25),
                            ylim=(-0.25, 0.25),
                            unblind=unblind,
                            kid=kids,
                            show=show
                            )

                    # save the sample to disk
                    sample.write(
                        out_dir / f'sample_{combo}_{max_sini}_{quality}.fits',
                        overwrite=overwrite
                        )

    return

if __name__ == '__main__':
    main()
