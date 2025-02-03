import numpy as np
import fitsio
from astropy.table import Table
from astropy.wcs import WCS
import matplotlib.pyplot as plt
from pathlib import Path
from glob import glob
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl

from kl_tools.utils import get_base_dir, MidpointNormalize

def plot_vmap(vmap, pa, wcs=None, title=None, show=False, outfile=None):

    norm = MidpointNormalize(
        midpoint=0,
        vmin=np.percentile(vmap, 1),
        vmax=np.percentile(vmap, 99)
        )
    
    plt.figure(figsize=(10, 10))

    if wcs is None:
        projection = None
    else:
        # projection = None
        projection = wcs
    ax = plt.subplot(projection=projection)

    img = ax.imshow(
        vmap,
        origin='lower',
        cmap='RdBu_r',
        norm=norm
        )

    ax.set_xlabel('RA (J2000)')
    ax.set_ylabel('Dec (J2000)')
    ax.grid(color='black', ls='dotted')

    plt.colorbar(img, ax=ax, label='km/s')

    # plt.colorbar(
    #     img, ax=ax, fraction=0.046, pad=0.05,
    #     orientation='vertical', label='km/s'
    #              )
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes('right', size='5%', pad=0.1)
    # cax = divider.append_axes('right', pad=0.1)
    # cbar = plt.colorbar(img, cax=cax)
    # cbar.set_label('km/s')

    #-------------------------
    # draw PA
    # we flip the origin, so do the same
    # pa = -pa
    pa_rad = np.deg2rad(pa)
    height, width = vmap.shape
    center_x, center_y = width / 2, height / 2
    length = max(vmap.shape)

    dx = length / 2 * np.cos(pa_rad)
    dy = length / 2 * np.sin(pa_rad)
    x0, y0 = center_x - dx, center_y - dy
    x1, y1 = center_x + dx, center_y + dy
    ax.plot([x0, x1], [y0, y1], color='red', linestyle='-', linewidth=2, label='VEL_PA')

    if title is not None:
        ax.set_title(title)

    if show is True:
        plt.show()

    if outfile is not None:
        plt.savefig(outfile)

    plt.close()

    return

def main():

    kross_file = get_base_dir() / 'data/kross/kross_release_v2.fits'
    kross = Table.read(kross_file)
    Nkross = len(kross)

    vmap_dir = get_base_dir() / 'data/kross/vmaps/'
    vmap_files = glob(f'{vmap_dir}/*.fits')
    Nvmap = len(vmap_files)

    assert Nvmap > 0
    print(f'Nkross={Nkross}; Nvmap={Nvmap}')

    for vmap_file in vmap_files:
        vmap, hdr = fitsio.read(vmap_file, header=True)
        name = Path(vmap_file).name.split('.fits')[0].strip()
        wcs = WCS(vmap_file)

        # kross names are 22 characters long
        search_name = name.ljust(22)
        obj = kross[kross['NAME'] == search_name]

        if len(obj) == 0:
            print(f'{name} not found in KROSS catalog')
            continue

        obj = obj[0]

        ktype = obj['KIN_TYPE'].strip()
        pa = obj['VEL_PA']

        title = f'{name}; {ktype}'
        plot_vmap(vmap, pa, wcs=wcs, title=title, show=True)

    return

if __name__ == '__main__':
    main()
