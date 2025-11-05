import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_line_on_image(im, cen, angle, ax, c='k', ls='--', label=None):
    """
    Plots a line on an image at a given angle using WCS projection.
    
    Parameters:
    im : Image data
    cen: Center of the image in pixel coordinates (x0, y0)
    angle : OrientedAngle of PA
    ax : Matplotlib Axes object
    c: color
    """

    # Get the dimensions of the image
    Nrow, Ncol = im.shape

    # gal center
    xgal, ygal = cen

    xc = Ncol / 2. + xgal
    yc = Nrow / 2. + ygal
    
    # Calculate the angle in radians
    angle_rad = angle.cartesian.rad

    if angle_rad > np.pi/2:
        angle_rad -= np.pi

    x0, x1 = 0, Ncol - 1

    y0 = np.tan(angle_rad) * (x0 - xc) + yc
    y1 = np.tan(angle_rad) * (x1 - xc) + yc

    # Create the line coordinates in pixel space
    line_x = np.array([x0, x1])
    line_y = np.array([y0, y1])
    
    ax.plot(
        line_x,
        line_y,
        color=c,
        linewidth=2,
        ls=ls,
        label=label
        )
    plt.xlim(0, Ncol)
    plt.ylim(0, Nrow)

    return

def plot(show, out_file=None, overwrite=False, dpi=300):
    '''
    Helper function to streamline plotting options

    Parameters
    ----------
    show : bool
        Whether to display the plot.
    overwrite : bool
        Whether to overwrite existing files.
        Default is False.
    out_file : str
        The output file to save the plot to, if desired.
        Default is None : the plot will not be saved.
    dpi : int
        The resolution of the plot.
        Default is 300.
    '''

    if out_file is not None:
        if overwrite is False and Path(out_file).exists():
            raise FileExistsError(
                f'{out_file} already exists and overwrite is set to False.'
            )
        plt.savefig(out_file, dpi=dpi)
    if show is True:
        plt.show()
    else:
        plt.close()
