import numpy as np
import matplotlib.pyplot as plt

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