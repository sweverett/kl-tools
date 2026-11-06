import numpy as np
import galsim
from galsim.angle import Angle, radians
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.optimize import minimize

# Assuming the user has the kl_tools modules in their python path
# and the intensity.py file is accessible.
# If not, these imports would need to be adjusted.
from kl_tools.intensity import build_intensity_map, InclinedExponential
from kl_tools.parameters import ImagePars

# --- Global variables to avoid re-calculating static data ---
# This is a common practice when using optimizers to speed things up.
TRUE_GALAXY_IMAGE = None
IMAGE_PARS = None
TRUE_THETA_PARS = None

def chi2_to_minimize(params):
    '''
    This is the objective function for the optimizer. It takes a list
    of parameters [sini, beta], performs a basis fit, and returns the
    reduced chi-squared.

    Args:
        params (list or np.ndarray): A list containing [sini, beta].

    Returns:
        float: The reduced chi-squared of the fit.
    '''
    sini, beta = params
    
    # --- Define Basis and Transformation Parameters ---
    # The optimizer changes sini and beta on each call.
    fit_theta_pars = TRUE_THETA_PARS.copy()
    fit_theta_pars['sini'] = sini

    basis_pars = {
        'basis_type': 'exp_shapelets',
        'basis_plane': 'disk',
        'skip_ground_state': False,
        'basis_kwargs': {
            'nmax': 10,
            'beta': beta,  # This is being optimized
            'psf': None
        }
    }

    # --- Perform Fit ---
    basis_fitter_imap = build_intensity_map('basis', basis_pars)
    fit_image = basis_fitter_imap.render(
        IMAGE_PARS,
        fit_theta_pars,
        pars=None,
        image=TRUE_GALAXY_IMAGE,
        mask=None,
    )

    # --- Calculate Chi-Squared ---
    residuals = TRUE_GALAXY_IMAGE - fit_image
    chi2 = np.sum(residuals**2)
    reduced_chi2 = chi2 / (IMAGE_PARS.Nx * IMAGE_PARS.Ny)
    
    print(f"Trying sini={sini:.3f}, beta={beta:.3f} -> Reduced Chi^2 = {reduced_chi2:.4f}")

    return reduced_chi2

def main():
    '''
    This script uses a numerical optimizer to find the best-fit values
    of sini and the basis scale 'beta' for a given galaxy model.
    '''
    print("--- Starting Basis Fit Optimization ---")

    # Use global variables to share data with the objective function
    global TRUE_GALAXY_IMAGE, IMAGE_PARS, TRUE_THETA_PARS

    # --- Basic Simulation Parameters ---
    nx, ny = 64, 64
    pixel_scale = 0.05
    IMAGE_PARS = ImagePars((nx, ny), pixel_scale=pixel_scale)

    # --- Step 1: Create the "True" Galaxy Image ---
    true_flux = 1000.
    true_hlr = 0.5
    true_sini = 0.8
    true_beta = 0.6 # The true beta is not explicitly set, but we can guess a value

    TRUE_THETA_PARS = {
        'g1': 0.02, 'g2': -0.01, 'theta_int': np.pi / 4,
        'sini': true_sini, 'x0': 0., 'y0': 0.,
    }

    true_galaxy_imap = InclinedExponential(flux=true_flux, hlr=true_hlr)
    print(f"Generating true galaxy image with sini={true_sini}, hlr={true_hlr}")
    TRUE_GALAXY_IMAGE = true_galaxy_imap.render(
        IMAGE_PARS, TRUE_THETA_PARS, pars=None
    )

    # --- Step 2: Optimize for sini and beta ---
    # Initial guess for the parameters [sini, beta]
    initial_guess = [0.5, 0.5]
    
    # Bounds for the parameters: 0 <= sini <= 1, and beta > 0
    bounds = [(0.0, 1.0), (0.1, 2.0)]

    print("\n--- Starting scipy.optimize.minimize ---")
    result = minimize(
        chi2_to_minimize,
        initial_guess,
        method='L-BFGS-B', # A good method for bounded problems
        bounds=bounds,
        options={'disp': True}
    )

    best_sini, best_beta = result.x

    # --- Step 3: Print and Plot Results ---
    print("\n--- Optimization Results ---")
    print(f"True sini: {true_sini:.3f}")
    print(f"Best-fit sini: {best_sini:.3f}")
    print(f"\nAssumed true beta (for comparison): {true_beta:.3f}")
    print(f"Best-fit beta: {best_beta:.3f}")
    print(f"\nMinimum reduced chi-squared: {result.fun:.4f}")

    # --- Generate final comparison plot with the best-fit model ---
    # Re-run the fit with the best parameters to get the final image
    final_theta_pars = TRUE_THETA_PARS.copy()
    final_theta_pars['sini'] = best_sini
    
    final_basis_pars = {
        'basis_type': 'exp_shapelets', 'basis_plane': 'disk', 'skip_ground_state': False,
        'basis_kwargs': {'nmax': 10, 'beta': best_beta, 'psf': None}
    }
    
    final_fitter_imap = build_intensity_map('basis', final_basis_pars)
    final_fit_image = final_fitter_imap.render(
        IMAGE_PARS, final_theta_pars, pars=None, image=TRUE_GALAXY_IMAGE, mask=None
    )

    plot_fit_comparison(
        TRUE_GALAXY_IMAGE, final_fit_image, hlr=true_hlr,
        true_sini=true_sini, best_sini=best_sini
    )


def plot_fit_comparison(true_image, best_fit_image, hlr, true_sini, best_sini):
    '''
    Generates a plot comparing the true image to the best-fit model.
    '''
    residuals = true_image - best_fit_image
    vmax = np.percentile(true_image, 99.5)
    vmin = np.percentile(true_image, 0.5)
    res_vmax = np.percentile(np.abs(residuals), 99.5)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    title1 = f'Original Model (hlr={hlr}, sini={true_sini:.2f})'
    title2 = f'Optimized Best Fit (sini={best_sini:.2f})'
    title3 = 'Residuals'

    images = [true_image, best_fit_image, residuals]
    titles = [title1, title2, title3]
    vmins = [vmin, vmin, -res_vmax]
    vmaxs = [vmax, vmax, res_vmax]
    cmaps = ['viridis', 'viridis', 'RdBu_r']

    for i, ax in enumerate(axes):
        im = ax.imshow(images[i], origin='lower', cmap=cmaps[i],
                       vmin=vmins[i], vmax=vmaxs[i])
        ax.set_title(titles[i])
        ax.set_xticks([])
        ax.set_yticks([])
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax)

    plt.suptitle('Optimized Basis Fit Comparison', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.show()


if __name__ == '__main__':
    main()
