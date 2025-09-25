import numpy as np
import galsim
from galsim.angle import Angle, radians
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


# Assuming the user has the kl_tools modules in their python path
# and the intensity.py file is accessible.
# If not, these imports would need to be adjusted.
from kl_tools.intensity import build_intensity_map, InclinedExponential
from kl_tools.parameters import ImagePars


def main():
    '''
    This script tests the performance of the basis fitting algorithm by:
    1. Creating a known galaxy image (an inclined exponential profile).
    2. Looping over different assumed inclination angles and fitting the
       image with basis functions for each.
    3. Plotting the coefficient power spectrum and chi-squared for each
       fit to compare basis compactness and goodness-of-fit.
    '''
    print("--- Starting Basis Fit Performance Test ---")

    # --- Basic Simulation Parameters ---
    nx, ny = 64, 64  # Image dimensions in pixels
    pixel_scale = 0.05  # arcsec / pixel
    image_pars = ImagePars((nx, ny), pixel_scale=pixel_scale)

    # --- Step 1: Create the "True" Galaxy Image ---
    # Define the true physical and transformation parameters for the galaxy
    true_flux = 1000.
    true_hlr = 0.5  # half-light radius in arcsec
    true_sini = 0.6  # True sine of the inclination angle (highly inclined)

    true_theta_pars = {
        'g1': 0.02,
        'g2': -0.01,
        'theta_int': np.pi / 4,  # 45-degree rotation
        'sini': true_sini,
        'x0': 0.,  # center x-offset in arcsec
        'y0': 0.,  # center y-offset in arcsec
    }

    # Build the inclined exponential intensity map object
    true_galaxy_imap = InclinedExponential(flux=true_flux, hlr=true_hlr)

    # Render the true galaxy image
    print(f"Generating true galaxy image with sini = {true_sini}")
    true_galaxy_image = true_galaxy_imap.render(
        image_pars,
        true_theta_pars,
        pars=None, # No extra parameters needed for this simple model
    )

    # --- Step 2: Loop over assumed inclinations and fit ---
    # Define the parameters for the basis function fit
    basis_pars = {
        'basis_type': 'exp_shapelets',
        'basis_plane': 'disk',
        'skip_ground_state': False,  # do not skip the ground state basis function
        'basis_kwargs': {
            'nmax': 25,       # Maximum order of the basis functions
            'beta': 0.306,      # Scale radius for the basis
            'psf': None
        }
    }

    # Define the assumed sini values to loop over
    assumed_sinis = [0.0, 0.2, 0.4, 0.6, 0.8]
    
    fit_results = []
    
    for assumed_sini in assumed_sinis:
        print(f"Fitting with assumed sini = {assumed_sini:.2f}...")
        
        fit_theta_pars = true_theta_pars.copy()
        fit_theta_pars['sini'] = assumed_sini

        basis_fitter_imap = build_intensity_map('basis', basis_pars)

        fit_image = basis_fitter_imap.render(
            image_pars,
            fit_theta_pars,
            pars=None,
            image=true_galaxy_image,
            mask=None,
        )

        # --- Calculate Diagnostics ---
        best_fit_coeffs = basis_fitter_imap.fitter.mle_coefficients
        power_spectrum = calculate_power_spectrum(
            best_fit_coeffs,
            nmax=basis_pars['basis_kwargs']['nmax']
        )
        
        residuals = true_galaxy_image - fit_image
        chi2 = np.sum(residuals**2)
        reduced_chi2 = chi2 / (nx * ny)

        fit_results.append({
            'assumed_sini': assumed_sini,
            'power_spectrum': power_spectrum,
            'chi2': reduced_chi2,
            'fit_image': fit_image,
        })

    # --- Step 3: Plot all results ---
    # Find the best and worst fits based on chi-squared
    chi2s = [r['chi2'] for r in fit_results]
    best_fit_index = np.argmin(chi2s)
    worst_fit_index = np.argmax(chi2s)
    
    best_fit_result = fit_results[best_fit_index]
    worst_fit_result = fit_results[worst_fit_index]

    print("Generating comparison plots...")
    plot_fit_comparison(
        true_galaxy_image,
        best_fit_result['fit_image'],
        worst_fit_result['fit_image'],
        hlr=true_hlr, true_sini=true_sini,
        best_assumed_sini=best_fit_result['assumed_sini'],
        worst_assumed_sini=worst_fit_result['assumed_sini'],
        basis_pars=basis_pars
    )

    plot_all_power_spectra(
        fit_results,
        nmax=basis_pars['basis_kwargs']['nmax'],
        true_sini=true_sini
    )

    plot_chi_squared_vs_sini(fit_results)


def plot_fit_comparison(true_image, best_fit_image, worst_fit_image, hlr, true_sini, best_assumed_sini, worst_assumed_sini, basis_pars):
    '''
    Generates a 2x3 multi-panel plot showing the true image, the best-fit
    model, and the residuals for both the best and worst fits.
    '''
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    plt.suptitle('Basis Fit Comparison', fontsize=16)

    # --- Data for plotting ---
    nmax = basis_pars['basis_kwargs']['nmax']
    basis_type = basis_pars['basis_type']
    
    # Row 1: Best-fit case
    res_best = true_image - best_fit_image
    images_best = [true_image, best_fit_image, res_best]
    title1_best = f'Original Model (Sersic n=1, hlr={hlr}, sini={true_sini})'
    title2_best = f'Best Fit (χ², sini={best_assumed_sini:.2f})'
    titles_best = [title1_best, title2_best, 'Residuals']

    # Row 2: Worst-fit case
    res_wrong = true_image - worst_fit_image
    images_wrong = [true_image, worst_fit_image, res_wrong]
    title1_wrong = f'Original Model (Sersic n=1, hlr={hlr}, sini={true_sini})'
    title2_wrong = f'Worst Fit (χ², sini={worst_assumed_sini:.2f})'
    titles_wrong = [title1_wrong, title2_wrong, 'Residuals']

    all_images = [images_best, images_wrong]
    all_titles = [titles_best, titles_wrong]

    # --- Plotting logic ---
    vmax = np.percentile(true_image, 99.5)
    vmin = np.percentile(true_image, 0.5)
    res_vmax = np.percentile(np.abs(np.array([res_best, res_wrong])), 99.5)

    for row in range(2):
        for col in range(3):
            ax = axes[row, col]
            is_residual = (col == 2)
            
            cmap = 'RdBu_r' if is_residual else 'viridis'
            img_vmin = -res_vmax if is_residual else vmin
            img_vmax = res_vmax if is_residual else vmax

            im = ax.imshow(all_images[row][col], origin='lower', cmap=cmap,
                           vmin=img_vmin, vmax=img_vmax)
            ax.set_title(all_titles[row][col])
            ax.set_xticks([])
            ax.set_yticks([])
            
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im, cax=cax)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("basis_fit_residuals")
    plt.show()


def get_shapelet_nm_ordering(nmax):
    '''
    Generates the standard (n, m) ordering for shapelets up to nmax.
    '''
    nm_list = []
    for n in range(nmax + 1):
        for m in range(-n, n + 1, 2):
            nm_list.append((n, m))
    return nm_list


def calculate_power_spectrum(coeffs, nmax):
    '''
    Calculates the average power of the basis coefficients as a function of
    their radial order 'n'.
    '''
    nm_map = get_shapelet_nm_ordering(nmax)
    avg_power_per_order = []
    radial_orders = np.arange(nmax + 1)

    for n_val in radial_orders:
        indices = [i for i, (n, m) in enumerate(nm_map) if n == n_val]
        if not indices:
            avg_power_per_order.append(0)
            continue
        order_coeffs = coeffs[indices]
        avg_power = np.mean(np.abs(order_coeffs)**2)
        avg_power_per_order.append(avg_power)
        
    return np.array(avg_power_per_order)


def plot_all_power_spectra(results, nmax, true_sini):
    '''
    Overplots the coefficient power spectra from multiple fits.
    '''
    plt.figure(figsize=(10, 7))
    radial_orders = np.arange(nmax + 1)

    for result in results:
        assumed_sini = result['assumed_sini']
        power_spectrum = result['power_spectrum']
        
        label = f'Assumed sini = {assumed_sini:.2f}'
        linewidth = 1.5
        zorder = 5

        if np.isclose(assumed_sini, true_sini):
            label += ' (True)'
            linewidth = 3.0
            zorder = 10 

        plt.plot(radial_orders, power_spectrum, marker='o', linestyle='-',
                 label=label, linewidth=linewidth, zorder=zorder)

    plt.yscale('log')
    plt.xlabel('Radial Order (n)')
    plt.ylabel('Average Coefficient Power <|c_nm|^2>')
    plt.title('Coefficient Power Spectrum vs. Assumed Inclination')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.xticks(radial_orders)
    plt.legend()
    plt.savefig("basis_spectrumplot")
    plt.show()


def plot_chi_squared_vs_sini(results):
    '''
    Plots the reduced chi-squared of the fit as a function of the
    assumed sini value.

    Args:
        results (list): A list of dictionaries, where each dict contains
                        'assumed_sini' and 'chi2'.
    '''
    sinis = [r['assumed_sini'] for r in results]
    chi2s = [r['chi2'] for r in results]

    # Find the minimum chi-squared to highlight
    min_chi2_index = np.argmin(chi2s)
    best_fit_sini = sinis[min_chi2_index]
    min_chi2 = chi2s[min_chi2_index]

    plt.figure(figsize=(10, 7))
    plt.plot(sinis, chi2s, marker='o', linestyle='-')
    
    # Highlight the minimum point
    plt.plot(best_fit_sini, min_chi2, marker='*', color='red', markersize=15,
             label=f'Minimum χ² at sini={best_fit_sini:.2f}')
    
    plt.xlabel('Assumed sini')
    plt.ylabel('Reduced Chi-Squared (χ²/N_pix)')
    plt.title('Goodness of Fit vs. Assumed Inclination')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.savefig("basis_chisq_plot")
    plt.show()


if __name__ == '__main__':
    main()
