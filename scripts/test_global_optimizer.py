import numpy as np
import galsim
from galsim.angle import Angle, radians
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.optimize import differential_evolution
from astropy.wcs import WCS
import sep

# Assuming the user has the kl_tools modules in their python path
# and the intensity.py file is accessible.
# If not, these imports would need to be adjusted.
from kl_tools.intensity import build_intensity_map
from kl_tools.parameters import ImagePars
from kl_tools.kross.data import get_kross_obj_data
from kl_tools.interloper_mask import create_interloper_mask

# --- Regularization strength ---
# This can be tuned based on L-curve analysis
LAMBDA_PENALTY = 1e-3

def load_hst_data(kid):
    '''
    Loads and preprocesses HST imaging data for a given KROSS object ID.

    Args:
        kid (int): The KROSS object ID.

    Returns:
        tuple: A tuple containing:
            - hst_image (np.ndarray): The original HST image.
            - image_pars (ImagePars): An ImagePars object with WCS info.
            - initial_params (dict): A dictionary of initial guesses for the fit.
            - interloper_mask (np.ndarray): A boolean mask of interloping sources.
    '''
    print(f"Loading HST data for KID {kid}...")
    obj_data = get_kross_obj_data(kid)
    hst_image = obj_data['hst']
    hst_hdr = obj_data['hst_hdr']
    image_wcs = WCS(hst_hdr)

    # Estimate background for masking
    bkg = sep.Background(hst_image)
    bkg_std = bkg.globalrms

    # Create a mask for foreground objects
    interloper_mask = create_interloper_mask(
        hst_image, bkg_std, threshold=1.5, gaussian_sigma=5.0
    )
    print(f"Masked {np.sum(interloper_mask)} pixels as interlopers.")

    # Setup ImagePars
    Nrow, Ncol = hst_image.shape
    image_pars = ImagePars((Ncol, Nrow), wcs=image_wcs)

    # Get initial parameter guesses from the central object
    objects = sep.extract(hst_image, 1.5, err=bkg_std)
    
    # Find the object closest to the center of the image
    im_center_y, im_center_x = np.array(hst_image.shape) / 2.0
    distances_sq = (objects['x'] - im_center_x)**2 + (objects['y'] - im_center_y)**2
    central_obj_index = np.argmin(distances_sq)
    central_obj = objects[central_obj_index]
    
    # Use geometric mean of semi-major/minor axes for HLR guess
    # and convert from pixels to arcseconds
    hlr_guess_pixels = np.sqrt(central_obj['a'] * central_obj['b'])
    hlr_guess_arcsec = hlr_guess_pixels * image_pars.pixel_scale

    initial_params = {
        'flux': central_obj['flux'],
        'hlr': hlr_guess_arcsec,
        'x0_pix': central_obj['x'],
        'y0_pix': central_obj['y'],
    }

    return hst_image, image_pars, initial_params, interloper_mask


def chi2_to_minimize(params, galaxy_image, image_pars, basis_pars, psf, interloper_mask):
    '''
    Objective function for the optimizer. It fits a composite model of
    a smooth Sersic profile plus a basis set and returns the chi-squared,
    including a regularization penalty on the basis function power.

    Args:
        params (list): A list of parameters to optimize:
                       [flux, hlr, n, g1, g2, theta_int, sini, x0, y0]
        galaxy_image (np.ndarray): The target galaxy image.
        image_pars (ImagePars): The image parameters object.
        basis_pars (dict): The parameters for the basis function set.
        psf (galsim.GSObject): The PSF model.
        interloper_mask (np.ndarray): The mask for interloping objects.

    Returns:
        float: The regularized, reduced chi-squared of the composite model fit.
    '''
    # Unpack all parameters that are being optimized
    flux, hlr, n, g1, g2, theta_int, sini, x0, y0 = params

    # --- 1. Render the smooth Sersic component ---
    sersic_theta_pars = {
        'g1': g1, 'g2': g2, 'theta_int': theta_int,
        'sini': sini, 'x0': x0, 'y0': y0,
    }

    inc = Angle(np.arcsin(sini), radians)
    smooth_model_profile = galsim.InclinedSersic(
        n=n, flux=flux, half_light_radius=hlr, inclination=inc
    )
    rot_angle = Angle(theta_int, radians)
    smooth_model_profile = smooth_model_profile.rotate(rot_angle)
    smooth_model_profile = smooth_model_profile.shear(g1=g1, g2=g2)
    
    convolved_smooth_profile = galsim.Convolve([smooth_model_profile, psf])

    offset = galsim.PositionD(x0 / image_pars.pixel_scale, y0 / image_pars.pixel_scale)
    smooth_model_image = convolved_smooth_profile.drawImage(
        nx=image_pars.Nx, ny=image_pars.Ny, scale=image_pars.pixel_scale, offset=offset
    ).array

    # --- 2. Fit the residuals with the basis functions ---
    residual_image = galaxy_image - smooth_model_image

    basis_fitter_imap = build_intensity_map('basis', basis_pars)
    
    basis_fit_image = basis_fitter_imap.render(
        image_pars,
        sersic_theta_pars,
        pars=None,
        image=residual_image,
        mask=interloper_mask, # Use the mask for the basis fit
    )

    # --- 3. Construct the final composite model ---
    final_composite_model = smooth_model_image + basis_fit_image

    # --- 4. Calculate the regularized chi-squared over unmasked pixels ---
    final_residuals = galaxy_image - final_composite_model
    
    # Apply mask before calculating chi-squared
    unmasked_pixels = ~interloper_mask
    chi2_fit = np.sum(final_residuals[unmasked_pixels]**2)
    
    basis_coeffs = basis_fitter_imap.fitter.mle_coefficients
    power_penalty = np.sum(np.abs(basis_coeffs)**2)
    
    total_chi2 = chi2_fit + LAMBDA_PENALTY * power_penalty
    
    # Normalize by the number of unmasked pixels
    n_pix_unmasked = np.sum(unmasked_pixels)
    reduced_chi2 = total_chi2 / n_pix_unmasked if n_pix_unmasked > 0 else 0

    return reduced_chi2

def main():
    '''
    This script fits a composite model (Sersic + basis functions) to a
    real HST galaxy image to measure its physical parameters.
    '''
    print("--- Starting Composite Model Fit Optimization on HST Data ---")

    # --- Step 1: Load and Prepare HST Data ---
    # KID 122 is a good face-on spiral to test with
    kid = 122
    galaxy_image, image_pars, initial_params, interloper_mask = load_hst_data(kid)

    # --- Define a simple PSF ---
    # For real data, one would use a PSF model derived from stars in the image.
    # Here, we use a simple Gaussian as a placeholder.
    psf_fwhm = 0.08 # arcsec, typical for HST
    psf = galsim.Gaussian(fwhm=psf_fwhm)
    print(f"Using a Gaussian PSF with FWHM = {psf_fwhm} arcsec")

    # --- Define Basis Parameters (with a fixed beta) ---
    basis_pars = {
        'basis_type': 'exp_shapelets',
        'basis_plane': 'disk',
        'skip_ground_state': False,
        'basis_kwargs': {
            'nmax': 10,
            'beta': 0.3,  # Fixed beta for fitting substructure
            'psf': psf   # Pass the PSF to the basis fitter
        }
    }

    # --- Step 2: Optimize for all Sersic and transformation parameters ---
    # Use initial guesses from data to set reasonable bounds
    # Bounds for [flux, hlr, n, g1, g2, theta_int, sini, x0, y0]
    # Note: SEP positions are in pixels, but model positions are in arcsec
    x0_guess_arcsec = (initial_params['x0_pix'] - image_pars.Nx/2) * image_pars.pixel_scale
    y0_guess_arcsec = (initial_params['y0_pix'] - image_pars.Ny/2) * image_pars.pixel_scale

    bounds = [
        (0.5 * initial_params['flux'], 2.0 * initial_params['flux']), # flux
        (0.5 * initial_params['hlr'], 2.0 * initial_params['hlr']),   # hlr
        (0.5, 4.0),                                                   # n
        (-0.5, 0.5),                                                  # g1
        (-0.5, 0.5),                                                  # g2
        (0, np.pi),                                                   # theta_int
        (0.0, 1.0),                                                   # sini
        (x0_guess_arcsec - 0.25, x0_guess_arcsec + 0.25),              # x0 (arcsec)
        (y0_guess_arcsec - 0.25, y0_guess_arcsec + 0.25)               # y0 (arcsec)
    ]

    # Package additional arguments for the objective function
    args = (galaxy_image, image_pars, basis_pars, psf, interloper_mask)

    print("\n--- Starting scipy.optimize.differential_evolution to fit composite model ---")
    result = differential_evolution(
        chi2_to_minimize,
        bounds,
        args=args,
        disp=True,
        polish=True,
        workers=12
        )

    # --- Step 3: Print and Plot Results ---
    best_params = result.x
    print("\n--- Optimization Results ---")
    param_names = ['flux', 'hlr', 'n', 'g1', 'g2', 'theta_int', 'sini', 'x0', 'y0']
    for i, name in enumerate(param_names):
        print(f"Best-fit {name}: {best_params[i]:.3f}")
    print(f"\nMinimum reduced chi-squared (penalized): {result.fun:.6f}")

    # --- Generate final comparison plot with the best-fit model ---
    final_flux, final_hlr, final_n, final_g1, final_g2, final_theta_int, final_sini, final_x0, final_y0 = best_params
    
    final_theta_pars = {
        'g1': final_g1, 'g2': final_g2, 'theta_int': final_theta_int,
        'sini': final_sini, 'x0': final_x0, 'y0': final_y0,
    }
    
    final_inc = Angle(np.arcsin(final_sini), radians)
    final_smooth_profile = galsim.InclinedSersic(n=final_n, flux=final_flux, half_light_radius=final_hlr, inclination=final_inc)
    final_rot_angle = Angle(final_theta_int, radians)
    final_smooth_profile = final_smooth_profile.rotate(final_rot_angle)
    final_smooth_profile = final_smooth_profile.shear(g1=final_g1, g2=final_g2)
    
    convolved_final_smooth = galsim.Convolve([final_smooth_profile, psf])
    final_offset = galsim.PositionD(final_x0 / image_pars.pixel_scale, final_y0 / image_pars.pixel_scale)
    final_smooth_image = convolved_final_smooth.drawImage(nx=image_pars.Nx, ny=image_pars.Ny, scale=image_pars.pixel_scale, offset=final_offset).array
    
    final_residual_image = galaxy_image - final_smooth_image
    final_fitter_imap = build_intensity_map('basis', basis_pars)
    final_basis_image = final_fitter_imap.render(image_pars, final_theta_pars, pars=None, image=final_residual_image, mask=interloper_mask)
    
    final_composite_image = final_smooth_image + final_basis_image

    plot_fit_comparison(
        galaxy_image, final_composite_image, final_smooth_image, final_basis_image, interloper_mask
    )


def plot_fit_comparison(true_image, composite_image, smooth_component, basis_component, mask):
    '''
    Generates a plot comparing the true image to the best-fit composite model
    and its components.
    '''
    # Copy images and apply mask for visualization
    true_image_vis = true_image.copy()
    true_image_vis[mask] = np.nan
    
    residuals = true_image - composite_image
    residuals[mask] = np.nan
    
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))

    vmax = np.nanpercentile(true_image_vis, 99.5)
    vmin = np.nanpercentile(true_image_vis, 0.5)
    res_vmax = np.nanpercentile(np.abs(residuals), 99.5)

    images = [true_image_vis, composite_image, residuals, smooth_component, basis_component]
    titles = ['Original Data', 'Final Composite Model', 'Residuals', 'Smooth Component', 'Substructure Component']
    cmaps = ['viridis', 'viridis', 'RdBu_r', 'viridis', 'viridis']
    vmins = [vmin, vmin, -res_vmax, vmin, None]
    vmaxs = [vmax, vmax, res_vmax, vmax, None]

    for i, ax in enumerate(axes):
        im = ax.imshow(images[i], origin='lower', cmap=cmaps[i], vmin=vmins[i], vmax=vmaxs[i])
        ax.set_title(titles[i])
        ax.set_xticks([])
        ax.set_yticks([])
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax)

    plt.suptitle(f'Composite Model Fit Comparison (KID {kid})', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.show()


if __name__ == '__main__':
    main()
