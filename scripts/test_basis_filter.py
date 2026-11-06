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

from pyswarms.single.global_best import GlobalBestPSO
import ipdb

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

# --- NEW FUNCTION (Task 1) ---
def generate_model_image(params, image_pars, psf):
    '''
    Renders a smooth, convolved Sersic model image based on input parameters.
    
    Args:
        params (list): A list of parameters to optimize:
                       [flux, hlr, n, g1, g2, theta_int, sini, x0, y0]
        image_pars (ImagePars): The image parameters object.
        psf (galsim.GSObject): The PSF model.

    Returns:
        np.ndarray: The rendered model image.
    '''
    
    # Unpack all parameters that are being optimized
    flux, hlr, n, g1, g2, theta_int, sini, x0, y0 = params

    # --- Render the smooth Sersic component ---
    inc = Angle(np.arcsin(sini), radians)
    smooth_model_profile = galsim.InclinedSersic(
        n=n, flux=flux, half_light_radius=hlr, inclination=inc
    )
    
    rot_angle = Angle(theta_int, radians)
    smooth_model_profile = smooth_model_profile.rotate(rot_angle)
    smooth_model_profile = smooth_model_profile.shear(g1=g1, g2=g2)

    convolved_smooth_profile = smooth_model_profile # Default if psf is None
    if psf is not None:
        convolved_smooth_profile = galsim.Convolve([smooth_model_profile, psf])

    offset = galsim.PositionD(x0 / image_pars.pixel_scale, y0 / image_pars.pixel_scale)
    smooth_model_image = convolved_smooth_profile.drawImage(
        nx=image_pars.Nx, ny=image_pars.Ny, scale=image_pars.pixel_scale, offset=offset).array
        
    return smooth_model_image

# --- MODIFIED FUNCTION (Task 1) ---
def chisq_to_minimize(params, galaxy_image, image_pars, psf, interloper_mask):
    '''
    Objective function for the optimizer. Calls generate_model_image and
    returns the chi-squared.

    Args:
        params (list): A list of parameters to optimize.
        galaxy_image (np.ndarray): The target galaxy image.
        image_pars (ImagePars): The image parameters object.
        psf (galsim.GSObject): The PSF model.
        interloper_mask (np.ndarray): The mask for interloping objects.
                                      (NOTE: Not currently used in chi-sq calc)

    Returns:
        float: The chi-squared of the model fit.
    '''
    
    # 1. Generate the model
    smooth_model_image = generate_model_image(params, image_pars, psf)
    
    # 2. Compute chi-squared
    # Note: The interloper_mask is not being used here, but is kept in
    # the function signature to match the optimizer's 'args' tuple.
    # To use it, you would mask the 'chi' array before summing.
    chi = galaxy_image - smooth_model_image
    
    chisq = np.sum(chi[interloper_mask == False]**2)
    
    return chisq


def run_pyswarm_optimization(
    objective_func,
    scipy_bounds,
    scipy_args,
    n_particles=50,
    n_iters=1000,
    # n_processes argument removed
    pso_options={'c1': 0.5, 'c2': 0.3, 'w': 0.9}):
    """
    Runs a PySwarm GlobalBestPSO optimization using inputs formatted
    for scipy.differential_evolution.

    NOTE: Does not support parallel processing, as pyswarms.single
    does not provide an 'n_processes' or 'workers' argument.
    """

    # --- 1. Define the internal objective wrapper ---
    # This wrapper adapts PySwarm's (n_particles, n_dims) array input
    # to your function's (n_dims,) vector input.
    
    def _pyswarm_adapter(all_particles, **static_kwargs):
        """
        Wrapper defined in-scope to "capture" the objective_func.
        It iterates over all particles and calls the user's function
        for each one.
        """
        n_particles = all_particles.shape[0]
        costs = np.zeros(n_particles)

        for i in range(n_particles):
            params_i = all_particles[i]
            # Call the original function with one particle's parameters
            # and the static kwargs passed by optimizer.optimize()
            costs[i] = objective_func(params_i, **static_kwargs)
            
        return costs

    # --- 2. Convert Scipy bounds to PySwarm bounds ---
    bounds_array = np.array(scipy_bounds)
    pyswarm_bounds = (bounds_array[:, 0], bounds_array[:, 1])
    dimensions = len(scipy_bounds)

    # --- 3. Convert Scipy args (tuple) to PySwarm kwargs (dict) ---
    # Argument names from your chisq_to_minimize() definition:
    arg_names = [
        'galaxy_image', 
        'image_pars', 
        'psf', 
        'interloper_mask'
    ]
    
    if len(scipy_args) != len(arg_names):
        raise ValueError(
            f"Argument mismatch: The 'scipy_args' tuple has {len(scipy_args)} "
            f"items, but the optimizer expected {len(arg_names)}: {arg_names}"
        )
            
    pyswarm_kwargs = dict(zip(arg_names, scipy_args))

    # --- 4. Instantiate and run the optimizer (CORRECTED) ---
    print(f"Initializing PySwarm PSO with {n_particles} particles for {n_iters} iterations...")
    
    # *** CORRECTED CONSTRUCTOR ***
    # Removed n_processes and kwargs
    optimizer = GlobalBestPSO(
        n_particles=n_particles,
        dimensions=dimensions,
        options=pso_options,
        bounds=pyswarm_bounds
    )

    # *** CORRECTED OPTIMIZE CALL ***
    # Run the optimization, passing the static arguments (pyswarm_kwargs)
    # here. They will be forwarded to our _pyswarm_adapter.
    cost, pos = optimizer.optimize(
        _pyswarm_adapter, 
        iters=n_iters, 
        verbose=True,
        **pyswarm_kwargs  # Pass static args to optimize()
    )

    print(f"Optimization finished. Best cost: {cost}")
    return cost, pos

# --- NEW FUNCTION (Task 2) ---
def visualize_fit_results(galaxy_image, best_fit_params, image_pars, psf):
    '''
    Generates and displays a 3-panel plot: Data, Model, and Residual.
    
    Args:
        galaxy_image (np.ndarray): The target galaxy image.
        best_fit_params (list): The final best-fit parameter list.
        image_pars (ImagePars): The image parameters object.
        psf (galsim.GSObject): The PSF model.
    '''
    print("Generating visualization of best-fit model...")

    # 1. Generate the best-fit model
    best_model_image = generate_model_image(best_fit_params, image_pars, psf)
    
    # 2. Calculate residual
    residual_image = galaxy_image - best_model_image
    
    # 3. Create the plots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Smooth Model Fit Results', fontsize=16)
    
    # Determine vmin/vmax for data and model
    vmax = np.percentile(galaxy_image, 99.5)
    vmin = np.percentile(galaxy_image, 1.0)
    
    # --- Plot Data ---
    im1 = ax1.imshow(galaxy_image, origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
    ax1.set_title('Data')
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im1, cax=cax1)
    
    # --- Plot Model ---
    im2 = ax2.imshow(best_model_image, origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
    ax2.set_title('Best-Fit Model')
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im2, cax=cax2)
    
    # Determine vmin/vmax for residual
    res_vmax = np.percentile(np.abs(residual_image), 99)
    res_vmin = -res_vmax
    
    # --- Plot Residual ---
    im3 = ax3.imshow(residual_image, origin='lower', cmap='bwr', vmin=res_vmin, vmax=res_vmax)
    ax3.set_title('Residual (Data - Model)')
    divider3 = make_axes_locatable(ax3)
    cax3 = divider3.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im3, cax=cax3)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.93]) # Adjust for suptitle
    plt.show()


# --- MODIFIED FUNCTION (Task 3) ---
def fit_smooth_model(galaxy_image, image_pars, initial_params, interloper_mask, psf):

    # get an initial guess.
    x0_guess_arcsec = (initial_params['x0_pix'] - image_pars.Nx/2) * image_pars.pixel_scale
    y0_guess_arcsec = (initial_params['y0_pix'] - image_pars.Ny/2) * image_pars.pixel_scale

    # Maybe pyswarm?
    bounds = [
        (0.5 * initial_params['flux'], 2.0 * initial_params['flux']), # flux
        (0.5 * initial_params['hlr'], 2.0 * initial_params['hlr']),   # hlr
        (0.5, 4.0),                                                  # n
        (-0.5, 0.5),                                                 # g1
        (-0.5, 0.5),                                                 # g2
        (0, np.pi),                                                  # theta_int
        (0.0, 1.0),                                                  # sini
        (x0_guess_arcsec - 0.25, x0_guess_arcsec + 0.25),            # x0
        (y0_guess_arcsec - 0.25, y0_guess_arcsec + 0.25)             # y0
        ]

    # Package additional arguments (no change)
    args = (galaxy_image, image_pars, psf, interloper_mask)

    # Run the new PySwarm optimizer
    # We use the same 'bounds' and 'args' variables directly
    best_chisq, best_params = run_pyswarm_optimization(
        objective_func=chisq_to_minimize,  # Pass your function
        scipy_bounds=bounds,
        scipy_args=args,
        n_particles=50,   # Set particle count
        n_iters=200,     # Set iteration count
        )
    
    # --- ADDED CALL (Task 3) ---
    # Visualize the results
    #visualize_fit_results(galaxy_image, best_params, image_pars, psf)
    return best_params

if __name__ == '__main__':
    # KID 122 is a good face-on spiral to test with
    #kid = 122
    #kid = 53
    kid = 32
    galaxy_image, image_pars, initial_params, interloper_mask = load_hst_data(kid)
    
    psf_fwhm = 0.08 # arcsec, typical for HST
    psf = galsim.Gaussian(fwhm=psf_fwhm)
    print(f"Using a Gaussian PSF with FWHM = {psf_fwhm} arcsec")

    # --- MODIFIED CALL (Task 3) ---
    best_fit_params = fit_smooth_model(galaxy_image, image_pars, initial_params, interloper_mask, psf)
    sersic_theta_pars = {
        'g1':        best_fit_params[3],
        'g2':        best_fit_params[4],
        'theta_int': best_fit_params[5],
        'sini':      best_fit_params[6],
        'x0':        best_fit_params[7],
        'y0':        best_fit_params[8],
        }

    
    # --- Define Basis Parameters (with a fixed beta) ---
    basis_pars = {
        'basis_type': 'exp_shapelets',
        'basis_plane': 'obs',
        'skip_ground_state': False,
        'basis_kwargs': {
            'nmax': 25,
            'beta': 0.001,  # Fixed beta for fitting substructure
            'psf': None   # Pass the PSF to the basis fitter
        }
    }

    model_image = generate_model_image(best_fit_params, image_pars, psf)
    residual_image =   galaxy_image - model_image

    basis_fitter_imap = build_intensity_map('basis', basis_pars)
    basis_fit_image = basis_fitter_imap.render(
        image_pars,
        sersic_theta_pars,
        weights=None,
        pars=None,
        image=residual_image,
        mask=interloper_mask, # Use the mask for the basis fit
        )

    # Subtract the basis fit from the original data, and re-fit.
    smoother_galaxy_image = galaxy_image - basis_fit_image
    smooth_fit_params = fit_smooth_model(smoother_galaxy_image, image_pars, initial_params, interloper_mask, psf)
    smooth_sersic_theta_pars = {
        'g1':        smooth_fit_params[3],
        'g2':        smooth_fit_params[4],
        'theta_int': smooth_fit_params[5],
        'sini':      smooth_fit_params[6],
        'x0':        smooth_fit_params[7],
        'y0':        smooth_fit_params[8],
        }
    smooth_model_image = generate_model_image(smooth_fit_params, image_pars, psf)
    fig,((ax1,ax2,ax3),(ax4,ax5,ax6),(ax7,ax8,ax9)) = plt.subplots(ncols=3,nrows=3,figsize=(7,7))
    ax1.imshow(galaxy_image);
    ax2.imshow(model_image);
    ax3.imshow(residual_image);
    ax4.imshow(basis_fit_image);
    ax5.imshow(galaxy_image-basis_fit_image);
    ax6.imshow(smooth_model_image);
    ax7.imshow(galaxy_image-basis_fit_image - smooth_model_image);
    ax8.imshow(smooth_model_image - model_image)
    fig.savefig(f"smooth_model_image_fit_comparison_{kid}.png")
    ipdb.set_trace()
