import numpy as np
import galsim
from galsim.angle import Angle, radians
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import fitsio
#from scipy.optimize import differential_evolution
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



def get_basis_power_spectrum(basis_obj):
    coeffs = basis_obj.fitter.mle_coefficients
    nmax = basis_obj.basis.nmax
    orders = []
    avg_power = []
    std_power = []
    
    for l in range(nmax):
        power_at_l = []
        for m in range(-l, l + 1):
            n = basis_obj.basis.lm_to_n(l, m)
            power = np.abs(coeffs[n])**2
            power_at_l.append(power)
        orders.append(l)
        avg_power.append(np.mean(power_at_l))
        std_power.append(np.std(power_at_l))
    return np.array(orders),np.array(avg_power),np.array(std_power)



def load_jwst_data(ext):
    '''
    Loads and preprocesses JWST imaging data for a given KROSS object ID.

    Args:
        ext (int): The JWST fits file extension number .

    Returns:
        tuple: A tuple containing:
            - jwst_image (np.ndarray): The original JWST image.
            - image_pars (ImagePars): An ImagePars object with WCS info.
            - initial_params (dict): A dictionary of initial guesses for the fit.
            - interloper_mask (np.ndarray): A boolean mask of interloping sources.
    '''
    print(f"Loading JWST data for EXT {ext}...")
    #obj_data = get_kross_obj_data(ext)
    jwst_image = fitsio.read("./data_compile_short_GDS_ID195280_emlonly.fits",ext)
    jwst_hdr = fitsio.read_header("./data_compile_short_GDS_ID195280_emlonly.fits",ext)
    image_wcs = WCS(jwst_hdr)

    # Estimate background for masking
    bkg = sep.Background(jwst_image)
    bkg_std = bkg.globalrms

    # Create a mask for foreground objects
    interloper_mask = create_interloper_mask(
        jwst_image, bkg_std, threshold=1.5, gaussian_sigma=5.0
    )
    print(f"Masked {np.sum(interloper_mask)} pixels as interlopers.")

    # Setup ImagePars
    Nrow, Ncol = jwst_image.shape
    image_pars = ImagePars((Ncol, Nrow), wcs=image_wcs)

    # Get initial parameter guesses from the central object
    objects = sep.extract(jwst_image, 1.5, err=bkg_std)
    
    # Find the object closest to the center of the image
    im_center_y, im_center_x = np.array(jwst_image.shape) / 2.0
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

    return jwst_image, image_pars, initial_params, interloper_mask



def generate_model_image(params, image_pars, psf):
    '''
    Renders a smooth, convolved Sersic model image based on input parameters.
    
    Args:
        params (list): A list of parameters to optimize:
                       [flux, hlr, g1, g2, theta_int, sini, x0, y0]
        image_pars (ImagePars): The image parameters object.
        psf (galsim.GSObject): The PSF model.

    Returns:
        np.ndarray: The rendered model image.
    '''
    
    # Unpack all parameters that are being optimized
    flux, hlr, g1, g2, theta_int, sini, x0, y0 = params

    # --- Render the smooth Sersic component ---
    inc = Angle(np.arcsin(sini), radians)
    smooth_model_profile = galsim.InclinedExponential(
        flux=flux, half_light_radius=hlr, inclination=inc)
    
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


def chisq_to_minimize(params, galaxy_image, image_pars, psf, interloper_mask, basis_pars):
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
    #flux, hlr, g1, g2, theta_int, sini, x0, y0 = params
    sersic_theta_pars = {
        'g1':        params[2],
        'g2':        params[3],
        'theta_int': params[4],
        'sini':      params[5],
        'x0':        params[6],
        'y0':        params[7],
        }

    
    resid = galaxy_image - smooth_model_image
    basis_fitter_imap = build_intensity_map('basis', basis_pars)
    basis_fit_image_full = basis_fitter_imap.render(
        image_pars,
        sersic_theta_pars,
        weights=None,
        pars=None,
        image=resid,
        mask=interloper_mask, # Use the mask for the basis fit
        )    

    
    chisq = np.sum(resid[interloper_mask == False]**2)

    lambda_reg = 1e-8
    ell,power,power_err = get_basis_power_spectrum(basis_fitter_imap)

    penalty = lambda_reg * np.sum(power*ell)

    
    return chisq + penalty


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
        'interloper_mask',
        'basis_pars'
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


def fit_smooth_model(galaxy_image, image_pars, initial_params, interloper_mask, psf, basis_pars):

    # get an initial guess.
    x0_guess_arcsec = (initial_params['x0_pix'] - image_pars.Nx/2) * image_pars.pixel_scale
    y0_guess_arcsec = (initial_params['y0_pix'] - image_pars.Ny/2) * image_pars.pixel_scale

    # Maybe pyswarm?
    bounds = [
        (0.5 * initial_params['flux'], 2.0 * initial_params['flux']), # flux
        (0.5 * initial_params['hlr'], 2.0 * initial_params['hlr']),   # hlr
        (-0.5, 0.5),                                                 # g1
        (-0.5, 0.5),                                                 # g2
        (0, np.pi),                                                  # theta_int
        (0.0, 1.0),                                                  # sini
        (x0_guess_arcsec - 0.25, x0_guess_arcsec + 0.25),            # x0
        (y0_guess_arcsec - 0.25, y0_guess_arcsec + 0.25)             # y0
        ]

    # Package additional arguments (no change)
    args = (galaxy_image, image_pars, psf, interloper_mask, basis_pars)

    # Run the new PySwarm optimizer
    # We use the same 'bounds' and 'args' variables directly
    best_chisq, best_params = run_pyswarm_optimization(
        objective_func=chisq_to_minimize,  # Pass your function
        scipy_bounds=bounds,
        scipy_args=args,
        n_particles=100,   # Set particle count
        n_iters=200,     # Set iteration count
        )
    
    ipb.set_trace()
    return best_params



def main(ext):


    galaxy_image, image_pars, initial_params, interloper_mask = load_jwst_data(ext)
    
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
    psf = None

    
    best_params = fit_smooth_model(galaxy_image, image_pars, initial_params, interloper_mask, psf, basis_pars)



if __name__ == '__main__':
    main(4)
