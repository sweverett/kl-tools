import numpy as np
from scipy.optimize import minimize, differential_evolution
import galsim as gs

def estimate_gal_properties(
    image,
    image_pars,
    guess,
    bounds,
    sersic_n=None,
    psf=None,
    loss='l2',
    weight=None,
    mask=None,
    method='galsim',
    draw_image_kwargs=None,
    gsparams=None,
    optimizer='minimize',
    optimize_kwargs=None,
):
    '''
    Estimate the best-fit profile parameters for an inclined galaxy model.

    Parameters
    ----------
    image : ndarray
        The 2D intensity image.
    image_pars : ImagePars
        Object describing shape, pixel scale, and WCS.
    guess : dict
        Initial guesses for parameters. Must include 'flux' and 'scale_radius'. 
        Angles must be in radians, scales & offsets in arcsec.
    bounds : dict
        Bounds for parameters. Must include 'flux' and 'scale_radius'. Angles
        must be in radians, scales & offsets in arcsec.
    sersic_n : float, optional
        Sérsic index. If passed, overrides the 'n' parameter in `guess` and `bounds`.
    psf : GSObject or None, optional
        Optional PSF for convolution with the galaxy model.
    loss : str, optional
        Loss function to use: 'l2' (default) or 'l1'.
    weight : ndarray or None, optional
        Optional weight map. Should match shape of `image`. Only used if `mask` is also applied correctly.
    mask : ndarray or None, optional
        Optional boolean mask of the same shape as `image`. True pixels are ignored in fitting and model evaluation.
    method  str, optional
        Fitting backend to use. Currently only 'galsim' is implemented.
    draw_image_kwargs : dict or None, optional
        Additional keyword arguments passed to `galsim.GSObject.drawImage`.
    gsparams : galsim.GSParams or None, optional
        Optional GSParams for controlling the precision of the Galsim 
        calculations. If None, default GSParams are used.
    optimizer : str, optional
        Optimizer to use. Currently only 'minimize' and 
        'differential_evolution' are supported.
    optimize_kwargs : dict or None, optional
        Additional keyword arguments for the optimizer, such as method

    Returns
    -------
    result : dict
        Dictionary with best-fit parameters, model image, optimizer success flag, and diagnostic messages.
    '''
  
    if method == 'galsim':
        return _estimate_gal_properties_with_galsim(
            image=image,
            image_pars=image_pars,
            guess=guess,
            bounds=bounds,
            sersic_n=sersic_n,
            psf=psf,
            loss=loss,
            weight=weight,
            mask=mask,
            draw_image_kwargs=draw_image_kwargs,
            gsparams=gsparams,
            optimizer=optimizer,
            optimize_kwargs=optimize_kwargs
        )
    else:
        raise NotImplementedError(f'Method "{method}" is not implemented.')

def _estimate_gal_properties_with_galsim(
    image,
    image_pars,
    guess,
    bounds,
    sersic_n,
    psf,
    loss,
    weight,
    mask,
    draw_image_kwargs,
    gsparams,
    optimizer,
    optimize_kwargs
):
    nx, ny = image_pars.Nx, image_pars.Ny
    pixel_scale = image_pars.pixel_scale

    default_guess = {
        'x0': 0.0,
        'y0': 0.0,
        'theta_int': 0.0,
        'sini': 0.5,
        'n': 1.0
    }
    default_bounds = {
        'x0': (pixel_scale*-nx/2., pixel_scale*nx/2.),
        'y0': (pixel_scale*-ny/2., pixel_scale*ny/2.),
        'theta_int': (0.0, np.pi),
        'sini': (0.0, 1.0),
        'n': (0.8, 4.5)
    }

    if 'flux' not in guess or 'scale_radius' not in guess:
        raise ValueError('guess must include flux and scale_radius')
    if 'flux' not in bounds or 'scale_radius' not in bounds:
        raise ValueError('bounds must include flux and scale_radius')

    if sersic_n is not None:
        # index is set explicitly, so don't scan over it
        if 'n' in guess:
            del guess['n']
        if 'n' in bounds:
            del bounds['n']
        del default_guess['n']
        del default_bounds['n']

    for key in default_guess:
        if key not in guess:
            guess[key] = default_guess[key]
        if key not in bounds:
            bounds[key] = default_bounds[key]

    keys = list(guess.keys())
    param_guess = [guess[k] for k in keys]
    param_bounds = [bounds[k] for k in keys]

    if optimize_kwargs is None:
        optimize_kwargs = {
            'method': 'L-BFGS-B',
        }

    if optimizer == 'minimize':
        result = minimize(
            _gal_residual,
            x0=param_guess,
            args=(keys, image, image_pars, psf, mask, weight, loss, draw_image_kwargs, gsparams, sersic_n),
            bounds=param_bounds,
            **optimize_kwargs
        )
    elif optimizer == 'differential_evolution':
        result = differential_evolution(
            func=_gal_residual,
            bounds=param_bounds,
            args=(keys, image, image_pars, psf, mask, weight, loss, draw_image_kwargs, gsparams, sersic_n),
            polish=True,
            **optimize_kwargs
        )

    best_params = dict(zip(keys, result.x))
    best_model = _render_model_image(
        best_params,
        image_pars,
        psf,
        draw_image_kwargs,
        gsparams,
        sersic_n
        )

    return {
        'params': best_params,
        'model_image': best_model,
        'success': result.success,
        'message': result.message,
        'fun': result.fun
    }

def _gal_residual(
        param_list,
        keys,
        image,
        image_pars,
        psf, mask,
        weight,
        loss,
        draw_image_kwargs,
        gsparams,
        sersic_n
        ):

    params = dict(zip(keys, param_list))
    if params['scale_radius'] <= 0 or params['flux'] <= 0:
        return np.inf

    model_image = _render_model_image(
        params, image_pars, psf, draw_image_kwargs, gsparams, sersic_n
        )

    diff = image - model_image
    if mask is not None:
        valid = ~mask
        diff = diff[valid]
        if weight is not None:
            weight = weight[valid]
            diff *= weight
    elif weight is not None:
        diff *= weight

    if loss == 'l2':
        return np.sum(diff**2)
    elif loss == 'l1':
        return np.sum(np.abs(diff))
    else:
        raise ValueError(f'Unknown loss: {loss}')

def _render_model_image(
        pars,
        image_pars,
        psf=None,
        draw_image_kwargs=None,
        gsparams=None,
        sersic_n=None
        ):

    if (pars['sini'] < 0) or (pars['sini'] > 1):
        raise ValueError(f"Invalid sini value: {pars['sini']}. Must be in [0, 1].")
    inclination = np.arcsin(pars['sini'])

    if (sersic_n is not None) and (sersic_n == 1):
      # this is faster than InclinedSersic for n == 1
      gal = gs.InclinedExponential(
        inclination=inclination*gs.radians,
        scale_radius=pars['scale_radius'],
        flux=pars['flux'],
        gsparams=gsparams
      )
    else:
      gal = gs.InclinedSersic(
        n=sersic_n if sersic_n is not None else pars['n'],
        inclination=inclination,
        scale_radius=pars['scale_radius'],
        flux=pars['flux'],
        gsparams=gsparams
      )

    gal = gal.rotate(pars['theta_int']*gs.radians)
    if psf is not None:
        gal = gs.Convolve([gal, psf])

    # galsim expects the offsets in pixel coords
    offset_pixels = [
        pars['x0'] / image_pars.pixel_scale,
        pars['y0'] / image_pars.pixel_scale
    ]

    if draw_image_kwargs is None:
        draw_image_kwargs = {}

    model = gal.drawImage(
        offset=gs.PositionD(offset_pixels),
        scale=image_pars.pixel_scale,
        nx=image_pars.Nx,
        ny=image_pars.Ny,
        **draw_image_kwargs
    ).array

    return model
