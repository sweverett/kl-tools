'''
This module provides tools and classes for PSF-related operations, particularly for 
computing PSF-blurred, intensity-weighted velocity maps
'''

import numpy as np
import galsim

from kl_tools.parameters import ImagePars
from kl_tools.velocity import VelocityMap
from kl_tools.intensity import IntensityMap
from kl_tools.utils import build_map_grid


def psf_convolved_vmap(
    imap: IntensityMap,
    imap_image_pars: ImagePars,
    vmap: VelocityMap,
    vmap_image_pars: ImagePars,
    psf: galsim.GSObject,
    oversample: int = 3,
    imap_deconv_psf: galsim.GSObject | None = None,
    denom_floor: float = 1e-8,
    x_extent_arcsec: float | None = None,
    y_extent_arcsec: float | None = None,
    return_conv_imap: bool = False,
):
    '''
    Compute v_obs = (PSF * (I * v)) / (PSF * I) on the requested grid.

    Coordinates are evaluated at the same sky locations by bridging through WCS:
    vmap pixels → world → imap pixels. Oversampling reduces PSF mixing biases.

    Parameters
    ----------
    imap : IntensityMap
        Intensity map evaluated in its native image coordinates (pixels).
    imap_image_pars : ImagePars
        Shape, pixel scale, and WCS for the intensity map.
    vmap : VelocityMap
        Velocity map evaluated in its native image coordinates (pixels).
    vmap_image_pars : ImagePars
        Shape, pixel scale, and WCS for the velocity/output plane.
    psf : galsim.GSObject
        Seeing PSF for convolution.
    oversample : int, optional
        Integer oversampling factor for the fine grid.
    imap_deconv_psf : galsim.GSObject or None, optional
        Optional PSF to deconvolve from imap before applying `psf`.
    denom_floor : float, optional
        Minimum denominator to avoid division by zero.
    x_extent_arcsec, y_extent_arcsec : float or None, optional
        If provided, override the physical span of the output (vmap) grid.
    return_conv_imap : bool, optional
        If True, also return the PSF-convolved intensity map on the vmap grid.

    Returns
    -------
    v_obs : ndarray
        Intensity-weighted velocity map on the vmap grid.
    I_obs : ndarray, optional
        PSF-convolved intensity map on the vmap grid (if requested).
    '''

    if (oversample < 1) or (int(oversample) != oversample):
        raise ValueError('oversample must be a positive integer')

    # the vmap image pars defines the output image pars
    Nx, Ny = vmap_image_pars.Nx, vmap_image_pars.Ny

    out_scale  = _ensure_pixel_scale(vmap_image_pars, x_extent_arcsec, y_extent_arcsec)
    fine_scale = out_scale / oversample

    # oversampled grid on the vmap plane (centered pixel coords)
    Xv_cen, Yv_cen = build_map_grid(
        Nx * oversample, Ny * oversample, indexing=vmap_image_pars.indexing
    )

    # vmap centered pixels -> absolute pixel indices for WCS
    cx_v, cy_v = _pix_center_origin_xy(vmap_image_pars)
    Xv_abs = Xv_cen + cx_v
    Yv_abs = Yv_cen + cy_v

    # vmap abs pixels -> world -> imap abs pixels -> imap centered pixels
    world = vmap_image_pars.pixel_to_world(Xv_abs, Yv_abs)
    Xi_abs, Yi_abs = imap_image_pars.world_to_pixel(world)
    cx_i, cy_i = _pix_center_origin_xy(imap_image_pars)
    Xi_cen = Xi_abs - cx_i
    Yi_cen = Yi_abs - cy_i

    # sample intrinsic maps on native pixel coords
    I_fine = np.asarray(imap(Xi_cen, Yi_cen), dtype=np.float64)
    V_fine = np.asarray(vmap('obs', Xv_cen, Yv_cen), dtype=np.float64)

    # Optional deconvolution of imap before applying seeing PSF
    if imap_deconv_psf is not None:
        I_fine = _convolve_fft(
            I_fine, fine_scale, galsim.Deconvolve([imap_deconv_psf])
            ).array

    Iv_fine = I_fine * V_fine

    # PSF convolution on the fine grid (FFT)
    I_blur_fine  = _convolve_fft(I_fine,  fine_scale, psf)
    Iv_blur_fine = _convolve_fft(Iv_fine, fine_scale, psf)

    # downsample to vmap grid: bin (sum) then convert to mean
    I_obs_img  = I_blur_fine.bin(oversample);  I_obs_img  /= oversample**2
    Iv_obs_img = Iv_blur_fine.bin(oversample); Iv_obs_img /= oversample**2

    # intensity-weighted velocity
    v_obs = Iv_obs_img.array / np.maximum(I_obs_img.array, denom_floor)
    I_obs = I_obs_img.array

    if return_conv_imap:
        return v_obs, I_obs

    return v_obs

# ------------------------------------------------------------
# Helper functions for the PSF convolution

def _pix_center_origin_xy(image_pars: ImagePars) -> tuple[float, float]:
    '''
    Return the 0-based pixel index of the image center (CRPIX - 0.5) in (x, y).
    '''

    crpix = image_pars.wcs.wcs.crpix  # FITS 1-based

    return float(crpix[0] - 0.5), float(crpix[1] - 0.5)


def _ensure_pixel_scale(
    image_pars: ImagePars,
    x_extent_arcsec: float | None = None,
    y_extent_arcsec: float | None = None,
) -> float:
    '''
    Decide working pixel scale [arcsec/pixel] for the output/vmap plane.

    If x/y extents are provided, they imply scale = extent / Npix. Both must agree.
    '''

    Nx, Ny = image_pars.Nx, image_pars.Ny
    sx = image_pars.pixel_scale if x_extent_arcsec is None else (x_extent_arcsec / Nx)
    sy = image_pars.pixel_scale if y_extent_arcsec is None else (y_extent_arcsec / Ny)
    if not np.isclose(sx, sy, rtol=0, atol=1e-12):
        raise ValueError(
            f'Non-square pixels implied by overrides: sx={sx:.6g}", sy={sy:.6g}". '
            'Provide equal extents or omit one.'
        )

    return float(sx)

def _convolve_fft(
        arr: np.ndarray, pixscale: float, psf: galsim.GSObject
        ) -> galsim.Image:
    '''
    FFT-based convolution via GalSim; returns a GalSim Image at the same pixel scale.
    '''

    img = galsim.ImageF(arr.astype(np.float32), scale=pixscale)
    ii  = galsim.InterpolatedImage(img, x_interpolant='lanczos3', normalize=False)
    conv = galsim.Convolve([ii, psf])

    return conv.drawImage(scale=pixscale, method='fft')
