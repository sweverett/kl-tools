'''
This module provides tools and classes for PSF-related operations, particularly for 
computing PSF-blurred, intensity-weighted velocity maps
'''

import numpy as np
import galsim
from typing import Tuple

from kl_tools.parameters import ImagePars
from kl_tools.velocity import VelocityMap
from kl_tools.intensity import IntensityMap
from kl_tools.utils import build_map_grid

def _make_vmap_fine_image_pars(vmap_image_pars: ImagePars, oversample: int) -> ImagePars:
    if oversample < 1 or int(oversample) != oversample:
        raise ValueError('oversample must be a positive integer')

    Nx_f = vmap_image_pars.Nx * oversample
    Ny_f = vmap_image_pars.Ny * oversample

    wcs_f = vmap_image_pars.wcs.deepcopy()

    # scale pixel size
    if getattr(wcs_f.wcs, "cd", None) is not None:
        wcs_f.wcs.cd = np.array(wcs_f.wcs.cd, dtype=float) / oversample
    else:
        wcs_f.wcs.cdelt = np.array(wcs_f.wcs.cdelt, dtype=float) / oversample

    # keep sky center fixed in pixel units
    wcs_f.wcs.crpix = np.array(wcs_f.wcs.crpix, dtype=float) * oversample

    # update raster size in the WCS
    wcs_f.pixel_shape = (Nx_f, Ny_f)   # (Naxis1, Naxis2) = (Nx, Ny)
    wcs_f.array_shape = (Ny_f, Nx_f)
    wcs_f.wcs.set()

    return ImagePars(shape=(Nx_f, Ny_f), wcs=wcs_f, indexing='xy')

def psf_weighted_velocity_from_arrays(
    I_fine: np.ndarray,               # intrinsic intensity on fine grid
    V_fine: np.ndarray,               # intrinsic velocity on same fine grid
    fine_pixel_scale: float,          # arcsec/pixel for the fine grid
    oversample: int,                  # fine grid is oversample× the vmap grid
    psf: galsim.GSObject,
    imap_deconv_psf: galsim.GSObject = None,
    denom_floor: float = 1e-8,
    return_conv_imap: bool = False,
    ):
    '''
    Core PSF operation: arrays in → PSF blur → bin → ratio
    Implements: v_obs = [ PSF * (I * v) ] / [ PSF * I ], on a fine grid,
    then bins back to the vmap grid.

    This function is model-agnostic *and has no WCS logic*
    '''

    if (oversample < 1) or (int(oversample) != oversample):
        raise ValueError('oversample must be a positive integer')
    if I_fine.shape != V_fine.shape:
        raise ValueError('I_fine and V_fine must have the same shape')

    # Optional deconvolution of a pre-blurred model intensity
    if imap_deconv_psf is not None:
        I_fine = _convolve_fft(
            I_fine, fine_pixel_scale, galsim.Deconvolve([imap_deconv_psf])
        ).array

    Iv_fine = I_fine * V_fine

    # PSF convolution on the fine grid (FFT)
    I_blur_fine  = _convolve_fft(I_fine,  fine_pixel_scale, psf)
    Iv_blur_fine = _convolve_fft(Iv_fine, fine_pixel_scale, psf)

    # Downsample to vmap grid: bin (sum) → mean
    I_obs_img  = I_blur_fine.bin(oversample, oversample)
    Iv_obs_img = Iv_blur_fine.bin(oversample, oversample)

    I_obs_img  /= oversample**2
    Iv_obs_img /= oversample**2

    I_obs = I_obs_img.array
    v_obs = Iv_obs_img.array / np.maximum(I_obs, denom_floor)

    if return_conv_imap:
        return (v_obs, I_obs)
    else:
        return v_obs

# ---------------------------------------------------------------------

def psf_convolved_vmap(
    imap: IntensityMap,
    vmap: VelocityMap,
    vmap_image_pars: ImagePars,
    psf: galsim.GSObject,
    theta_pars: dict,
    pars: dict = None,
    oversample: int = 3,
    imap_deconv_psf: galsim.GSObject = None,
    denom_floor: float = 1e-8,
    return_conv_imap: bool = False,
):
    '''
    High-level convenience wrapper: render intrinsic I,V on fine grid → call core PSF op
      1) Build vmap-matched oversampled grid.
      2) Render intrinsic I and V on that exact grid (no interpolation).
      3) Apply PSF blur + bin + ratio.

    Returns v_obs (and I_obs if requested) on the ORIGINAL vmap grid.
    '''

    fine_pars = _make_vmap_fine_image_pars(vmap_image_pars, oversample)

    # Intrinsic composite intensity (image, _continuum)
    I_fine = imap.render(
        image_pars=fine_pars,
        theta_pars=theta_pars,
        pars=pars,
        im_type='emission',
        redo=True,
    )

    # Intrinsic velocity on the same fine grid.
    Nx_f, Ny_f = fine_pars.Nx, fine_pars.Ny
    Xv_cen, Yv_cen = build_map_grid(Nx_f, Ny_f, indexing=vmap_image_pars.indexing)
    pix_scale = fine_pars.pixel_scale
    V_fine = vmap('obs', Xv_cen*pix_scale, Yv_cen*pix_scale)

    return psf_weighted_velocity_from_arrays(
        I_fine=I_fine,
        V_fine=V_fine,
        fine_pixel_scale=fine_pars.pixel_scale,
        oversample=oversample,
        psf=psf,
        imap_deconv_psf=imap_deconv_psf,
        denom_floor=denom_floor,
        return_conv_imap=return_conv_imap,
    )


# ---------------------------------------------------------------------
# helper functions for the PSF convolution

def _convolve_fft(
    arr: np.ndarray, pixscale: float, psf: galsim.GSObject
) -> galsim.Image:
    '''
    FFT-based convolution via GalSim; returns a GalSim Image at the same pixel scale.
    '''

    img = galsim.ImageF(arr.astype(np.float32), scale=pixscale)
    ii  = galsim.InterpolatedImage(img)
    conv = galsim.Convolve([ii, psf])

    Nx = arr.shape[1]
    Ny = arr.shape[0]

    return conv.drawImage(
        scale=pixscale, method='fft', nx=Nx, ny=Ny
        )

# def _build_map_grid(
#         Nx: int, Ny: int, indexing: str = 'xy'
#         ) -> Tuple[np.ndarray, np.ndarray]:
#     '''
#     Minimal centered grid builder (pixel centers, origin at image center).
#     Matches the convention used elsewhere: returns (X, Y) in 'xy' or 'ij'.
#     '''

#     # centered coords: integers with 0 at center pixel
#     x = (np.arange(Nx, dtype=np.float64) - (Nx - 1) / 2.0)
#     y = (np.arange(Ny, dtype=np.float64) - (Ny - 1) / 2.0)
#     X, Y = np.meshgrid(x, y, indexing='xy')

#     if indexing not in ('xy', 'ij'):
#         raise ValueError("indexing must be 'xy' or 'ij'")

#     return (X, Y) if indexing == 'xy' else (Y, X)
