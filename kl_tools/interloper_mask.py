import numpy as np
import sep
import galsim as gs
from scipy.ndimage import binary_dilation

def create_interloper_mask(image, bkg_std, threshold=1.5, margin_pixels=1, gaussian_sigma=None):
    """
    Generates a mask for interlopers in an image using the SEP library.

    This function detects all sources in an image. It identifies the source
    closest to the image center as the primary target. It then masks all
    other sources (interlopers), except for those that are adjacent to the
    primary target, or within a specified pixel margin of it.

    The source detection can be enhanced by filtering with a Gaussian kernel,
    specified by its standard deviation.

    Args:
        image (np.ndarray): The input image data.
        bkg_std (float): The standard deviation of the image background noise.
        threshold (float, optional): The detection threshold in units of
                                     background standard deviation. Defaults to 1.5.
        margin_pixels (int, optional): The margin in pixels around the central
                                       source. Interlopers within this margin
                                       will not be masked. Defaults to 1.
        gaussian_sigma (float, optional): The standard deviation (in pixels) of a
                                          Gaussian kernel for filtering before
                                          detection. If None, no filtering is
                                          applied. Defaults to None.

    Returns:
        np.ndarray: A boolean mask with the same shape as the input image,
                    where True values correspond to pixels to be masked.
    """
    # SEP works with a specific data type and byte order, so we make a copy.
    data = image.copy().astype(np.float64)

    # If a Gaussian sigma is provided, create the filter kernel.
    filter_kernel = None
    if gaussian_sigma is not None and gaussian_sigma > 0:
        # Create a normalized 2D Gaussian kernel using GalSim.
        # The kernel size is chosen to be ~8x the sigma to avoid truncation effects.
        kernel_size = 2 * int(4.0 * gaussian_sigma + 0.5) + 1
        kernel = gs.Gaussian(sigma=gaussian_sigma).drawImage(
            nx=kernel_size, ny=kernel_size, scale=1.
        ).array
        kernel /= np.sum(kernel)
        filter_kernel = kernel

    # Use sep to extract sources and get a segmentation map.
    try:
        objects, segmap = sep.extract(
            data,
            thresh=threshold,
            err=bkg_std,
            segmentation_map=True,
            filter_kernel=filter_kernel
        )
    except Exception as e:
        print(f"Warning: SEP source extraction failed with error: {e}")
        print("Returning an empty mask.")
        return np.zeros_like(image, dtype=bool)

    # If no sources are found, no masking is needed.
    if (objects is None) or (len(objects) == 0):
        return np.zeros_like(image, dtype=bool)

    # Find the object closest to the center of the image.
    im_center_y, im_center_x = np.array(image.shape) / 2.0
    distances_sq = (objects['x'] - im_center_x)**2 + (objects['y'] - im_center_y)**2
    central_obj_index = np.argmin(distances_sq)

    # In the segmentation map, the label for an object is its index + 1.
    central_label = central_obj_index + 1

    # If there's only one object, no interlopers to mask.
    if len(objects) <= 1:
        return np.zeros_like(image, dtype=bool)

    # Create a boolean mask for just the central source.
    central_source_mask = (segmap == central_label)

    # Dilate the central source mask to include adjacent pixels up to the margin.
    structure = np.ones((3, 3), dtype=bool)
    dilated_mask = binary_dilation(central_source_mask, structure=structure, iterations=margin_pixels)

    # Get the labels of all objects that fall within this dilated region.
    labels_to_keep = np.unique(segmap[dilated_mask])

    # The final mask is True for all pixels that belong to any source
    # but are NOT in our set of labels to keep.
    is_source = (segmap != 0)
    is_kept_source = np.isin(segmap, labels_to_keep)
    interloper_mask = is_source & ~is_kept_source

    return interloper_mask