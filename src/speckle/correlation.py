import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from scipy.signal import correlate2d
from scipy.interpolate import RegularGridInterpolator, RectBivariateSpline
from numba import jit

def pattern_gradient(subset):
    """Image gradients using sobel."""
    grad_x = cv2.Sobel(subset, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(subset, cv2.CV_64F, 0, 1, ksize=3)
    return grad_x, grad_y

@jit(nopython=True)
def ssd(reference_subset: np.ndarray, deformed_subset: np.ndarray) -> float:
    ssd_value = np.sum((reference_subset - deformed_subset) ** 2)
    return ssd_value

@jit(nopython=True)
def nssd(reference_subset, deformed_subset) -> float:

    mean_ref = np.mean(reference_subset)
    mean_def = np.mean(deformed_subset)
    std_ref = np.std(reference_subset)
    std_def = np.std(deformed_subset)

    normed_ref = (reference_subset - mean_ref) / std_ref
    normed_def = (deformed_subset - mean_def) / std_def

    nssd_value = np.sum((normed_ref - normed_def) ** 2)

    return nssd_value

@jit(nopython=True)
def znssd(reference_subset, deformed_subset) -> float:

    mean_ref = np.mean(reference_subset)
    mean_def = np.mean(deformed_subset)

    zero_mean_ref = reference_subset - mean_ref
    zero_mean_def = deformed_subset - mean_def

    znssd_value = np.sum((zero_mean_ref - zero_mean_def) ** 2)

    return znssd_value

@jit(nopython=True)
def subset( image: np.ndarray, x: int, y: int, subset_size: int) -> np.ndarray:
    """
    Parameters:
    x (int): x-coord of subset center in image
    y (int): y-coord of subset center in image

    """

    half_size = subset_size // 2

    # reference image subset
    x1, x2 = x - half_size, x + half_size + 1
    y1, y2 = y - half_size, y + half_size + 1

    # Ensure indices are within bounds
    if (x1 < 0 or y1 < 0 or x2 > image.shape[1] or y2 > image.shape[0]):
        raise ValueError(f"Subset exceeds image boundaries.\nSubset Pixel Range:\n"
                        f"x1: {x1}\n"
                        f"x2: {x2}\n"
                        f"y1: {y1}\n"
                        f"y2: {y2}")

    # Extract subsets
    subset = image[y1:y2, x1:x2]

    return subset



def perform_interpolation(image: np.ndarray, interp_x: int, interp_y: int, kind: str) -> np.ndarray:
    """
    Interpolatation using scipy.interpolate.RegularGridInterpolator

    Parameters:
    interp_x (int): number of interpolations between each pixel along x axis
    interp_y (int): number of interpolations between each pixel along y axis
    kind     (str): Type of interpolation. Supported values are linear", "nearest", "slinear", "cubic", "quintic" and "pchip".

    Returns: 
    np.ndarray: Interpolated image.
    """

    dims_ref = image.shape

    x  = np.linspace(0, dims_ref[0] - 1, dims_ref[0])
    y  = np.linspace(0, dims_ref[0] - 1, dims_ref[1])
    interpolator_ref = interpolate.RegularGridInterpolator((y, x),image, method=kind)

    # new grid
    x_vals = np.linspace(0, dims_ref[1] - 1, (dims_ref[1] - 1) * interp_x + 1)
    y_vals = np.linspace(0, dims_ref[0] - 1, (dims_ref[0] - 1) * interp_y + 1)
    x_mesh, y_mesh = np.meshgrid(x_vals, y_vals)

    new_points = np.column_stack([y_mesh.ravel(), x_mesh.ravel()])
    interped_vals = interpolator_ref(new_points)


    interped_image = interped_vals.reshape(len(y_mesh), len(x_mesh))

    # plt.figure()
    # plt.scatter(x_vals,vals[0,:])
    # plt.scatter(x,image_ref[0,:])
    # plt.show()

    return interped_image

def spline_interpolation_object(image: np.ndarray, interp_x: int, interp_y: int, degree: int=3) -> np.ndarray:
    """
    Interpolation using RectBivariateSpline.
    
    Parameters:
    image (np.ndarray): Input 2D image array.
    interp_x (int): Number of interpolations between each pixel along the x-axis.
    interp_y (int): Number of interpolations between each pixel along the y-axis.
    degree (int): degree of polynomial. Default is 3
    
    Returns:
    np.ndarray: Interpolated image.
    """
    dims_ref = image.shape
    
    x = np.arange(dims_ref[1])  # Original x-coordinates
    y = np.arange(dims_ref[0])  # Original y-coordinates
    
    # Create the spline interpolator
    interpolator = RectBivariateSpline(y, x, image, kx=3, ky=3)  # Cubic spline by default
    
    return interpolator


def spline_interpolation_image(image: np.ndarray, interp_x: int, interp_y: int, degree: int=3) -> np.ndarray:
    """
    Interpolation using RectBivariateSpline.
    
    Parameters:
    image (np.ndarray): Input 2D image array.
    interp_x (int): Number of interpolations between each pixel along the x-axis.
    interp_y (int): Number of interpolations between each pixel along the y-axis.
    degree (int): degree of polynomial. Default is 3
    
    Returns:
    np.ndarray: Interpolated image.
    """
    dims_ref = image.shape
    
    x = np.arange(dims_ref[1])  # Original x-coordinates
    y = np.arange(dims_ref[0])  # Original y-coordinates
    
    # Create the spline interpolator
    interpolator = RectBivariateSpline(y, x, image, kx=3, ky=3)  # Cubic spline by default
    
    # New grid for interpolation
    x_vals = np.linspace(0, dims_ref[1] - 1, (dims_ref[1] - 1) * interp_x + 1)
    y_vals = np.linspace(0, dims_ref[0] - 1, (dims_ref[0] - 1) * interp_y + 1)
    
    # Perform interpolation
    interped_image = interpolator(y_vals, x_vals)
    
    return interped_image


def global_search(ref_subset: np.ndarray, image_def: np.ndarray) -> np.ndarray:

    subset_size = ref_subset.shape[0] * 4
    min_x = subset_size // 2
    min_y = subset_size // 2
    max_x = image_def.shape[0] - subset_size // 2
    max_y = image_def.shape[1] - subset_size // 2
    
    ssd_map = np.zeros((max_x-min_x, max_y-min_y))

    # Loop over a grid of positions to create multiple residuals
    for u in range(min_x, max_x):
        for v in range(min_y, max_y):

            i = u - min_x
            j = v - min_y

            def_subset = subset(image_def, u, v, subset_size)
            ssd_map[j,i] = ssd(ref_subset,def_subset)


    return ssd_map




def global_find_min(ssd_map: np.ndarray) -> tuple[int,int,float]:

    ssd_min = np.min(ssd_map)
    indices = np.unravel_index(np.argmin(ssd_map), ssd_map.shape)
    u_abs = indices[0]
    v_abs = indices[1]
    
    return u_abs,v_abs,ssd_min



def global_search_opencv(ref_subset: np.ndarray, image_def: np.ndarray, method: str) -> tuple[float,float, float, np.ndarray]:

    if method == "ssd":
        method = getattr(cv,"TM_SQDIFF")
    elif method == "nssd":
        method = getattr(cv,"TM_SQDIFF_NORMED")
    elif method == "zncc":
        method = getattr(cv,"TM_CCOEFF_NORMED")
    #elif method = "znssd":
    #    method = getattr(cv,"TM_SQDIFF")
    else:
        raise ValueError("Unknown correlation function:" + method)

    #need to convert to float32 for opencv
    subset = ref_subset.astype(np.float32)
    deformed_image = image_def.astype(np.float32)

    method = cv.TM_SQDIFF
    correlation_map = cv.matchTemplate(deformed_image,subset,method)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(correlation_map)
    
    return min_loc[0], min_loc[1], min_val, correlation_map




def global_search_scipy(ref_subset, image_def) -> tuple[int, int, float]:

    subset_size = ref_subset.shape[0]

   # use scipy.signal
    ref_squared = np.sum(ref_subset ** 2)
    corr = correlate2d(image_def, ref_subset, mode="valid", boundary="fill", fillvalue=0)

    ssd_map = ref_squared - 2 * corr

    # debugging
    # plt.figure(figsize=(10, 5))
    # plt.subplot(1, 2, 1)
    # plt.title("Original Image")
    # plt.imshow(ssd_map, cmap="gray")
    # plt.show()

    v_final, u_final = np.unravel_index(np.argmin(ssd_map), ssd_map.shape)
    
    return u_final, v_final, np.min(ssd_map)



def wave_distort(image, amplitude=2, frequency=2*np.pi, axis=0):
    """
    Applies a sine wave distortion to a 2D image along a single axis,
    without changing the image's overall dimensions.

    Parameters:
        image (numpy.ndarray): 2D array representing the image.
        amplitude (float): The maximum displacement of pixels along the axis.
        frequency (float): Frequency of the sine wave.
        axis (int): Axis along which to apply the distortion (0 for rows, 1 for columns).

    Returns:
        numpy.ndarray: Distorted image with the same dimensions as input.
    """
    h, w = image.shape

    # Generate the grid of original coordinates
    y = np.arange(h)
    x = np.arange(w)
    
    # Create a RegularGridInterpolator
    interpolator = RegularGridInterpolator((y, x), image, bounds_error=False, fill_value=0)

    # Distort the coordinates using a sine wave along the specified axis
    if axis == 0:  # Distortion along rows (stretch/squash vertically)
        y_new = y + amplitude * np.sin(2 * np.pi * x / w * frequency)
        x_new = x  # No change along x-axis
    else:  # Distortion along columns (stretch/squash horizontally)
        x_new = x + amplitude * np.sin(2 * np.pi * y / h * frequency)
        y_new = y  # No change along y-axis

    # Create a meshgrid for the new distorted coordinates
    Y_new, X_new = np.meshgrid(y_new, x_new)

    # Flatten the meshgrid to use in interpolator
    new_coords = np.column_stack([X_new.ravel(), Y_new.ravel()])

    # Get the interpolated values at the new coordinates
    distorted_image = interpolator(new_coords).reshape(h, w)

    return distorted_image


