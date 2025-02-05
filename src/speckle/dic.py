import numpy as np
import cv2 as cv
import sys
import time
from tqdm import tqdm

from speckle import correlation


def dic_local_spline_interpolation(reference_image: np.ndarray,
                                   deformed_image: np.ndarray,
                                   subset_size: int,
                                   subset_step: int,
                                   roi_radius: int,
                                   interpolation: int,
                                   corr_crit: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculates pixel displacements between a reference and deformed image.
    Correlation criterion is calculated globally in the deformed image for each subset in the reference image.
    Interpolation is performed in region of interest (ROI) around correlation minumum.
    value of u,v is taken as the minumum value in the ROI.

    Parameters:
    reference_image     (np.ndarray): Reference Image
    deformed_image      (np.ndarray): Deformed Image
    subset_size         (int): Size of local subsets in pixels 
    subset_step         (int): subset step size in pixels
    roi_radius          (int): Radius of the square of interest around correlation minimum
    interpolation       (int): Number of points between each pixel
    corr_crit           (str): The correlation criterion that will be used. Options are "ssd","nssd","zncc".

    Returns: 
    u   (np.ndarray): Pixel displacement in x direction for each reference subset
    v   (np.ndarray): Pixel displacement in y direction for each reference subset
    """

    time_start = time.perf_counter()


    min_x = subset_size // 2
    min_y = subset_size // 2
    max_x = reference_image.shape[0] - subset_size // 2
    max_y = reference_image.shape[1] - subset_size // 2

    # dont use subsets if rows/cols < 10
    edge_cutoff = 10

    x_values = np.arange(min_x+edge_cutoff, max_x-edge_cutoff, subset_step)
    y_values = np.arange(min_y+edge_cutoff, max_y-edge_cutoff, subset_step)
    shape = (len(y_values), len(x_values)) 

    total_iterations = x_values.shape[0] * y_values.shape[0]

    # Initialize 2D arrays
    u_arr = np.zeros(shape)
    v_arr = np.zeros(shape)

    progress_bar = tqdm(total=total_iterations, desc=f"{'Local DIC Progress:':45}",position=0)
    
    for i, x in enumerate(x_values):
        for j, y in enumerate(y_values):

            ref_subset = correlation.subset(reference_image, x, y, subset_size)
            u, v, ssd, ssd_map = correlation.global_search_opencv(ref_subset, deformed_image, corr_crit)

            roi_x_min = u - roi_radius
            roi_x_max = u + roi_radius
            roi_y_min = v - roi_radius
            roi_y_max = v + roi_radius

            # region of interest
            roi = ssd_map[roi_y_min:roi_y_max,roi_x_min:roi_x_max]

            # interpolate region of interest
            roi_interp = correlation.spline_interpolation(roi, interpolation, interpolation, 3)
            min_val, max_val, min_loc, max_loc = cv.minMaxLoc(roi_interp)

            u_arr[i,j] = float(min_loc[0])/float(interpolation) - roi_radius
            v_arr[i,j] = float(min_loc[1])/float(interpolation) - roi_radius

            progress_bar.update(1)

    return u_arr, v_arr



def dic_global_spline_interpolation(reference_image: np.ndarray,
                                    deformed_image: np.ndarray,
                                    subset_size: int,
                                    subset_step: int,
                                    roi_radius: int,
                                    interpolation: int,
                                    corr_crit: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculates pixel displacements between a reference and deformed image.
    Interpolation is performed globally on reference and deformed image.
    Uses a global grid search to find interpolated reference subset in interpolated deformed image.

    Parameters:
    reference_image     (np.ndarray): Reference Image
    deformed_image      (np.ndarray): Deformed Image
    subset_size         (int): Size of local subsets in pixels 
    subset_step         (int): subset step size in pixels
    roi_radius          (int): Radius of the square of interest around correlation minimum
    interpolation       (int): Number of points between each pixel
    corr_crit           (str): The correlation criterion that will be used. Options are "ssd","nssd","zncc".

    Returns: 
    u   (np.ndarray): Pixel displacement in x direction for each reference subset
    v   (np.ndarray): Pixel displacement in y direction for each reference subset
    """

    time_start = time.perf_counter()

    # perform interpolation of entire reference and deformed image
    reference_image_interp  = correlation.spline_interpolation(reference_image, interpolation, interpolation, 3)
    deformed_image_interp = correlation.spline_interpolation(deformed_image, interpolation, interpolation, 3)


    step = 10 * interpolation # subpx
    subset_size = 21 * interpolation #subpx

    min_x = subset_size // 2
    min_y = subset_size // 2
    max_x = reference_image_interp.shape[0] - subset_size // 2
    max_y = reference_image_interp.shape[1] - subset_size // 2

    # dont use subsets if rows/cols < 10
    edge_cutoff = 10 * interpolation

    x_values = np.arange(min_x+edge_cutoff, max_x-edge_cutoff, step)
    y_values = np.arange(min_y+edge_cutoff, max_y-edge_cutoff, step)
    shape = (len(y_values), len(x_values)) 

    total_iterations = x_values.shape[0] * y_values.shape[0]

    # Initialize 2D arrays
    u_arr = np.zeros(shape)
    v_arr = np.zeros(shape)

    progress_bar = tqdm(total=total_iterations, desc=f"{'Global DIC Progress:':45}",position=0)

    for i, x in enumerate(x_values):
        for j, y in enumerate(y_values):
            
            ref_subset = correlation.subset(reference_image_interp, x, y, subset_size)
            u, v, ssd, ssd_map = correlation.global_search_opencv(ref_subset, deformed_image_interp, corr_crit)

            
            u_arr[i,j] = (u - float(x) + float(min_x))/float(interpolation)
            v_arr[i,j] = (v - float(y) + float(min_y))/float(interpolation)

            # Update progress
            progress_bar.update(1)
    
    return u_arr, v_arr