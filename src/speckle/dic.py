import numpy as np
import cv2 as cv
import sys
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from icecream import ic
from numba import jit
from scipy.optimize import minimize, least_squares, leastsq, brute

from . import correlation

def reference_image_interpolation_roi_gridsearch(reference_image: np.ndarray,
                                      deformed_image: np.ndarray,
                                      subset_size: int,
                                      subset_step: int,
                                      subset_search_radius: int,
                                      interpolation: int,
                                      corr_crit: str) -> tuple[np.ndarray, np.ndarray]:
    


    time_start = time.perf_counter()


    # get the interpolation of the entire reference image
    interpolator = correlation.spline_interpolation_object(reference_image, interpolation, interpolation, 3)

    min_x = subset_size // 2
    min_y = subset_size // 2
    max_x = reference_image.shape[1] - subset_size // 2
    max_y = reference_image.shape[0] - subset_size // 2

    # dont use subsets if rows/cols < 10
    edge_cutoff = 10

    x_values = np.arange(min_x+edge_cutoff, max_x-edge_cutoff, subset_step)
    y_values = np.arange(min_y+edge_cutoff, max_y-edge_cutoff, subset_step)
    shape = (len(y_values), len(x_values)) 

    total_iterations = x_values.shape[0] * y_values.shape[0]

    # Initialize 2D arrays
    u_arr = np.zeros(shape)
    v_arr = np.zeros(shape)

    progress_bar = tqdm(total=total_iterations, desc=f"{'Searching for deformed subsets in the interpolated reference image using a grid search within user specified search radius':150}",position=0)

    # looping over the subsets
    for i, x in enumerate(x_values):
        for j, y in enumerate(y_values):

            # Take a subset from the non-interpolated deformed image and look for that in the reference Image.
            # This is becuase we only need to perform a single interpolation on the reference subset rather
            # than an interpolation on every deformed_image in the pipeline
            subset = correlation.subset(deformed_image, x, y, subset_size)

            # get the boundary of our search
            roi_xmin = x - subset_search_radius
            roi_xmax = x + subset_search_radius
            roi_ymin = y - subset_search_radius
            roi_ymax = y + subset_search_radius
            step = 1.0/(interpolation)

            u_val, v_val, ssd = subset_search_rigid_grid(subset, roi_xmin, roi_xmax, roi_ymin, roi_ymax, step, interpolator)

            # value is negative because its deformed to reference (not reference to deformed)
            u_arr[j,i] = - (u_val - x)
            v_arr[j,i] = - (v_val - y)

            progress_bar.update(1)

    return u_arr, v_arr



def reference_image_interpolation_minimizer(reference_image: np.ndarray,
                                      deformed_image: np.ndarray,
                                      subset_size: int,
                                      subset_step: int,
                                      bounds: np.ndarray,
                                      corr_crit: str) -> tuple[np.ndarray, np.ndarray]:
    


    time_start = time.perf_counter()


    # get the interpolation of the entire reference image
    interpolator = correlation.spline_interpolation_object(reference_image, 3)

    min_x = subset_size // 2
    min_y = subset_size // 2
    max_x = reference_image.shape[1] - subset_size // 2
    max_y = reference_image.shape[0] - subset_size // 2

    # dont use subsets if rows/cols < 10
    edge_cutoff = 50

    x_values = np.arange(min_x+edge_cutoff, max_x-edge_cutoff, subset_step)
    y_values = np.arange(min_y+edge_cutoff, max_y-edge_cutoff, subset_step)
    shape = (len(y_values), len(x_values), 6) 

    total_iterations = x_values.shape[0] * y_values.shape[0]

    # Initialize 2D arrays
    p_arr = np.zeros(shape)
    ssd_arr = np.zeros((len(y_values), len(x_values)))

    progress_bar = tqdm(total=total_iterations, desc=f"{'Searching for deformed subsets in the interpolated reference image using scipy.optimize.minimize':150}",position=0)


    p = np.array([0.0,0.0,0.0,0.0,0.0,0.0])

    # looping over the subsets
    for i, x in enumerate(x_values):
        for j, y in enumerate(y_values):

            # Take a subset from the non-interpolated deformed image and look for that in the reference Image.
            # This is becuase we only need to perform a single interpolation on the reference subset rather
            # than an interpolation on every deformed_image in the pipeline
            subset = correlation.subset(deformed_image, x, y, subset_size)

            half_size = subset_size // 2

            # reference image subset
            x1, x2 = x - half_size, x + half_size + 1
            y1, y2 = y - half_size, y + half_size + 1

            # list of coordinates 
            coords_x = np.arange(x1,x2,1)
            coords_y = np.arange(y1,y2,1)

            #pixel coordinates of reference subset
            xx, yy = np.meshgrid(coords_x,coords_y)

            # sol = brute(subset_search_affine_minimizer, p, args=(subset,interpolator,xx,yy),bounds=bounds)
            sol = minimize(subset_search_affine_minimizer, p, args=(subset,interpolator,xx,yy),bounds=bounds)
            # sol = minimize(subset_search_affine_minimizer, p, args=(subset, interpolator, x, y),bounds=bounds, method='L-BFGS-B',jac=gradient)
            # sol = leastsq(subset_search_affine_minimizer, p, args=(subset,interpolator,x,y))
            # sol = least_squares(subset_search_affine_minimizer, p, args=(subset,interpolator,x,y),bounds=bounds)
            # exit(0)
            p = sol.x
            ssd_val = sol.fun


            # value is negative because its deformed subset looking searching in reference image
            p_arr[j,i,0:6]  = p
            ssd_arr[j,i] = ssd_val

            progress_bar.update(1)

    return p_arr, ssd_arr





def local_correlation_interpolation(reference_image: np.ndarray,
                                        deformed_image: np.ndarray,
                                        subset_size: int,
                                        subset_step: int,
                                        correlation_roi_radius: int,
                                        interpolation: int,
                                        corr_crit: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculates pixel displacements between a reference and deformed image.
    Correlation criterion is calculated globally in the deformed image for each subset in the reference image.
    Interpolation is performed in region of interest (ROI) around correlation minumum.
    value of u,v is taken as the minumum value in the ROI.

    Parameters:
    reference_image         (np.ndarray): Reference Image
    deformed_image          (np.ndarray): Deformed Image
    subset_size             (int): Size of local subsets in pixels 
    subset_step             (int): subset step size in pixels
    correlation_roi_radius  (int): Radius of the square of interest around correlation minimum
    interpolation           (int): Number of points between each pixel
    corr_crit               (str): The correlation criterion that will be used. Options are "ssd","nssd","zncc".

    Returns: 
    u   (np.ndarray): Pixel displacement in x direction for each reference subset
    v   (np.ndarray): Pixel displacement in y direction for each reference subset
    """

    time_start = time.perf_counter()


    min_x = subset_size // 2
    min_y = subset_size // 2
    max_x = reference_image.shape[1] - subset_size // 2
    max_y = reference_image.shape[0] - subset_size // 2

    # dont use subsets if rows/cols < 10
    edge_cutoff = 10

    x_values = np.arange(min_x+edge_cutoff, max_x-edge_cutoff, subset_step)
    y_values = np.arange(min_y+edge_cutoff, max_y-edge_cutoff, subset_step)
    shape = (len(y_values), len(x_values)) 

    total_iterations = x_values.shape[0] * y_values.shape[0]

    # Initialize 2D arrays
    u_arr = np.zeros(shape)
    v_arr = np.zeros(shape)

    progress_bar = tqdm(total=total_iterations, desc=f"{'Searching for deformed subsets GLOBALLY in the reference image. Interpolate the correlation to find subpixel shift:':150}",position=0)

    for i, x in enumerate(x_values):
        for j, y in enumerate(y_values):

            def_subset = correlation.subset(deformed_image, x, y, subset_size)
            u, v, ssd, ssd_map = correlation.global_search_opencv(def_subset, reference_image, corr_crit)

            roi_x_min = u - correlation_roi_radius
            roi_x_max = u + correlation_roi_radius
            roi_y_min = v - correlation_roi_radius
            roi_y_max = v + correlation_roi_radius

            # region of interest
            roi = ssd_map[roi_y_min:roi_y_max,roi_x_min:roi_x_max]

            # interpolate region of interest
            roi_interp = correlation.spline_interpolation_image(roi, interpolation, interpolation, 3)
            min_val, max_val, min_loc, max_loc = cv.minMaxLoc(roi_interp)

            u_arr[j,i] = - float(min_loc[0])/float(interpolation) + correlation_roi_radius
            v_arr[j,i] = - float(min_loc[1])/float(interpolation) + correlation_roi_radius

            progress_bar.update(1)

    return u_arr, v_arr









def global_spline_interpolation(reference_image: np.ndarray,
                                    deformed_image: np.ndarray,
                                    subset_size: int,
                                    subset_step: int,
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
    reference_image_interp  = correlation.spline_interpolation_image(reference_image, interpolation, interpolation, 3)
    deformed_image_interp = correlation.spline_interpolation_image(deformed_image, interpolation, interpolation, 3)


    step = subset_step * interpolation # subpx
    subset_size = 21 * interpolation #subpx

    min_x = subset_size // 2
    min_y = subset_size // 2
    max_x = reference_image_interp.shape[1] - subset_size // 2
    max_y = reference_image_interp.shape[0] - subset_size // 2

    # dont use subsets if rows/cols < 10
    edge_cutoff = 10 * interpolation

    x_values = np.arange(min_x+edge_cutoff, max_x-edge_cutoff, step)
    y_values = np.arange(min_y+edge_cutoff, max_y-edge_cutoff, step)
    shape = (len(y_values), len(x_values)) 

    total_iterations = x_values.shape[0] * y_values.shape[0]

    # Initialize 2D arrays
    u_arr = np.zeros(shape)
    v_arr = np.zeros(shape)

    progress_bar = tqdm(total=total_iterations, desc=f"{'Searching for interpolated deformed subsets GLOBALLY in interpolated reference image using openCV':150}",position=0)

    for i, x in enumerate(x_values):
        for j, y in enumerate(y_values):

            def_subset = correlation.subset(deformed_image_interp, x, y, subset_size)
            u, v, ssd, ssd_map = correlation.global_search_opencv(def_subset, reference_image_interp, corr_crit)
            # ic(u,v)

            u_arr[j,i] = - (u - float(x) + float(min_x))/float(interpolation)
            v_arr[j,i] = - (v - float(y) + float(min_y))/float(interpolation)

            # Update progress
            progress_bar.update(1)

    return u_arr, v_arr





def subset_search_rigid_grid(subset: np.ndarray, xmin: float,xmax: float, ymin:float, ymax: float, step: int, interpolator) -> tuple[float,float,float]:

    subset_size = subset.shape[0] // 2

    # initialise vars
    u_val = 0.0
    v_val = 0.0
    ssd_val = float("Inf")


    # loop over all subsets in the search area
    for subpx_x in np.arange(xmin,xmax,step):
        for subpx_y in np.arange(ymin,ymax,step):

            # get my 'interpolated' subset to compare against
            subpx_x_max = subpx_x+subset.shape[0]-1
            subpx_y_max = subpx_y+subset.shape[0]-1
            xvals = np.linspace(subpx_x,subpx_x_max,subset.shape[0])
            yvals = np.linspace(subpx_y,subpx_y_max,subset.shape[0])
            # get the interpolated values. Need to convert from subset centre coords to subset corner.
            ref_subset = interpolator(yvals - subset_size, xvals - subset_size)
            #calcuate the ssd for deformed subset vs interpolated reference
            ssd_temp = correlation.ssd(subset,ref_subset)
            if ssd_temp < ssd_val:
                ssd_val = ssd_temp
                u_val = subpx_x
                v_val = subpx_y

    return u_val,v_val,ssd_val







def subset_search_affine_minimizer(p: np.ndarray, reference_subset: np.ndarray, interpolator, xx: np.ndarray ,yy: np.ndarray) -> float:

    # apply shape function to obtain subset subpixel coordinates. should always be the same size as reference subset
    deformed_coords_x = (p[0] + (1 + p[2]) * xx + p[3] * yy)
    deformed_coords_y = (p[1] + p[4] * xx + (1 + p[5]) * yy)

    interp_values = interpolator(deformed_coords_y,deformed_coords_x,grid=False) 

    ssd_val = correlation.ssd(reference_subset, interp_values)

    return ssd_val

def gradient(p: np.ndarray, reference_subset: np.ndarray, interpolator, xx: np.ndarray, yy: np.ndarray) -> np.ndarray:
    # Compute the deformed coordinates for the affine transformation
    deformed_coords_x = p[0] + (1 + p[2]) * xx + p[3] * yy
    deformed_coords_y = p[1] + p[4] * xx + (1 + p[5]) * yy

    # Compute the derivatives of the interpolation with respect to x and y
    dx_values = interpolator(deformed_coords_y, deformed_coords_x, dx=1, dy=0, grid=False)  # Derivative w.r.t. x
    dy_values = interpolator(deformed_coords_y, deformed_coords_x, dx=0, dy=1, grid=False)  # Derivative w.r.t. y

    # Compute the SSD residuals
    residual_x = reference_subset - dx_values
    residual_y = reference_subset - dy_values

    # Initialize the gradient vector with the same size as p
    grad = np.zeros_like(p)

    # Gradients w.r.t. p[0] (translation in x)
    grad[0] = -2 * np.sum(residual_x * dx_values)

    # Gradients w.r.t. p[1] (translation in y)
    grad[1] = -2 * np.sum(residual_y * dy_values)

    # Gradients w.r.t. p[2] (scaling in x)
    grad[2] = -2 * np.sum(residual_x * dx_values * xx)

    # Gradients w.r.t. p[3] (shear in x-y plane)
    grad[3] = -2 * np.sum(residual_x * dx_values * yy)

    # Gradients w.r.t. p[4] (shear in y-x plane)
    grad[4] = -2 * np.sum(residual_y * dy_values * xx)

    # Gradients w.r.t. p[5] (scaling in y)
    grad[5] = -2 * np.sum(residual_y * dy_values * yy)

    return grad
