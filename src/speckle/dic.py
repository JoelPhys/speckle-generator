import numpy as np
import cv2 as cv
import sys
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from icecream import ic
from numba import jit


from speckle import correlation


def dic_global_spline_interpolation(reference_image: np.ndarray,
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
    reference_image_interp = correlation.spline_interpolation_image(reference_image, interpolation, interpolation, 3)
    deformed_image_interp = correlation.spline_interpolation_image(deformed_image, interpolation, interpolation, 3)


    step = subset_step * interpolation # subpx
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


            u_arr[j,i] = (u - float(x) + float(min_x))/float(interpolation)
            v_arr[j,i] = (v - float(y) + float(min_y))/float(interpolation)

            # Update progress
            progress_bar.update(1)

    return u_arr, v_arr





def dic_reference_image_interpolation(reference_image: np.ndarray,
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
            step = 1.0/(interpolation+1.0)

            u_val, v_val, ssd = subset_search_rigid_grid(subset, roi_xmin, roi_xmax, roi_ymin, roi_ymax, step, interpolator)

            # value is negative because its deformed to reference (not reference to deformed)
            u_arr[j,i] = - (u_val - x)
            v_arr[j,i] = - (v_val - y)

            progress_bar.update(1)

    return u_arr, v_arr




def dic_local_correlation_interpolation(reference_image: np.ndarray,
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

            roi_x_min = u - correlation_roi_radius
            roi_x_max = u + correlation_roi_radius
            roi_y_min = v - correlation_roi_radius
            roi_y_max = v + correlation_roi_radius

            # region of interest
            roi = ssd_map[roi_y_min:roi_y_max,roi_x_min:roi_x_max]

            # interpolate region of interest
            roi_interp = correlation.spline_interpolation_image(roi, interpolation, interpolation, 3)
            min_val, max_val, min_loc, max_loc = cv.minMaxLoc(roi_interp)

            u_arr[j,i] = float(min_loc[0])/float(interpolation) - correlation_roi_radius
            v_arr[j,i] = float(min_loc[1])/float(interpolation) - correlation_roi_radius

            progress_bar.update(1)

    return u_arr, v_arr



def dic_global_spline_interpolation(reference_image: np.ndarray,
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


            u_arr[j,i] = (u - float(x) + float(min_x))/float(interpolation)
            v_arr[j,i] = (v - float(y) + float(min_y))/float(interpolation)

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
            xvals = np.arange(subpx_x,subpx_x+subset.shape[0],1)
            yvals = np.arange(subpx_y,subpx_y+subset.shape[0],1)

            # get the interpolated values. Need to convert from subset centre coords to subset corner.
            ref_subset = interpolator(yvals - subset_size, xvals - subset_size)

            #calcuate the ssd for deformed subset vs interpolated reference
            ssd_temp = correlation.ssd(subset,ref_subset)
            if ssd_temp < ssd_val:
                ssd_val = ssd_temp
                u_val = subpx_x
                v_val = subpx_y

    return u_val,v_val,ssd_val


def subset_search_affine_minimizer(p: np.ndarray, reference_subset: np.ndarray, deformed_image: np.ndarray, interpolator, x: int ,y: int):
    """
    Given a reference subset and deformed image find the values of p (affine transformation) that 
    minimise the cost function (SSD,NSSD) using a levenberg-marquardt algorithm
    
    Parameters:
    p                       (np.ndarray): parameters that are optimised by LM algorithm. Eq. (5.20) of Schreier book.
    reference_subset        (np.ndarray): reference subset
    deformed_image_interp   (np.ndarray): interpolated deformed image where we will search for the reference subset
    x                       (int): coordinate at centre of reference subset along x-axis
    y:                      (int): coordinate at centre of reference subset along y-axis
    subset_size             (np.ndarray): subset_size in number of pixels
    """

    subset_size = reference_subset.shape[0]
    half_size = subset_size // 2

    # reference image subset
    x1, x2 = x - half_size, x + half_size + 1
    y1, y2 = y - half_size, y + half_size + 1

    subset_x = np.arange(x - half_size, x + half_size + 1)
    subset_y = np.arange(y - half_size, y + half_size + 1)
    xx, yy = np.meshgrid(subset_x, subset_y)    

    # list of coordinates 
    coords_x = np.arange(x1,x2,1)
    coords_y = np.arange(y1,y2,1)

    #pixel coordinates of reference subset
    xx, yy = np.meshgrid(coords_x,coords_y)

    # apply shape function to obtain subset subpixel coordinates. should always be the same size as reference subset
    deformed_coords_x = (p[0] + (1 + p[2]) * xx + p[3] * yy).flatten()
    deformed_coords_y = (p[1] + p[4] * xx + (1 + p[5]) * yy).flatten()

    interp_values = interpolator(deformed_coords_y,deformed_coords_x,grid=False) 

    ssd_val = correlation.ssd(reference_subset.flatten(), interp_values)

    # debugging
    #ic(deformed_coords_x)
    #ic(deformed_coords_y)
    #ic(deformed_coords_x.shape)
    #ic(reference_subset.shape)
    #ic(p)
    #ic(ssd_val)

    # figure debugging
    # fig, axes = plt.subplots(1, 3, figsize=(10, 7)) 
    # axes[0].set_title("Subset in Original Image")
    # axes[1].set_title("Deformed Image")
    # axes[0].imshow(reference_subset,origin="lower")
    # axes[1].scatter(deformed_coords_x,deformed_coords_y,c=interp_values)
    # plt.tight_layout()  
    # plt.show()


    return ssd_val
