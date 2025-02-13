
import numpy as np
from icecream import ic
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from scipy.optimize import least_squares, minimize
from scipy.interpolate import RegularGridInterpolator, RectBivariateSpline
import speckle

from speckle import correlation

def main():


    width = 200
    height = 200
    pattern = speckle.Pattern(image_width=width,image_height=height, mean_radius=3, spacing=7, variability=0.6,stddev_size=0.1,gray_level=256,stddev_smooth=1.0,seed=1)
    speckle_pattern = pattern.generate()

    x = 100
    y = 100
    subset_size = 51
    reference_subset = correlation.subset(speckle_pattern, x, y, subset_size)

    #-----------------------------------------------------------------------------------------------------------------------------
    # subpixel roll
    #-----------------------------------------------------------------------------------------------------------------------------

    # copy the original pattern
    deformed_pattern = speckle_pattern.copy()

    #create interpolation
    interp_x = interp_y = 4
    subpx_samp = 1.0/interp_x
    shift = 5

    # pixel values of original images
    orig_x = np.arange(speckle_pattern.shape[1])
    orig_y = np.arange(speckle_pattern.shape[0])

    # Pixel values of interpolated image
    high_res_x = np.linspace(0, speckle_pattern.shape[1] - 1, (speckle_pattern.shape[1] - 1) * interp_x + 1)
    high_res_y = np.linspace(0, speckle_pattern.shape[0] - 1, (speckle_pattern.shape[0] - 1) * interp_y + 1)

    ic(high_res_x)
    # interpolate the deformed pattern and shift the pixels using np.roll by the amount 'shift'
    deformed_pattern_interp = speckle.spline_interpolation_image(deformed_pattern, interp_y, interp_x, 3)
    deformed_pattern_interp = np.roll(deformed_pattern_interp,shift=shift,axis=1)
    spline = RectBivariateSpline(high_res_y, high_res_x, deformed_pattern_interp)
    deformed_pattern = spline(orig_y, orig_x)

    #-----------------------------------------------------------------------------------------------------------------------------
    # Stretching along rows (vertical direction) by a factor of 2
    #-----------------------------------------------------------------------------------------------------------------------------

    # deformed_image = zoom(speckle_pattern, (1, 2))
    # deformed_image = deformed_image[:,0:200]
    # arr = correlation.subset(speckle_pattern,x,y,subset_size)
    # ic(arr.shape)
    # ic(deformed_image.shape)


    #-----------------------------------------------------------------------------------------------------------------------------
    # Define our interpolator for the deformed image
    #-----------------------------------------------------------------------------------------------------------------------------

    interpolator = RectBivariateSpline(np.arange(0,width), np.arange(0,height), deformed_pattern, kx=3, ky=3)


    #-----------------------------------------------------------------------------------------------------------------------------
    # minimizer
    #-----------------------------------------------------------------------------------------------------------------------------
    p = np.array([2.5,2.5,0.0,0.0,0.0,0.0])  # Initial guess for affine parameters
    # lower_bounds = [0, 0, 0, 0, 0, 0]  # Replace with actual lower limits
    # upper_bounds = [2, 2, 0, 0, 0, 0]  # Replace with actual upper limits
    bounds = [(0, 4), (0,4), (0, 0), (0, 0), (0, 0), (0, 0)]  
    sol = minimize(subset_search_lm, p, args=(reference_subset,deformed_pattern,interpolator,x,y),bounds=bounds)
    # sol = minimize(subset_search_lm, p, args=(reference_subset,deformed_image,x,y),bounds=(lower_bounds,upper_bounds))



if __name__ == '__main__':
    main()



# OVERALL WORKFLOW FOR LM + AFFINE
# so everything needs to be done with coordinates, rather than np.ndarrays rectangular subsets
# initial guess I am going to assume is p = [0,0,0,0,0,0]
# extract subset from interpolated deformed image using p values -> this is where I'm going to have to update everything.
# calculate cost function (SSD, NSSD)
# LM checks for minimisation


# subset extraction WORKFLOW
# initial guess is perfectly square subset
# get subpixel values
# convert to array indices
# extract graylevel values for each array index
# pass to DIC

