import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline
import sys
import time
import cv2 as cv
from icecream import ic

import speckle

# Example Usage
# Generate a Speckle pattern and print some basic statitics.
width = 400
height = 400
pattern = speckle.Pattern(image_width=width,image_height=height, mean_radius=3, spacing=7, variability=0.6,stddev_size=0.4,gray_level=256,stddev_smooth=1.0,seed=1)
speckle_pattern = pattern.generate()



# --------------------------------------------------------------------------------------
# Generate the deformed image
# --------------------------------------------------------------------------------------

# copy the original pattern
deformed_pattern = speckle_pattern.copy()

#create interpolation
interp_x = interp_y = 4
subpx_samp = 1.0/interp_x
shift = 1

# pixel values of original images
orig_x = np.arange(speckle_pattern.shape[1])
orig_y = np.arange(speckle_pattern.shape[0])

# Pixel values of interpolated image
high_res_x = np.linspace(0, speckle_pattern.shape[1] - 1, (speckle_pattern.shape[1] - 1) * interp_x + 1)
high_res_y = np.linspace(0, speckle_pattern.shape[0] - 1, (speckle_pattern.shape[0] - 1) * interp_y + 1)

# interpolate the deformed pattern and shift the pixels using np.roll by the amount 'shift'
deformed_pattern_interp = speckle.spline_interpolation(deformed_pattern, interp_y, interp_x, 3)
deformed_pattern_interp_shift = np.roll(deformed_pattern_interp,shift=shift, axis=1)

# convert back to dimensions of original deformed_pattern
spline = RectBivariateSpline(high_res_y, high_res_x, deformed_pattern_interp_shift)
deformed_pattern = spline(orig_y, orig_x)


# --------------------------------------------------------------------------------------
# Plot a line segment from the reference and deformed image for a comparison
# --------------------------------------------------------------------------------------

x = np.arange(0.0,height,1.0)
x1 = np.arange(0.0,height-0.99,subpx_samp)
plt.figure()
plt.plot(x1,deformed_pattern_interp[0,:],'-s',color="red")
plt.plot(x,speckle_pattern[0,:],'-o',color="blue")
plt.xlabel("Pixel Value")
plt.ylabel("Graylevel")
plt.show()

u_arr, v_arr = speckle.dic_local_spline_interpolation(reference_image=speckle_pattern, deformed_image=deformed_pattern, subset_size=21, subset_step=10, roi_radius=3, interpolation=3, corr_crit="ssd")
u_arr, v_arr = speckle.dic_global_spline_interpolation(reference_image=speckle_pattern, deformed_image=deformed_pattern, subset_size=21, subset_step=10, roi_radius=3, interpolation=3, corr_crit="ssd")


#---------------------------------------------------------------------------------------
# Plot results
# --------------------------------------------------------------------------------------
plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(speckle_pattern, cmap="gray")
plt.subplot(1, 3, 2)
plt.title("Wave Distorted Image")
plt.imshow(deformed_pattern, cmap="gray")
plt.subplot(1, 3, 3)
plt.title("Displacement, u")
plt.imshow(u_arr, cmap="gray", vmin=-10, vmax=10)
plt.colorbar()
plt.show()