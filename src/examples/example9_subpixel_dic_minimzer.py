
import numpy as np
from icecream import ic
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from scipy.optimize import least_squares, minimize
from scipy.interpolate import RegularGridInterpolator, RectBivariateSpline

import speckle

width = 400
height = 400
pattern = speckle.Pattern(image_width=width,image_height=height, mean_radius=3, spacing=7, variability=0.6,stddev_size=0.1,gray_level=256,stddev_smooth=1.0,seed=1)
speckle_pattern = pattern.generate()


#-----------------------------------------------------------------------------------------------------------------------------
# subpixel roll
#-----------------------------------------------------------------------------------------------------------------------------


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
deformed_pattern_interp = speckle.correlation.spline_interpolation_image(deformed_pattern, interp_y, interp_x, 3)

split = deformed_pattern_interp.shape[0]


# Number of segments
num_segments = 4
shifts = [0, 1, 2, 3]  
segment_size = deformed_pattern_interp.shape[0] // num_segments

# Apply the shifts in a loop
for i in range(num_segments):
    start = i * segment_size
    end = (i + 1) * segment_size if i < num_segments - 1 else None  # Ensure last segment includes all remaining rows
    deformed_pattern_interp[start:end, :] = np.roll(deformed_pattern_interp[start:end, :], shift=shifts[i], axis=1)

# convert back to dimensions of original deformed_pattern
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
# DIC
#-----------------------------------------------------------------------------------------------------------------------------
bounds = [(-4, 4), (0,0), (0, 0), (0, 0), (0, 0), (0, 0)]  
u_arr, v_arr = speckle.dic.reference_image_interpolation_minimizer(reference_image=speckle_pattern, deformed_image=deformed_pattern, bounds=bounds, subset_size=21, subset_step=10, interpolation=interp_x, corr_crit="ssd")
u_arr, v_arr = speckle.dic.global_spline_interpolation(reference_image=speckle_pattern, deformed_image=deformed_pattern, subset_size=21, subset_step=10, interpolation=interp_x, corr_crit="ssd")

#---------------------------------------------------------------------------------------
# Plot results
# --------------------------------------------------------------------------------------
plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(speckle_pattern, cmap="gray",vmin=0, vmax=255)
plt.xlabel("Pixel")
plt.ylabel("Pixel")
plt.subplot(1, 3, 2)
plt.title("Distorted Image")
plt.imshow(deformed_pattern, cmap="gray",vmin=0, vmax=255)
plt.xlabel("Pixel")
plt.ylabel("Pixel")
plt.subplot(1, 3, 3)
plt.title("Displacement, u")
plt.imshow(u_arr, cmap="viridis",vmin=0,vmax=2.0)
plt.xlabel("subset")
plt.ylabel("subset")
plt.colorbar(label="Pixel shift")
plt.show()
