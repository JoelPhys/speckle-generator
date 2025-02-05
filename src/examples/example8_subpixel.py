import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline
import sys
import time
import cv2 as cv

import speckle

# Example Usage
# Generate a Speckle pattern and print some basic statitics.
pattern = speckle.Pattern(image_width=200,image_height=200, mean_radius=3, spacing=7, variability=0.6,stddev_size=0.4,gray_level=256,stddev_smooth=1.0,seed=1)
speckle_pattern = pattern.generate()



# --------------------------------------------------------------------------------------
# Generate the deformed image
# --------------------------------------------------------------------------------------


#deformed_pattern = speckle.wave_distort(speckle_pattern, amplitude=5.0, frequency=5.0, axis=0)

# copy the original pattern
deformed_pattern = speckle_pattern.copy()

#create interpolation
interp_x = interp_y = 4
shift = 5
orig_x = np.arange(speckle_pattern.shape[1])
orig_y = np.arange(speckle_pattern.shape[0])
high_res_x = np.linspace(0, speckle_pattern.shape[1] - 1, (speckle_pattern.shape[1] - 1) * interp_x + 1)
high_res_y = np.linspace(0, speckle_pattern.shape[0] - 1, (speckle_pattern.shape[0] - 1) * interp_y + 1)

deformed_pattern_interp = speckle.spline_interpolation(deformed_pattern, interp_y, interp_x, 3)
deformed_pattern_interp_shift = np.roll(deformed_pattern_interp,shift=shift, axis=1)
spline = RectBivariateSpline(high_res_y, high_res_x, deformed_pattern_interp_shift)
deformed_pattern_resampled = spline(orig_y, orig_x)


# --------------------------------------------------------------------------------------
# Plot reference and deformed image
# --------------------------------------------------------------------------------------
#plt.figure(figsize=(10, 5))
#plt.subplot(1, 2, 1)
#plt.title("Original Image")
#plt.imshow(deformed_pattern_interp, cmap="gray")
#plt.subplot(1, 2, 2)
#plt.title("Wave Distorted Image")
#plt.imshow(deformed_pattern_interp_shift, cmap="gray")
#plt.show()

x = np.arange(0.0,200.0,1.0)
x1 = np.arange(0.0,199.01,0.25)
plt.figure()
plt.plot(x,deformed_pattern[0,:],'--o')
plt.plot(x,deformed_pattern_resampled[0,:],'--s')
#plt.plot(x1,deformed_pattern_interp[0,:],'--D')
#plt.plot(x1,deformed_pattern_interp_shift[0,:],'--v')
plt.show()

edge_cutoff = 10
step = 10 # pixels
subset_size = 21 # pixels
min_x = subset_size // 2 
min_y = subset_size // 2 
max_x = speckle_pattern.shape[0] - subset_size // 2
max_y = speckle_pattern.shape[1] - subset_size // 2

# --------------------------------------------------------------------------------------
# get dimensions of output arrays
# --------------------------------------------------------------------------------------
x_values = np.arange(min_x, max_x, step)
y_values = np.arange(min_y, max_y, step)
shape = (len(y_values), len(x_values)) 

total_iterations = x_values.shape[0] * y_values.shape[0]

# Initialize 2D arrays
u_arr = np.zeros(shape)
v_arr = np.zeros(shape)


# --------------------------------------------------------------------------------------
# loop over subsets
# --------------------------------------------------------------------------------------
time_start_loop = time.perf_counter()
progress = 0

ref_subset = speckle.subset(speckle_pattern, 100, 100, subset_size)
u,v,ssd,ssd_map = speckle.correlation_global_map_opencv(ref_subset, deformed_pattern_resampled,"ssd")
ssd_map_interp = speckle.spline_interpolation(ssd_map, interp_y, interp_x, 3)
min_val, max_val, min_loc, max_loc = cv.minMaxLoc(ssd_map_interp)
#ssd_map = speckle.correlation_global_map(ref_subset, deformed_pattern_resampled)
#u,v,ssd_min = speckle.correlation_global_find_min(ssd_map)



print(min_loc[0]/4.0 - 100.0 + float(min_x))

plt.figure()
plt.imshow(ssd_map)
plt.colorbar()
plt.show()

plt.figure()
plt.plot(ssd_map_interp[89*4,:])
plt.plot(ssd_map_interp[90*4,:])
plt.plot(ssd_map_interp[91*4,:])
plt.show()
u = u - float(100) + float(min_x)  
v = v - float(100) + float(min_y)
print(u,v)
exit()

 


for i, x in enumerate(x_values):
    for j, y in enumerate(y_values):
        
        ref_subset = speckle.subset(speckle_pattern, x, y, subset_size)
        u, v, ssd, ssd_map = speckle.correlation_global_map_opencv(ref_subset, deformed_pattern_interp_shift, "ssd")

        u_arr[j, i] = u/4.0 - float(x) + float(min_x)  
        v_arr[j, i] = v/4.0 - float(y) + float(min_y)


        # Update progress
        progress += 1
        progress_percentage = int((progress / total_iterations) * 100)
        bar_length = 50
        filled_length = int(bar_length * progress_percentage / 100)
        bar = '=' * filled_length + '-' * (bar_length - filled_length)

        # Print progress bar
        sys.stdout.write(f'\r[{bar}] {progress_percentage}%')
        sys.stdout.flush()


time_end_loop = time.perf_counter()
duration2= time_end_loop - time_start_loop

# --------------------------------------------------------------------------------------
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


