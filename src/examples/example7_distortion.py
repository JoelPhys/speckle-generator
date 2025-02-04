import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

import speckle

# Example Usage
# Generate a Speckle pattern and print some basic statitics.
pattern = speckle.Pattern(image_width=512,image_height=512, mean_radius=3, spacing=7, variability=0.6,stddev_size=0.4,gray_level=256,stddev_smooth=1.0,seed=1)
speckle_pattern = pattern.generate()

deformed_pattern = speckle.wave_distort(speckle_pattern, amplitude=0.1, frequency=5.0, axis=0)

# Plot results
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(speckle_pattern, cmap="gray")
plt.subplot(1, 2, 2)
plt.title("Wave Distorted Image")
plt.imshow(deformed_pattern, cmap="gray")
plt.show()

subset_size = 31
ref_centre_x = subset_size // 2 + subset_size-1
ref_centre_y = subset_size // 2 + subset_size-1
step=1

interp_image_ref = speckle.perform_interpolation(speckle_pattern,2,2,'cubic')
interp_image_def = speckle.perform_interpolation(deformed_pattern,2,2,'cubic')


subset_size = 21
min_x = subset_size // 2
min_y = subset_size // 2
max_x = speckle_pattern.shape[0] - subset_size // 2
max_y = speckle_pattern.shape[1] - subset_size // 2

for x in range(min_x,max_x,step):
    for y in range(min_y,max_y,step):

        ref_subset = speckle.subset(interp_image_ref, x, y, subset_size)
        u,v,ssd,ssd_map = speckle.correlation_global_map_opencv(ref_subset, interp_image_def)
        x_arr.append(x)
        y_arr.append(y)
        u_arr.append(u-x+min_x)
        v_arr.append(v-y+min_y)

time_end_loop = time.perf_counter()
duration2= time_end_loop - time_start_setup
