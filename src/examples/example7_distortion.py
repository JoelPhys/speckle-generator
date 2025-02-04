import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
import sys
import time

import speckle

# Example Usage
# Generate a Speckle pattern and print some basic statitics.
pattern = speckle.Pattern(image_width=1024,image_height=1024, mean_radius=3, spacing=7, variability=0.6,stddev_size=0.4,gray_level=256,stddev_smooth=1.0,seed=1)
speckle_pattern = pattern.generate()

deformed_pattern = speckle.wave_distort(speckle_pattern, amplitude=5.0, frequency=5.0, axis=0)

subset_size = 31
ref_centre_x = subset_size // 2 + subset_size-1
ref_centre_y = subset_size // 2 + subset_size-1
step=10


subset_size = 21
min_x = subset_size // 2
min_y = subset_size // 2
max_x = speckle_pattern.shape[0]- subset_size // 2
max_y = speckle_pattern.shape[1] - subset_size // 2

x_values = np.arange(min_x, max_x, step)
y_values = np.arange(min_y, max_y, step)
shape = (len(y_values), len(x_values)) 

total_iterations = x_values.shape[0] * y_values.shape[0]

# Initialize 2D arrays
u_arr = np.zeros(shape)
v_arr = np.zeros(shape)

progress = 0

time_start_loop = time.perf_counter()

for i, x in enumerate(x_values):
    for j, y in enumerate(y_values):
        
        ref_subset = speckle.subset(speckle_pattern, x, y, subset_size)
        u, v, ssd, ssd_map = speckle.correlation_global_map_opencv(ref_subset, deformed_pattern, "ssd")

        u_arr[j, i] = u - x + min_x  
        v_arr[j, i] = v - y + min_y


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

# Plot results
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
plt.show()


