import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from icecream import ic

import speckle


def wave_distort(image, amplitude=10, frequency=2*np.pi, axis=0):
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

# Example Usage
# Generate a Speckle pattern and print some basic statitics.
pattern = speckle.Pattern(image_width=200,image_height=200, mean_radius=3, spacing=7, variability=0.6,stddev_size=0.4,gray_level=256,stddev_smooth=1.0,seed=1)
speckle_pattern = pattern.generate()

def_pattern = wave_distort(speckle_pattern, amplitude=2.0, frequency=2.0, axis=0)
ic(def_pattern.shape)

# Plot results
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(speckle_pattern, cmap="gray")
plt.subplot(1, 2, 2)
plt.title("Wave Distorted Image")
plt.imshow(def_pattern, cmap="gray")
plt.show()

subset_size = 31
correlation = speckle.Correlation(image_ref=speckle_pattern, image_def=def_pattern,subset_size=subset_size)
ref_centre_x = subset_size // 2 + subset_size-1
ref_centre_y = subset_size // 2 + subset_size-1
step=1

interped_ref_image, interped_def_image = correlation.perform_interpolation(2,2,'cubic')
# ic(interped_ref_image.shape)


subset_size = 51
correlation = speckle.Correlation(image_ref=speckle_pattern, image_def=deformed_pattern,subset_size=subset_size)
min_x = subset_size // 2
min_y = subset_size // 2
max_x = speckle_pattern.shape[0] - subset_size // 2
max_y = speckle_pattern.shape[1] - subset_size // 2


# correlation.perform_interpolation(4,4,'linear')

ref_subset = correlation.subset(speckle_pattern, 75, 75)
u,v,ssd = correlation.global_search(ref_subset, deformed_pattern)
print(u,v,ssd)
