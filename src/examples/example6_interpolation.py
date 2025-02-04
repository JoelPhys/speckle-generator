
import numpy as np
import matplotlib.pyplot as plt
from icecream import ic

import speckle


# Example Usage
# Generate a Speckle pattern and print some basic statitics.
width = 512
height = 512
pattern = speckle.Pattern(image_width=width,image_height=height, mean_radius=3, spacing=8, variability=0.6,stddev_size=0.4,gray_level=256,stddev_smooth=1.0,seed=100)
speckle_pattern = pattern.generate()

linear_ref = speckle.perform_interpolation(speckle_pattern,4,4,'linear')
slinear_ref = speckle.perform_interpolation(speckle_pattern,4,4,'slinear')
cubic_ref = speckle.perform_interpolation(speckle_pattern,4,4,'cubic')
quintic_ref = speckle.perform_interpolation(speckle_pattern,4,4,'quintic')



x  = np.linspace(0, width - 1, width)
x_vals = np.linspace(0, width - 1, (width - 1) * 4 + 1)

plt.figure()
plt.plot(x,speckle_pattern[0,:],'-o', linewidth=4,markersize=12,color="black",label="Original")
plt.plot(x_vals,linear_ref[0,:],'--*', label="linear")
plt.plot(x_vals,slinear_ref[0,:],'--x', label="slinear")
plt.plot(x_vals,cubic_ref[0,:],'--s', label="cubic")
plt.plot(x_vals,quintic_ref[0,:],'--p', label="quintic")
plt.ylabel("Gray Level")
plt.xlabel("Pixel")
plt.xlim([0, 20])
plt.legend()
plt.show()
