import matplotlib.pyplot as plt
import numpy as np
import speckle


# our input array
spacing_array = [10,15,20,25,30]

# our Output array
spacing_fft = np.zeros(len(spacing_array))


# Loop over spacing values specified in spacing_array
for i, spacing in enumerate(spacing_array):


    # generate speckle pattern with a small amount of variability
    pattern = speckle.Pattern(image_width=1000,image_height=1000, mean_radius=5, spacing=spacing, variability=0.2,stddev=0.0)
    speckle_pattern = pattern.generate()



    # Perform FFT
    analysis = speckle.Analysis(speckle_pattern)
    fft_spectrum = analysis.fft()
    fft_magnitude  = analysis.fft_mag()



    # Use the FFT spectrum along y=0 horizontal to calculate grid spacing.
    horizontal = fft_magnitude[:,1000//2]
    spacing_fft[i] = analysis.fft_gridspacing(horizontal)
    print("grid spacing calculated from FFT:", spacing_fft[i])



# Fit to a linear equation to find gradient and intercept.
# Should be: y(x) ~ 1.0*x + 0.0
gradient, intercept = np.polyfit(spacing_array, spacing_fft, 1)



# Scatter plot of grid spacing as calculated from the FFT peaks as a function of the actual user specified speckle spacing
plt.scatter(spacing_array,spacing_fft)
plt.title("gradient: " + f"{gradient:.2f}" + ", intercept: " + f"{intercept:.2f}")
plt.xlabel("User Defined Spacing [Pixels]")
plt.ylabel("FFT Determined Spacing [Pixels]")
plt.show()
