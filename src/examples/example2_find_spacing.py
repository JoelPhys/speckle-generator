import speckle
import matplotlib.pyplot as plt
import numpy as np



spacing_array = [10,15,20,25,30]
spacing_fft = np.zeros(len(spacing_array))

for i, spacing in enumerate(spacing_array):

    pattern = speckle.Pattern(image_width=1000,image_height=1000, mean_radius=5, spacing=spacing, variability=0.2,stddev=0.0)
    speckle_pattern = pattern.generate()

    # Perform FFT
    analysis = speckle.Analysis(speckle_pattern)
    fft_spectrum = analysis.fft()
    fft_magnitude  = analysis.fft_mag()

    # FFT spectrum along y=0 horizontal.
    horizontal = fft_magnitude[:,1000//2]
    spacing_fft[i] = analysis.fft_gridspacing(horizontal)
    print("grid spacing calculated from FFT:", spacing_fft[i])



# Fit to a linear equation to find gradient and intercept.
# Should be: y(x) ~ 1.0*x + 0.0
gradient, intercept = np.polyfit(spacing_array, spacing_fft, 1)

# figure
plt.scatter(spacing_array,spacing_fft)
plt.title("gradient: " + f"{gradient:.2f}" + ", intercept: " + f"{intercept:.2f}")
plt.xlabel("User Defined Spacing [Pixels]")
plt.ylabel("FFT Determined Spacing [Pixels]")
plt.show()
