# Speckle Pattern Generator

<!-- TOC -->

- [Speckle Pattern Generator](#speckle-pattern-generator)
  - [Overview](#overview)
  - [Installation](#installation)
  - [Basic Structure](#basic-structure)
  - [Key Classes and Methods](#key-classes-and-methods)
      - [Pattern Class](#pattern-class)
      - [Image Class](#image-class)
      - [Analysis Class](#analysis-class)
  - [Examples](#examples)
      - [example1\_generate\_and\_save\_pattern.py](#example1_generate_and_save_patternpy)
      - [example2\_find\_spacing.py](#example2_find_spacingpy)
      - [example3\_auto\_correlation\_peak\_sharpness.py](#example3_auto_correlation_peak_sharpnesspy)

<!-- /TOC -->

<!-- /TOC -->

## Overview
Simple Python package to create, visualise, and perform analysis on speckle patterns. Modules include tools for creating customizable speckle patterns, performing Fourier analysis, calculate speckle pattern metrics (Shannon Entropy, Auto-Correlation Peak Sharpness (ACPS) and Mean Intensity Gradient (MIG)), as well as visualisation/saving patterns.

---

## Installation
Clone the repository and install package:
```bash
git clone <repository-url>
cd speckle-generator
pip install .
```

## Basic Structure
The package is organized into the following modules:
* `pattern.py`: Defines the Pattern class for creating and manipulating speckle patterns.
* `image.py`: Contains the Image class for visualizing and saving speckle patterns.
* `analysis.py`: Implements the Analysis class for performing various analyses, including FFT, autocorrelation, and calculating statistical metrics.
* `grid.py`: methods used to generate the speckle pattern grid, randomly shift speckle locations and create cricular speckle masks.

A series of example scripts demonstrating usage can be found in `speckle/examples/`
```
speckle/
│
├── src/
│   ├── pattern.py   # Handles speckle pattern generation.
│   ├── image.py     # Manipulates and visualizes patterns.
│   ├── analysis.py  # Analytical tools for speckle evaluation.
│   ├── grid.py      # Methods for grid creation.
│
├── examples/
│   ├── example1_generate_and_save_pattern.py
│   ├── example2_find_spacing.py
│   ├── example3_auto_correlation_peak_sharpness.py
│
└── README.md        # Documentation.
```



## Key Classes and Methods

#### Pattern Class
Handles the creation of speckle patterns.

**Attributes**:
* `image_width (int), image_height (int)`: Pixel dimensions of the output pattern.
* `spacing (int)`: Mean distance between speckles.
* `variability (float)`: Maximum % variability in speckle displacement from a uniform grid.
* `mean_radius (int), stddev (float)`: Controls speckle radius distribution.

**Methods**:
* `generate() -> np.array`: Creates a speckle pattern.
* `rotate(angle: float) -> np.array`: Rotates the pattern.
* `statistics() -> None`: Prints pattern statistics (e.g., density, radius info).

#### Image Class
Visualisation and image saving of speckle pattern.

**Attributes**:
* `pattern (np.array)`: 2D speckle pattern to perform visualisation.

**Methods**:
* `balance(balance: float)`: Adjusts the black-and-white balance of the pattern.
* `invert(invert: bool)`: Inverts the pattern's intensity.
* `show()`: Displays the pattern using Matplotlib.
* `save(filename: str, resolution: int, format: str)`: Saves the pattern as an image.

#### Analysis Class
Performs some basic analysis / diagnostic on patterns.

**Attributes**:
* `pattern (np.array)`: 2D speckle pattern to perform analysis / diagnostics.

**Methods**:
* `fft()`: Computes the 2D FFT of the pattern.
* `fft_mag()`: Returns the magnitude of the FFT.
* `fft_gridspacing(spec: np.array)`: Estimates grid spacing using peaks in FFT spectrum.
* `fft_plotimage(spectrum: np.array, cmap: str)`: Using Matplotlib plot a 2d image of the FFT spectrum. Frequencies are shifted to (0,0). `cmap` takes colormap option.
* `fft_plotline(slice: np.array)`: Plots a 'slice' of the FFT spectrum. User must extract slice from the 2d array to pass to this method.
* `auto_correlation_peak_sharpness() -> float`: Calculates the sharpness of the autocorrelation peak in the speckle pattern.
* `mean_intensity_gradient() -> float`: Calculates the mean intensity gradient.

## Examples

#### example1_generate_and_save_pattern.py
Demonstrates basic functionality of the speckle package. Creates a speckle pattern with user specified params, adjusts its balance, inverts it, and saves it in `.tiff` format.
```python
import speckle

# Generate a Speckle pattern and print some basic statitics.
pattern = speckle.Pattern(image_width=512,image_height=512, mean_radius=4, spacing=10, variability=0.6,stddev=1.0)
speckle_pattern = pattern.generate()
pattern.statistics()

# Generate an image from the speckle pattern 
img = speckle.Image(speckle_pattern)

# Set the black white balance. Value between 0.0 and 1.0
img.balance(0.7)

# Invert the image
img.invert(invert=True)

#Display image using matplotlib and save as a lossless .tiff file
img.show()
img.save(filename="./example1.tiff",format="TIFF")
```

#### example2_find_spacing.py
Estimate the grid spacing of speckle patterns using FFT (Fast Fourier Transform). By varying the speckle spacing parameter, it shows how to extract and verify grid spacing from the frequency domain data. This is particularly useful for understanding how pattern generation parameters influence spatial frequency.
```python
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
```
#### example3_auto_correlation_peak_sharpness.py
Demonstrates the effect of speckle variability on the auto-correlation peak sharpness of speckle patterns. Iterates over different % variability ranges and then calculates the Auto-Correlation Peak Sharpness (ACPS).
```python
import matplotlib.pyplot as plt
import numpy as np
import speckle

# input array. Evenly spaced array betweeon 0.0 and 0.95 in steps of 0.05
variability_array = np.arange(0, 0.95, 0.05)

# our output array. Needs to be the same size as our input
acps = np.zeros(len(variability_array))

# Loop over our desired values of variability
for i,var in enumerate(variability_array):

    # generate a speckle pattern for each value of variability specified on line 6.
    pattern = speckle.Pattern(image_width=500,image_height=500, mean_radius=4, spacing=10, variability=var,stddev=0.0)
    speckle_pattern = pattern.generate()

    # calculate auto correlation peak sharpness (ACPS)
    analysis = speckle.Analysis(speckle_pattern)
    acps[i] = analysis.auto_correlation_peak_sharpness()
    print(acps[i])


# plot auto correlation peak sharpness as a function of speckle location variability
plt.plot(variability_array,acps)
plt.xlabel('Grid Pattern Variablility [%]')
plt.ylabel("Auto Correlation Peak Sharpness")
plt.show()
```