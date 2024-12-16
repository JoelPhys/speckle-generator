# Speckle Pattern Generator

<!-- TOC -->

- [Speckle Pattern Generator](#speckle-pattern-generator)
  - [Overview](#overview)
  - [Installation](#installation)
  - [Structure](#structure)
  - [Usage](#usage)
    - [Generate a Speckle Pattern](#generate-a-speckle-pattern)
    - [Speckle Pattern Analysis](#speckle-pattern-analysis)
    - [Manipulate and Save the Pattern](#manipulate-and-save-the-pattern)

<!-- /TOC -->late and Save the Pattern](#manipulate-and-save-the-pattern)

<!-- /TOC -->

## Overview
The **Speckle Pattern Generator** Python package provides tools to generate, analyze, and manipulate speckle patterns for simulations or image processing tasks. It includes modules for creating customizable speckle patterns, performing Fourier analysis, and visualizing or saving the resulting images.

---

## Installation
Clone the repository and install the required dependencies:
```bash
git clone <repository-url>
cd speckle-pattern-generator
pip install -r requirements.txt
```

## Structure
The package is organized into the following modules:
* `generate.py`: Generates speckle patterns with customizable parameters.
* `analysis.py`: Provides tools for Fourier Transform analysis of speckle patterns.
* `image.py`: Offers utilities for visualizing, saving, and manipulating generated patterns.

## Usage
The `Generate` class creates speckle patterns with specified size, spacing, and variability. The class takes the following parameters:

* `image_height (int)`: Height of the speckle image in pixels.
* `image_width (int)`: Width of the speckle image in pixels.
* `spacing (int)`: Average spacing between speckles.
* `mean_radius (int, optional)`: Mean radius of speckles (default: 4).
* `stddev (float, optional)`: Standard deviation of speckle radius (default: 1).
* `variability (float, optional)`: Variability in speckle placement (default: 0.4).
### Generate a Speckle Pattern
```python
from speckle.generate import Generate
import matplotlib.pyplot as plt

# Create speckle pattern generator
generator = Generate(image_height=512, image_width=512, spacing=50, mean_radius=4, stddev=1, variability=0.4)

# Generate the speckle pattern
pattern = generator.pattern()

# Show statistics
print(generator.statistics(pattern))

# Visualize the pattern
plt.imshow(pattern, cmap='gray')
plt.title('Speckle Pattern')
plt.show()
```

### Speckle Pattern Analysis
The `Analysis` class enables fast fourier transforms of speckle patterns. The class has the following methods:
* `fft(pattern: np.array)`: Computes the 2D Fast Fourier Transform (FFT) of the speckle pattern.
* `fft_mag(fft_pattern: np.array(complex))`: Calculates the magnitude spectrum of the FFT result.
* `fftplot(fft_pattern: np.array)`: Visualizes the magnitude spectrum in logarithmic scale.

An example using the class:

```python
from speckle.analysis import Analysis

analyzer = Analysis()

# Perform FFT and magnitude spectrum
fft_result = analyzer.fft(pattern)
magnitude_spectrum = analyzer.fft_mag(fft_result)

# Plot the FFT magnitude spectrum
analyzer.fftplot(magnitude_spectrum)
```

### Manipulate and Save the Pattern
The `Image` class provides utilities to adjust, invert, display, and save speckle patterns. THe class contains four methods:

* `balance(balance: float (0.0 - 1.0))`: Adjusts the intensity of the pattern.
* `invert(invert: bool)`: Inverts the pattern colors (white ↔ black).
* `show()`: Displays the speckle pattern using matplotlib.pyplot
* `save(filename: str, format: str, resolution: int (dpi))`: Saves the speckle pattern as an image.

```Python
from speckle.image import Image

# Initialize with speckle pattern
img = Image(pattern)

# Apply adjustments
img.balance(balance=0.8)
img.invert(invert=True)

# Show the pattern
img.show()

# Save the pattern
img.save(filename='speckle.tiff', format='TIFF', resolution=300)
```

