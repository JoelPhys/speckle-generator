import speckle
import matplotlib.pyplot as plt
import numpy as np

variability_array = np.arange(0, 0.95, 0.05)
acps = np.zeros(len(variability_array))

for i,var in enumerate(variability_array):

    # generate a speckle pattern
    pattern = speckle.Pattern(image_width=500,image_height=500, mean_radius=4, spacing=10, variability=var,stddev=0.0)
    speckle_pattern = pattern.generate()
    
    # calculate auto correlation peak sharpness (acps)
    analysis = speckle.Analysis(speckle_pattern)
    acps[i] = analysis.auto_correlation_peak_sharpness()
    print(acps[i])

# plot auto correlation peak sharpness as a function of speckle location variability
plt.plot(variability_array,acps)
plt.xlabel('Grid Pattern Variablility [%]')
plt.ylabel("Auto Correlation Peak Sharpness")
plt.show()
