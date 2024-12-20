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
