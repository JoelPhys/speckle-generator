import matplotlib.patches as patches
import matplotlib.pyplot as plt

import speckle

# Generate a Speckle pattern and print some basic statitics.
pattern = speckle.Pattern(image_width=512,image_height=512, mean_radius=3, spacing=8, variability=0.6,stddev_size=1.0,gray_level=256,stddev_smooth=1.0,seed=1)
speckle_pattern = pattern.generate()
# Generate an image for the reference 
pattern_img = speckle.Image(speckle_pattern,256)
pattern_img.balance(1.0)
pattern_img.invert(invert=True)
pattern_img.show()

# ---------------------------------------------------------------------------
# take a arbitraty subset and change its location in the deformed image
deformed_pattern = speckle_pattern.copy()
deformed_pattern[50:101,50:101] = speckle_pattern[400:451,400:451]
deformed_pattern[400:451,400:451] = speckle_pattern[50:101,50:101]

deformed_img = speckle.Image(deformed_pattern,256)
deformed_img.balance(1.0)
deformed_img.invert(invert=True)
deformed_img.show()


# step = 5


subset_size = 51
correlation = speckle.Correlation(image_ref=speckle_pattern, image_def=deformed_pattern,subset_size=subset_size)
min_x = subset_size // 2
min_y = subset_size // 2
max_x = speckle_pattern.shape[0] - subset_size // 2
max_y = speckle_pattern.shape[1] - subset_size // 2


# correlation.perform_interpolation(4,4,'linear')

ref_subset = correlation.subset(speckle_pattern, 75, 75)
u,v,ssd = correlation.global_search_loops(ref_subset, deformed_pattern)
print(u,v,ssd)





