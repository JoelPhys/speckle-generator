import matplotlib.patches as patches
import matplotlib.pyplot as plt

import speckle

# Generate a Speckle pattern and print some basic statitics.
subset_size = 21
width = subset_size * 10
height= subset_size * 10


pattern = speckle.Pattern(image_width=width,image_height=height, mean_radius=3, spacing=8, variability=0.6,stddev_size=1.0,gray_level=256,stddev_smooth=1.0,seed=1)
speckle_pattern = pattern.generate()
# Generate an image for the reference 
pattern_img = speckle.Image(speckle_pattern,256)
pattern_img.balance(1.0)
pattern_img.invert(invert=True)
pattern_img.show()

# ---------------------------------------------------------------------------
# take a arbitraty subset and change its location in the deformed image
deformed_pattern = speckle_pattern.copy()
deformed_pattern[0:subset_size*4,:] = speckle_pattern[(width-subset_size*4):,:]
deformed_pattern[(width-subset_size*6):,:] = speckle_pattern[0:subset_size*6,:]


# ---------------------------------------------------------------------------
# view deformed image
deformed_img = speckle.Image(deformed_pattern,256)
deformed_img.balance(1.0)
deformed_img.invert(invert=True)
deformed_img.show()

# ---------------------------------------------------------------------------
# define our correlation object. Specify our initial, and deformated images as well as subset size.
min_x = subset_size // 2
min_y = subset_size // 2
max_x = speckle_pattern.shape[0] - subset_size // 2
max_y = speckle_pattern.shape[1] - subset_size // 2


x_arr = []
y_arr = []
u_arr = []
v_arr = []

# ---------------------------------------------------------------------------
# Loop over all our reference subsets and calculate the with a step size specified below
# find the minimum value in the ssd using a global search of all deformed subsets.
step = subset_size
for x in range(min_x,max_x,step):
    for y in range(min_y,max_y,step):

        ref_subset = speckle.subset(speckle_pattern, x, y, subset_size)
        ssd_map = speckle.correlation_global_map(ref_subset, deformed_pattern)
        u,v,ssd = speckle.correlation_global_find_min(ssd_map)
        print(x,y,u,v,ssd)
        x_arr.append(x)
        y_arr.append(y)
        u_arr.append(u-x)
        v_arr.append(v-y)
        


# Creating plot
plt.figure()
fig, ax = plt.subplots(figsize = (12, 7))
ax.quiver(x_arr, y_arr, u_arr, v_arr)
plt.show()



