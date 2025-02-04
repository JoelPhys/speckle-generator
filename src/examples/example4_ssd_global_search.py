import matplotlib.patches as patches
import matplotlib.pyplot as plt
import time
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
deformed_pattern[50:101,50:101] = speckle_pattern[400:451,300:351]
deformed_pattern[400:451,300:351] = speckle_pattern[50:101,50:101]

deformed_img = speckle.Image(deformed_pattern,256)
deformed_img.balance(1.0)
deformed_img.invert(invert=True)
deformed_img.show()


# step = 5


subset_size = 51
min_x = subset_size // 2
min_y = subset_size // 2
max_x = speckle_pattern.shape[0] - subset_size // 2
max_y = speckle_pattern.shape[1] - subset_size // 2


#specify local subset
ref_subset = speckle.subset(speckle_pattern, 75, 75, subset_size)

# generate correlation map
time_start_setup = time.perf_counter()
ssd_opencv = speckle.correlation_global_map_opencv(ref_subset, deformed_pattern)
time_end_loop = time.perf_counter()
duration1 = time_end_loop - time_start_setup

time_start_setup = time.perf_counter()
ssd_manual = speckle.correlation_global_map(ref_subset, deformed_pattern)
time_end_loop = time.perf_counter()
duration2= time_end_loop - time_start_setup

print("Time taken for OpenCV SSD Correlation:", duration1, "[s]")
print("Time taken for manual SSD Correlation:", duration2, "[s]")

ssd_diff = ssd_manual - ssd_opencv

plt.figure(figsize=(10, 5)) 
plt.subplot(1, 3, 1)
plt.imshow(ssd_opencv)
plt.colorbar()
plt.xlabel("u")
plt.ylabel("v")
plt.title("SSD OpenCV"
plt.subplot(1, 3, 2)
plt.imshow(ssd_manual)
plt.colorbar()
plt.xlabel("u")
plt.ylabel("v")
plt.title("SSD Manual")
plt.subplot(1, 3, 3)
plt.imshow(ssd_diff)
plt.colorbar()
plt.xlabel("u")
plt.ylabel("v")
plt.title("SSD Manual - SSD OpenCV")


plt.tight_layout()  # Adjust layout for better spacing
plt.show()
