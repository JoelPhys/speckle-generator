import speckle

# Generate a Speckle pattern and print some basic statitics.
pattern = speckle.Pattern(image_width=2000,image_height=2000, mean_radius=6, spacing=12, variability=0.6,stddev_size=1.0,gray_level=256,stddev_smooth=1.0)
speckle_pattern = pattern.generate()
pattern.statistics()
pattern.levels_histogram()

# Generate an image from the speckle pattern 
img = speckle.Image(speckle_pattern,256)

# Set the black white balance. Value between 0.0 and 1.0
img.balance(1.0)

# Invert the image
img.invert(invert=True)

#Display image using matplotlib and save as a lossless .tiff file
img.show()
img.save(filename="./example1.tiff",format="TIFF")
