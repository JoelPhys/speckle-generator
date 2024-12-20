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
