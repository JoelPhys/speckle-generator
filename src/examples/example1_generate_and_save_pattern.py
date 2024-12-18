import speckle

# Generate a Speckle pattern and print some basic statitics.
pattern = speckle.Pattern(image_width=500,image_height=500, mean_radius=4, spacing=10, variability=0.6,stddev=1.0)
speckle_pattern = pattern.generate()
pattern.statistics()


# speckle pattern image
img = speckle.Image(speckle_pattern)
img.balance(0.6)
img.invert(invert=True)
img.show()
img.save(filename="./example1.tiff",format="TIFF", resolution=300)
