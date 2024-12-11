import speckle

# print(speckle_joel.__version__)
print("test")

generate = speckle.Generate(image_width=4000,image_height=1000, radius=5, variability=0.3)
speckle_pattern = generate.pattern()

image = speckle.Image(speckle_pattern)
image.balance(0.5)
image.invert(True)
image.show()
image.save("./test.tiff",format="TIFF",resolution=300)

fft_analysis = speckle.Analysis(pattern=speckle_pattern)
fft_analysis.fft()
fft_analysis.fftplot()