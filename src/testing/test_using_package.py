import speckle

generate = speckle.Generate(image_width=500,image_height=500, mean_radius=4, spacing=20, variability=0.2,stddev= 0.1)
speckle_pattern = generate.pattern()
stats = generate.statistics(speckle_pattern)
print(stats)

fft_analysis = speckle.Analysis()
fft_pattern = fft_analysis.fft(speckle_pattern)
fft_mag  = fft_analysis.fft_mag(fft_pattern)
fft_analysis.fftplot(fft_mag)