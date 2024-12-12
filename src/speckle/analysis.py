import numpy as np
import matplotlib.pyplot as plt


class Analysis:

    def __init__(self,pattern: np.array):
        self.pattern = pattern

    def fft(pattern: np.array) -> np.array:
        fft = np.fft.fftshift(np.fft.fft2(self.pattern))
        return fft
    
    def fft_mag(self) -> np.array:
        magnitude_spectrum = np.abs(self.fft)
        return magnitude_spectrum

    def fftplot(magnitude_spectrum):        
        plt.figure()
        plt.title('FFT Analysis of Speckle Pattern')
        plt.imshow(np.log(1 + magnitude_spectrum), cmap='viridis')
        plt.colorbar(label='Magnitude')
        plt.show()