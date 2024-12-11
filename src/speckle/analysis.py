import numpy as np
import matplotlib.pyplot as plt


class Analysis:

    def __init__(self,pattern: np.array):
        self.pattern = pattern
        self.fft: np.array
        self.magnitude_spectrum: np.array


    def fft(self) -> None:
        self.fft = np.fft.fftshift(np.fft.fft2(self.pattern))
        self.magnitude_spectrum = np.abs(self.fft)

    def fftplot(self) -> None:
        if self.magnitude_spectrum is None:
            raise ValueError("FFT analysis has not been performed. Please run analyze() first.")
        
        plt.figure()
        plt.title('FFT Analysis of Speckle Pattern')
        plt.imshow(np.log(1 + self.magnitude_spectrum), cmap='viridis')
        plt.colorbar(label='Magnitude')
        plt.show()