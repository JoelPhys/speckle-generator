import numpy as np
import matplotlib.pyplot as plt


class Analysis:

    def __init__(self):
        self.pattern = None

    def fft(self, pattern: np.array) -> np.array:
        fft = np.fft.fftshift(np.fft.fft2(pattern))
        return fft
    
    def fft_mag(self, fft: np.array) -> np.array:
        mag_spec = np.abs(fft)
        return mag_spec

    def fftplot(self, mag_spec: np.array) -> None:        

        freq_x = np.fft.fftshift(np.fft.fftfreq(mag_spec.shape[0]))
        freq_y = np.fft.fftshift(np.fft.fftfreq(mag_spec.shape[1]))
        
        plt.figure()
        plt.title('FFT Analysis of Speckle Pattern')
        plt.imshow(np.log(1 + mag_spec), cmap='viridis',extent=(freq_x[0], freq_x[-1], freq_y[0], freq_y[-1]))
        plt.colorbar(label='Magnitude')
        plt.show()