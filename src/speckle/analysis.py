import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy import signal

class Analysis:

    def __init__(self,pattern: np.array):
        self.pattern = pattern



    def fft(self) -> np.array:
        """ 
        return frequency shifted 2D fft of speckle pattern 
        """

        fft = np.fft.fftshift(np.fft.fft2(self.pattern))

        return fft



    def fft_mag(self) -> np.array:
        """ 
        returns frequency shifted fft magnitude of the fft array returned from Analysis.fft()
        """
        
        mag_spec = np.abs(np.fft.fftshift(np.fft.fft2(self.pattern)))
        
        return mag_spec



    def fft_plotimage(self, spectrum: np.array, cmap: str = "viridis") -> None:     
        """ 
        image plot of FFT pattern. Displays correct frequency ranges. Viridis color map. 
        """

        # Get frequency values from array dimensions.
        freq_x = np.fft.fftshift(np.fft.fftfreq(spectrum.shape[0]))
        freq_y = np.fft.fftshift(np.fft.fftfreq(spectrum.shape[1]))
        freq_range = [freq_x[0], freq_x[-1], freq_y[0], freq_y[-1]]

        # image plot of spectrum magnitude.
        plt.figure()
        plt.title('FFT Analysis of Speckle Pattern')
        plt.imshow(np.log(1 + spectrum), cmap=cmap,extent=freq_range)
        plt.colorbar(label='Magnitude')
        plt.show()

        return None


    def fft_plotline(self, slice: np.array) -> None:
        """ 
        line plot of FFT pattern along required direction. User provide 1D array 
        """

        # Get frequency values from array dimensions.
        freq_x = np.fft.fftshift(np.fft.fftfreq(slice.shape[0]))

        # Line plot along 1D path specified by user.
        plt.figure()
        plt.title('FFT Analysis of Speckle Pattern')
        plt.xlabel("Frequency")
        plt.ylabel("Amplitude")
        plt.plot(freq_x,slice)
        plt.show()

        return None



    def fft_gridspacing(self, spec: np.array, prominence=(0.3,1.0)) -> None:
        """ 
        estimate the grid spacing from the fft spectrum.
        User provides one-sided array. 
        """

        freq_x = np.fft.fftfreq(spec.size)

        one_sided = np.flip(spec[:(spec.size//2)])
        one_sided_normed = one_sided / np.max(one_sided)

        peaks, properties = signal.find_peaks(one_sided_normed, prominence=prominence)

        spacing = 1.0 / freq_x[peaks[0]]

        return spacing





    def mean_intensity_gradient(self) -> float:
        """ 
        Mean Intensity Gradient. Based on the below: 
        https://www.sciencedirect.com/science/article/abs/pii/S0143816613001103 
        """

        # calc grad along each direction. Easy to just use sobel filter.
        gradient_x = ndimage.sobel(self.pattern, axis=1)
        gradient_y = ndimage.sobel(self.pattern, axis=0)

        # mag
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

        # plot for debugging
        plt.figure()
        plt.imshow(gradient_magnitude)
        plt.colorbar(label='Magnitude')
        plt.show()

        #get mean of 2d array.
        mean_gradient = np.mean(gradient_magnitude)

        return mean_gradient



    def shannon_entropy(self) -> float:
        """ 
        shannon entropy for speckle patterns. Based on the below: 
        https://www.sciencedirect.com/science/article/abs/pii/S0030402615007950 
        """

        #count occurances of each value. bincount doesn't like 2d arrays. flatten to 1d.
        bins = np.bincount(self.pattern.flatten()) / self.pattern.size

        # reset shannon_entropy
        shannon_entropy = 0.0

        # loop over gray leves
        for i in range(0,2):
            shannon_entropy -= bins[i] * math.log2(bins[i])

        return shannon_entropy


    
    def auto_correlation_peak_sharpness(self) -> float:
        """
        Compute the autocorrelation function (ACF) of the pattern. Based on the below:
        https://link.springer.com/chapter/10.1007/978-1-4614-4235-6_34
        """

        acf_normed = self.auto_correlation(self.pattern)
        r_peak = self.peak_sharpness(acf_normed)

        return r_peak





    def auto_correlation(self, pattern: np.array) -> np.array:
        """
        Calculations Autocorrelatio of a 2D np.array. See here for some good matlab docs:
        https://uk.mathworks.com/help/econ/autocorrelation-and-partial-autocorrelation.html
        """

        fft_forward = np.fft.fft2(pattern)
        power_spec = np.abs(fft_forward)**2
        ftt_back = np.fft.fftshift(np.fft.ifft2(power_spec))
        acf = np.real(ftt_back)
        acf_normed = acf / np.max(acf)

        return acf_normed




    def peak_sharpness(self, acf_normed: np.array) -> float:
        """
        auto correlation peak sharpness. 
        Equation (1): https://doi.org/10.1007/978-1-4614-4235-6_34
        """

        peak_value = acf_normed.max()
        peak_x, peak_y = np.unravel_index(acf_normed.argmax(), acf_normed.shape)

        mean_cardinal = (acf_normed[peak_x , peak_y+1] + \
                         acf_normed[peak_x,  peak_y+1] + \
                         acf_normed[peak_x+1,peak_y  ] + \
                         acf_normed[peak_x-1,peak_y  ]) / 4.0


        r_peak = math.sqrt((peak_value - 0.0) / (peak_value - mean_cardinal))

        return r_peak
