import matplotlib.pyplot as plt
import numpy as np
import PIL

class Image:

    def __init__(self,pattern: np.array):
        self.pattern = pattern



    def balance(self, balance: float) -> None:
        """
        Args:
            balance (float): value between 0.0 and 1.0 that gets multiplied with the speckle pattern
        Returns:
            None
        """

        self.pattern = self.pattern * balance

        return None




    def invert(self, invert: bool = False) -> None:
        """
        Inverts the pattern attribute if the invert parameter is set to True.

        Parameters:
        invert (bool): A flag indicating whether to invert the pattern. If True, the pattern 
                    attribute is updated by subtracting its current value from 1. 
                    If False, the pattern remains unchanged.

        Returns:
        None
        """

        if invert:
            self.pattern = 1 - self.pattern

        return None



    def show(self, vmin: float = 0.0, vmax: float = 1.0) -> None:
        """
        Displays the current pattern as an image using Matplotlib.

        Returns:
        None
        """
        plt.figure()
        plt.xlabel('Pixels')
        plt.ylabel('Pixels')
        plt.imshow(self.pattern,cmap='gray', vmin=vmin, vmax=vmax)
        plt.show()

        return None


    def save(self,filename: str, resolution: int, format: str = "TIFF") -> None:
        """
        Save 2D array as an image with PIL package.

        Args:
        filename   (str): name of output image
        format     (str): filetype, currently accepts PNG, BMP, TIFF
        resolution (int):

        Returns:
        None: Saves image to directory withuser specified details.
        """

        allowed_formats = {"PNG", "BMP", "TIFF"}
        format_upper = format.upper()

        if format_upper not in allowed_formats:
            raise ValueError(f"Invalid format '{format}'. Allowed formats are: {', '.join(allowed_formats)}")

        image = PIL.Image.fromarray((self.pattern * 255).astype(np.uint8))
        dpi = (resolution, resolution)
        image.save(filename, format=format, dpi=dpi)

        return None
