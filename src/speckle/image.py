import matplotlib.pyplot as plt
import numpy as np
import PIL

class Image:

    def __init__(self,pattern: np.array, gray_level: int=256):
        self.pattern = pattern
        self.gray_level = gray_level



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
            self.pattern = self.gray_level - 1 - self.pattern

        return None



    def show(self) -> None:
        """
        Displays the current pattern as an image using Matplotlib.

        Returns:
        None
        """
        plt.figure()
        plt.xlabel('Pixels')
        plt.ylabel('Pixels')
        plt.imshow(self.pattern,cmap='gray', vmin=0.0, vmax=self.gray_level-1)
        plt.colorbar()
        plt.show()

        return None


    def save(self,filename: str, format: str = "TIFF", resolution: int = 512) -> None:
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


        # convert to 256 when saving.
        convertion_factor = 255 / (self.gray_level - 1) 

        image = PIL.Image.fromarray((self.pattern * convertion_factor).astype(np.uint8))
        dpi = (resolution, resolution)
        image.save(filename, format=format, dpi=dpi)

        return None
