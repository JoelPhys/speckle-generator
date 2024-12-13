import matplotlib.pyplot as plt
import numpy as np
import PIL

class Image:

    def __init__(self,pattern: np.array):
        self.pattern = pattern
    
    def balance(self, balance: float) -> None:
        self.pattern = self.pattern * balance

    def invert(self, invert: bool) -> None:
        if invert:
            self.pattern = 1 - self.pattern

    def show(self) -> None:
        plt.figure()
        plt.title('Test Speckle Pattern')
        plt.imshow(self.pattern,cmap='gray', vmin=0, vmax=1) 
        plt.show()

    def save(self,filename: str,format: str, resolution: int) -> None:
        image = PIL.Image.fromarray((self.pattern * 255).astype(np.uint8))
        dpi = (resolution, resolution)
        print(dpi)
        image.save(filename, format=format, dpi=dpi)