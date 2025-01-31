import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import interpolate
from icecream import ic


class Correlation:
    def __init__(self, image_ref: np.ndarray, image_def: np.ndarray, subset_size: int = 21):
        self.image_ref = image_ref
        self.image_def = image_def
        self.subset_size = subset_size

    def pattern_gradient(self, subset):
        """Image gradients using sobel."""
        grad_x = cv2.Sobel(subset, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(subset, cv2.CV_64F, 0, 1, ksize=3)
        return grad_x, grad_y

    def ssd(self, reference_subset, deformed_subset) -> float:
        ssd_value = np.sum((reference_subset - deformed_subset) ** 2)
        return ssd_value
    


    def nssd(reference_subset, deformed_subset) -> float:

        mean_ref = np.mean(reference_subset)
        mean_def = np.mean(deformed_subset)
        std_ref = np.std(reference_subset)
        std_def = np.std(deformed_subset)

        normed_ref = (reference_subset - mean_ref) / std_ref
        normed_def = (deformed_subset - mean_def) / std_def

        nssd_value = np.sum((normed_ref - normed_def) ** 2)

        return nssd_value
    
    def znssd(reference_subset, deformed_subset) -> float:

        mean_ref = np.mean(reference_subset)
        mean_def = np.mean(deformed_subset)

        zero_mean_ref = reference_subset - mean_ref
        zero_mean_def = deformed_subset - mean_def

        znssd_value = np.sum((zero_mean_ref - zero_mean_def) ** 2)

        return znssd_value


    def subsets(self, x: int, y: int, u: int, v: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Parameters:
        x (int): x-coord of subset center in reference image
        y (int): y-coord of subset center in reference image
        u (int): x-coord of subset centre in deformed image
        v (int): y-coord of subset centre in deformed image
        """

        half_size = self.subset_size // 2

        # reference image subset
        x1, x2 = x - half_size, x + half_size + 1
        y1, y2 = y - half_size, y + half_size + 1

        # deformed subset
        x1_def, x2_def = u - half_size, u + half_size + 1
        y1_def, y2_def = v - half_size, v + half_size + 1

        # print(x1,x2,y1,y2,x1_def,x2_def,y1_def,y2_def)

        # Ensure indices are within bounds
        if (x1 < 0 or y1 < 0 or x2 > self.image_ref.shape[1] or y2 > self.image_ref.shape[0] or
            x1_def < 0 or y1_def < 0 or x2_def > self.image_def.shape[1] or y2_def > self.image_def.shape[0]):
            raise ValueError("Subset exceeds image boundaries.")

        # Extract subsets
        subset_ref = self.image_ref[y1:y2, x1:x2]
        subset_def = self.image_def[y1_def:y2_def, x1_def:x2_def]

        return subset_ref, subset_def
    

    def perform_interpolation(self, interp_x: int, interp_y: int, kind: str) -> None:
        """
        Parameters:
        interp_x (int): number of interpolations between each pixel along x axis
        interp_y (int): number of interpolations between each pixel along y axis
        kind     (str): Type of interpolation. Supported values are linear", "nearest", "slinear", "cubic", "quintic" and "pchip".
        """

        dims_ref = self.image_ref.shape

        x  = np.linspace(0, dims_ref[0] - 1, dims_ref[0])
        y  = np.linspace(0, dims_ref[0] - 1, dims_ref[1])
        interpolator_ref = interpolate.RegularGridInterpolator((y, x), self.image_ref, method=kind)
        interpolator_def = interpolate.RegularGridInterpolator((y, x), self.image_def, method=kind)

        # new grid
        x_vals = np.linspace(0, dims_ref[1] - 1, (dims_ref[1] - 1) * interp_x + 1)
        y_vals = np.linspace(0, dims_ref[0] - 1, (dims_ref[0] - 1) * interp_y + 1)
        x_mesh, y_mesh = np.meshgrid(x_vals, y_vals)

        new_points = np.column_stack([y_mesh.ravel(), x_mesh.ravel()])
        interped_vals_ref = interpolator_ref(new_points)
        interped_vals_def = interpolator_def(new_points)


        vals = interped_vals_ref.reshape(len(y_mesh), len(x_mesh))
        self.image_def = interped_vals_def.reshape(len(y_mesh), len(x_mesh))

        plt.figure()
        plt.scatter(x_vals,vals[0,:])
        plt.scatter(x,self.image_ref[0,:])
        plt.show()

        ic(self.image_ref.shape)

