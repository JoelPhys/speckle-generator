import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import interpolate
from icecream import ic
from scipy.optimize import least_squares
from scipy.signal import correlate2d
from numba import jit


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


    def subset(self, image: np.ndarray, x: int, y: int) -> np.ndarray:
        """
        Parameters:
        x (int): x-coord of subset center in image
        y (int): y-coord of subset center in image

        """

        half_size = self.subset_size // 2

        # reference image subset
        x1, x2 = x - half_size, x + half_size + 1
        y1, y2 = y - half_size, y + half_size + 1

        # Ensure indices are within bounds
        if (x1 < 0 or y1 < 0 or x2 > self.image_ref.shape[1] or y2 > self.image_ref.shape[0]):
            raise ValueError(f"Subset exceeds image boundaries.\nSubset Pixel Range:\n"
                            f"x1: {x1}\n"
                            f"x2: {x2}\n"
                            f"y1: {y1}\n"
                            f"y2: {y2}")
    
        # Extract subsets
        subset = image[y1:y2, x1:x2]

        return subset


    def perform_interpolation(self, interp_x: int, interp_y: int, kind: str) -> tuple[np.ndarray, np.ndarray]:
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


        interped_image_ref = interped_vals_ref.reshape(len(y_mesh), len(x_mesh))
        interped_image_def = interped_vals_def.reshape(len(y_mesh), len(x_mesh))

        # plt.figure()
        # plt.scatter(x_vals,vals[0,:])
        # plt.scatter(x,self.image_ref[0,:])
        # plt.show()

        return interped_image_ref, interped_image_def
    

    @jit
    def global_search_loops(ref_subset, image_def) -> tuple[int,int,float]:

        subset_size = ref_subset.shape[0]
        min_x = subset_size // 2
        min_y = subset_size // 2
        max_x = image_def.shape[0] - subset_size // 2
        max_y = image_def.shape[1] - subset_size // 2


        # Loop over a grid of positions to create multiple residuals
        ssd_final = float('inf')
        for u in range(min_x, max_x):
            for v in range(min_y, max_y):

                def_subset = self.subset(self.image_def, u, v)
                ssd_temp = self.ssd(ref_subset,def_subset)
                if ssd_temp < ssd_final:
                    ssd_final = ssd_temp
                    u_final = u
                    v_final = v

                if ssd_final == 0.0:
                    break


        return u_final,v_final,ssd_final
                    


    def global_search_scipy(self, ref_subset, image_def) -> tuple[int, int, float]:

        subset_size = ref_subset.shape[0]
        
        # use scipy.signal
        ref_squared = np.sum(ref_subset ** 2)
        corr = correlate2d(image_def, ref_subset, mode="valid", boundary="fill", fillvalue=0)

        ssd_map = ref_squared - 2 * corr

        # debugging
        # plt.figure(figsize=(10, 5))
        # plt.subplot(1, 2, 1)
        # plt.title("Original Image")
        # plt.imshow(ssd_map, cmap="gray")
        # plt.show()

        v_final, u_final = np.unravel_index(np.argmin(ssd_map), ssd_map.shape)
        
        return u_final, v_final, np.min(ssd_map)

    # def cost_function(self, displacement, subset_ref, image_def):


    #     x_disp = int(round(displacement[0]))  # Round to nearest integer
    #     y_disp = int(round(displacement[1]))  # Round to nearest integer
        
    #     subset_size = subset_ref.shape[0]
    #     min_x = subset_size // 2
    #     min_y = subset_size // 2
    #     max_x = image_def.shape[0] - subset_size // 2 - 1
    #     max_y = image_def.shape[1] - subset_size // 2 - 1
    #     print(max_x,ma

    #     residuals = []
    #     print(displacement)

    #     # Loop over a grid of positions to create multiple residuals
    #     for x in range(min_x, max_x):
    #         for y in range(min_y, max_y):
    #             # Apply displacement to the subset (rounding to integers)
    #             x_disp = int(round(x + displacement[0]))
    #             y_disp = int(round(y + displacement[1]))

    #             # Ensure displacement is within bounds
    #             if x_disp < 0 or x_disp >= image_def.shape[1] or y_disp < 0 or y_disp >= image_def.shape[0]:
    #                 continue  # Skip out-of-bounds subsets

    #             # Extract deformed subset
    #             subset_def = self.subset(image_def, x_disp, y_disp)
                
    #             # Compute SSD residual for this subset
    #             residual = self.ssd(subset_ref, subset_def)
    #             residuals.append(residual)

        
    #     residuals.append(residual)

    #     return np.array([residual])     


    # def levenberg_marquardt(self, subset_ref: np.ndarray, image_def: np.ndarray):

    #     initial_displacement = (100,100)
    #     result = least_squares(
    #         fun=self.cost_function,
    #         x0=initial_displacement,  # initial guess of displacement
    #         diff_step=1,
    #         args=(subset_ref, image_def),  # args: (reference, deformed, window_size, method)
    #         method='lm'  # Use Levenberg-Marquardt
    #     )

    #     # Extract the final displacement
    #     optimized_displacement = result.x
    #     print(optimized_displacement)
    #     return optimized_displacement





