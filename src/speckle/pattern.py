import numpy as np
from scipy import ndimage
from speckle.grid import create_flattened, random_shift, circle_mask

class Pattern:

    def __init__(self,image_height: int ,image_width: int, spacing: int, mean_radius: int = 4, stddev: int = 1, variability: float = 0.4):
        self.image_height = image_height
        self.image_width = image_width
        self.mean_radius = mean_radius
        self.var = variability
        self.stddev = stddev
        self.spacing = spacing
        self.radius = np.zeros((image_width,image_height), dtype=int)
        self.random_shift_x = np.zeros((image_width,image_height), dtype=int)
        self.random_shift_y = np.zeros((image_width,image_height), dtype=int)
        self.pattern = np.zeros((image_width,image_height), dtype=int)


    def generate(self) -> np.array:
        """
        Generate a speckle pattern based on grid positions and random shifts.
        """

        # speckles per row/col
        nspeckles_x = self.image_width // self.spacing
        nspeckles_y = self.image_height // self.spacing

        # total number of speckles
        nspeckles = nspeckles_x * nspeckles_y

        # uniformly spaced grid of speckles.
        speck_grid_x, speck_grid_y = create_flattened(nspeckles_x, nspeckles_y, self.image_width, self.image_height)

        # apply random shift
        low  = -self.var * self.spacing
        high =  self.var * self.spacing
        speck_grid_x_shft = random_shift(speck_grid_x, low, high, nspeckles)
        speck_grid_y_shft = random_shift(speck_grid_y, low, high, nspeckles)


        # pull speckle size from a normal distribution with user defined mean size and standard deviation.
        self.speck_size = np.random.normal(self.mean_radius, self.stddev, nspeckles).astype(int)


        for ii in range(0, nspeckles):
            x,y,mask = circle_mask(speck_grid_x_shft[ii], speck_grid_y_shft[ii], self.speck_size[ii], self.image_width, self.image_height)
            self.pattern[x[mask], y[mask]] = 1

        return self.pattern




    def rotate(self, angle: float) -> np.array:
        """
        rotate the speckle pattern by user specified angle (degrees)
        """

        rotated_pattern = ndimage.rotate(self.pattern, angle, reshape=False)

        return rotated_pattern





    def statistics(self) -> None:

        # Calculate pattern density (percent of black pixels)
        pattern_density = np.average(self.pattern) * 100  # Convert to percentage
        
        # Calculate average, min, and max radius
        radius_avg = np.average(self.radius)
        radius_min = np.min(self.radius)
        radius_max = np.max(self.radius)
        shift_avgx = np.average(self.random_shift_x)
        shift_avgy = np.average(self.random_shift_y)

        print(f"{'Pattern Density:':35}" +  f"{pattern_density:.2f}" + " [%]")
        print(f"{'Average Radius:':35}" + f"{radius_avg:.2f}" + " [pixels]" )
        print(f"{'Min Radius:':35}" + f"{radius_min}" + " [pixels]" )
        print(f"{'Max Radius:':35}" + f"{radius_max}" + " [pixels]" )
        print(f"{'Average of random shift in x:':35}" + f"{shift_avgx:.2f}" + " [pixels]" )
        print(f"{'Average of random shift in y:':35}" + f"{shift_avgy:.2f}" + " [pixels]" )

        return None
