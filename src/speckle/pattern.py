import numpy as np
from scipy import ndimage
from speckle.grid import create_flattened_grid, random_shift_grid, circle_mask

class Pattern:
    """
    A class for generating and manipulating speckle patterns.

    Attributes:
        seed            (int): Random seed to be used numpy.
        image_height    (int): Height of the image (in pixels).
        image_width     (int): Width of the image (in pixels).
        spacing         (int): Spacing between speckles (in pixels).
        mean_radius     (int): The mean radius of each speckle (in pixels).
        stddev          (int): The standard deviation for the speckle size (in pixels).
        var           (float): % Variability factor for random shifts applied to speckles.
        radius     (np.array): Array storing the radius of each speckle.
        grid_x     (np.array): Array storing the random shift in the x-direction for each speckle.
        grid_y     (np.array): Array storing the random shift in the y-direction for each speckle.
        pattern    (np.array): The 2D array representing the generated speckle pattern.


    Methods:
        generate() -> np.array:
            Generates a speckle pattern based on the specified parameters.
        
        rotate(angle: float) -> np.array:
            Rotates the generated speckle pattern by a specified angle (in degrees).
        
        statistics() -> None:
            Prints basic statistics about the generated speckle pattern, including its density,
            average radius, minimum and maximum radius, and average shifts in both x and y directions.
    """

    def __init__(self, image_width: int=512, image_height: int=512, mean_radius: int=4, spacing: int=10, variability: float=0.6, stddev: float=1.0, seed: int=None):
        self.seed = seed  
        self.image_height = image_height
        self.image_width = image_width
        self.mean_radius = mean_radius
        self.var = variability
        self.stddev = stddev
        self.spacing = spacing
        self.radii = np.zeros((image_width,image_height), dtype=int)
        self.grid_x = np.zeros((image_width,image_height), dtype=int)
        self.grid_y = np.zeros((image_width,image_height), dtype=int)
        self.pattern = np.zeros((image_width,image_height), dtype=int)


    def generate(self) -> np.array:
        """
        Generate a speckle pattern based on default or user provided paramters.

        Args:
            None

        Returns:
            np.array: 2D speckle pattern.
        """

        # random seed
        np.random.seed(self.seed)

        # speckles per row/col
        nspeckles_x = self.image_width // self.spacing
        nspeckles_y = self.image_height // self.spacing

        # total number of speckles
        nspeckles = nspeckles_x * nspeckles_y

        # uniformly spaced grid of speckles.
        grid_x_uniform, grid_y_uniform = create_flattened_grid(nspeckles_x, nspeckles_y, self.image_width, self.image_height)

        # apply random shift
        low  = -self.var * self.spacing
        high =  self.var * self.spacing
        self.grid_x = random_shift_grid(grid_x_uniform, low, high, nspeckles)
        self.grid_y = random_shift_grid(grid_y_uniform, low, high, nspeckles)


        # pull speckle size from a normal distribution with user defined mean size and standard deviation.
        self.radii = np.random.normal(self.mean_radius, self.stddev, nspeckles).astype(int)


        # loop over all grid points and create a circle mask. Mask then applied to pattern array.
        for ii in range(0, nspeckles):
            x,y,mask = circle_mask(self.grid_x[ii], self.grid_y[ii], self.radii[ii], self.image_width, self.image_height)
            self.pattern[x[mask], y[mask]] = 1

        return self.pattern




    def rotate(self, angle: float) -> np.array:
        """
        takes the pattern generated by Pattern.generate() and rotates it by a user defined angle
        Args:
            angle (float): Angle of clockwise rotaion (degrees)

        Returns:
            np.array: Rotated speckle pattern. Note: It does not reshape the pattern.
        """

        rotated_pattern = ndimage.rotate(self.pattern, angle, reshape=False)

        return rotated_pattern





    def statistics(self) -> None:
        """
        Print some very basic statistics about the speckle pattern.

        Args:
            None

        Returns:
            None: Prints mean pixel value and speckle size information to stdout
        """
        
        pattern_density = np.average(self.pattern) * 100  # Convert to percentage
        
        # Calculate average, min, and max radius
        radius_avg = np.average(self.radii)
        radius_min = np.min(self.radii)
        radius_max = np.max(self.radii)

        print(f"{'Mean Pixel Value of Entire Pattern:':45}" +  f"{pattern_density:.2f}" + " [%]")
        print(f"{'Average Speckle Radius:':45}" + f"{radius_avg:.2f}" + " [pixels]" )
        print(f"{'Min Speckle Radius:':45}" + f"{radius_min}" + " [pixels]" )
        print(f"{'Max Speckle Radius:':45}" + f"{radius_max}" + " [pixels]" )

        return None
