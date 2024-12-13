import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

class Generate:

    def __init__(self,image_height: int ,image_width: int, spacing: int, mean_radius: int = 4, stddev: int = 1, variability: float = 0.4):
        self.image_height = image_height
        self.image_width = image_width
        self.mean_radius = mean_radius
        self.var = variability
        self.stddev = stddev
        self.spacing = spacing
        self.radius = None
        self.shift_x = None
        self.shift_y = None


    def pattern(self) -> np.array:

        pattern = np.zeros((self.image_height,self.image_width), dtype=int)

        # completely random locations
        # num_speckles = 2000
        # loc = np.random.choice(pattern.size,  num_speckles, replace=False)
        # row, col = np.unravel_index(loc, (self.image_height, self.image_width))

        # speckles per row/col
        speckles_per_row = self.image_width // self.spacing
        speckles_per_col = self.image_height // self.spacing 

        # total number of speckles
        num_speckles = speckles_per_row * speckles_per_col

        print(speckles_per_row)
        print(speckles_per_col)
        print(num_speckles)


        grid_x, grid_y = np.meshgrid(np.linspace(0, self.image_width - 1, speckles_per_row),
                                     np.linspace(0, self.image_height - 1, speckles_per_col))

        grid_x = grid_x.flatten()
        grid_y = grid_y.flatten()

        # Apply random shifts based on variability
        self.shift_x = np.random.uniform(-self.var * self.spacing, self.var * self.spacing, num_speckles)
        self.shift_y = np.random.uniform(-self.var * self.spacing, self.var * self.spacing, num_speckles)

        # Apply the shifts to the grid positions
        row = (grid_y + self.shift_y).astype(int)
        col = (grid_x + self.shift_x).astype(int)

        # Make sure the positions are within bounds
        row = np.clip(row, 0, self.image_height - 1)
        col = np.clip(col, 0, self.image_width - 1)

        # pull speckle size from a normal distribution with user defined mean size and standard deviation.
        self.radius = np.random.normal(self.mean_radius,self.stddev,num_speckles).astype(int)
        

        # create circles
        for ii in range(0,row.size):

            l_bound = row[ii]-self.radius[ii]
            r_bound = row[ii]+self.radius[ii]+1
            b_bound = col[ii]-self.radius[ii]
            t_bound = col[ii]+self.radius[ii]+1

            for xx in range(l_bound,r_bound):

                for yy in range(b_bound,t_bound):

                    if (xx - row[ii])**2 + (yy - col[ii])**2 <= self.radius[ii]**2:

                        if 0 <= xx < self.image_height and 0 <= yy < self.image_width:

                            pattern[xx, yy] = 1


        return pattern
    

    def statistics(self, speckle_pattern: np.array) -> str:

        # Calculate pattern density (percent of black pixels)
        pattern_density = np.average(speckle_pattern) * 100  # Convert to percentage
        
        # Calculate average, min, and max radius
        radius_avg = np.average(self.radius)
        radius_min = np.min(self.radius)
        radius_max = np.max(self.radius)
        shift_avgx = np.average(abs(self.shift_x))
        shift_avgy = np.average(abs(self.shift_y))

        stats = "Pattern Density: \t" + f"{pattern_density:.2f}" + "% \n" + \
            "Average Radius: \t" + f"{radius_avg:.2f}" + " pixels \n" + \
            "Min Radius: \t \t" + f"{radius_min}" + " pixels \n" + \
            "Max Radius: \t \t" + f"{radius_max}" + " pixels \n" + \
            "Average shift in x: \t \t" + f"{shift_avgx:.2f}" + " pixels \n" + \
            "Average shift in y: \t \t" + f"{shift_avgy:.2f}" + " pixels \n"

        return stats




