import numpy as np


class Generate:

    def __init__(self,image_height: int ,image_width: int,radius: int, variability: float):
        self.image_height = image_height
        self.image_width = image_width
        self.radius = radius
        self.variability = variability

    def pattern(self) -> np.array:

        num_speckles = 2000
        pattern = np.zeros((self.image_height,self.image_width), dtype=int)

        # completely random locations
        # loc = np.random.choice(pattern.size,  num_speckles, replace=False)
        # row, col = np.unravel_index(loc, (self.image_height, self.image_width))

        # grid with some variability
        grid = 50
        num_speckles = grid * grid
        grid_size = int(np.sqrt(num_speckles))  # sqrt(num_speckles) by sqrt(num_speckles)
        grid_x, grid_y = np.meshgrid(np.linspace(0, self.image_width - 1, grid_size),
                                     np.linspace(0, self.image_height - 1, grid_size))
        
        print(grid_size)
        
        grid_x = grid_x.flatten()
        grid_y = grid_y.flatten()

        # Apply random shifts based on variability
        shift_x = np.random.uniform(-self.variability * (self.image_width / grid_size), 
                                    self.variability * (self.image_width / grid_size), num_speckles)
        shift_y = np.random.uniform(-self.variability * (self.image_height / grid_size), 
                                    self.variability * (self.image_height / grid_size), num_speckles)

        # Apply the shifts to the grid positions
        row = (grid_y + shift_y).astype(int)
        col = (grid_x + shift_x).astype(int)

        # Make sure the positions are within bounds
        row = np.clip(row, 0, self.image_height - 1)
        col = np.clip(col, 0, self.image_width - 1)

        # create circles
        for ii in range(0,row.size):

            l_bound = row[ii]-self.radius
            r_bound = row[ii]+self.radius+1
            b_bound = col[ii]-self.radius
            t_bound = col[ii]+self.radius+1

            for xx in range(l_bound,r_bound):

                for yy in range(b_bound,t_bound):

                    if (xx - row[ii])**2 + (yy - col[ii])**2 <= self.radius**2:

                        if 0 <= xx < self.image_height and 0 <= yy < self.image_width:

                            pattern[xx, yy] = 1


        return pattern