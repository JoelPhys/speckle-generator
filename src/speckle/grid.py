import numpy as np


def create_flattened(nspeckles_x: int, nspeckles_y: int, width: int, height: int) -> tuple:
    """
    Return a flattened grid for speckle locations along x and y.
    Evenly spaced grid based on axis size and the no. speckles along axis.
    """

    grid_x, grid_y = np.meshgrid(np.linspace(0, width-1,  nspeckles_x),
                                    np.linspace(0, height-1, nspeckles_y))

    grid_flattened_x = grid_x.flatten()
    grid_flattened_y = grid_y.flatten()

    return grid_flattened_x, grid_flattened_y




def random_shift(grid: np.array, low: int, high: int, nsamples: int) -> np.array:
    """
    Takes a uniformly spaced grid as input as applies a random shift to each position.
    grid   (np.array): grid to apply shifts
    low         (int): lowest possible value returned from shift
    high        (int): high possible value returned from shift
    nsamples    (int): number of speckles for shift.
    """
    rand_shift_size = np.random.uniform(low, high, nsamples).astype(int)
    updated_grid = grid.astype(int) + rand_shift_size

    return updated_grid



def circle_mask(pos_x: int, pos_y: int, radius: int, img_width: int, img_height: int) -> tuple:
    """
    create a circular 'speckle' mask.
    """

    min_x = max(pos_x - radius, 0)
    min_y = max(pos_y - radius, 0)
    max_x = min(pos_x + radius + 1, img_width)
    max_y = min(pos_y + radius + 1, img_height)

    # Generate mesh grid of possible (xx, yy) points
    x, y = np.meshgrid(np.arange(min_x, max_x), np.arange(min_y, max_y))

    # Update the pattern for points inside the circle's radius
    mask = (x - pos_x)**2 + (y - pos_y)**2 <= radius**2

    return x, y, mask
