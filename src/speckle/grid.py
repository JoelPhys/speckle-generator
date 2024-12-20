import numpy as np


def create_flattened_grid(nspeckles_x: int, nspeckles_y: int, width: int, height: int) -> tuple:
    """
    Return a flattened grid for speckle locations along x and y.
    Evenly spaced grid based on axis size and the no. speckles along axis.
    Args:
        nspeckles_x (int): Number of speckles along x-axis
        nspeckles_y (int): Number of speckles along y-axis
        width       (int): Width of entire image in Pixels
        height      (int): Height of entire image in Pixels

    Returns:
        tuple (np.array, np.array): speckle indexes for each axis.
    """

    grid_x, grid_y = np.meshgrid(np.linspace(0, width-1,  nspeckles_x),
                                    np.linspace(0, height-1, nspeckles_y))

    grid_flattened_x = grid_x.flatten()
    grid_flattened_y = grid_y.flatten()

    return grid_flattened_x, grid_flattened_y




def random_shift_grid(grid: np.array, low: int, high: int, nsamples: int) -> np.array:
    """
    Takes a uniformly spaced grid as input as applies a random shift to each position. Pulls shift value from a uniform distribution
    Args:
        grid   (np.array): grid to apply shifts
        low         (int): lowest possible value returned from shift
        high        (int): high possible value returned from shift
        nsamples    (int): number of speckles for shift.

    Returns:
        np.array: a numpy array of updated speckle locations after applying a random shift.
    """
    rand_shift_size = np.random.uniform(low, high, nsamples).astype(int)
    updated_grid = grid.astype(int) + rand_shift_size

    return updated_grid



def circle_mask(pos_x: int, pos_y: int, radius: int, img_width: int, img_height: int) -> tuple:
    """
    Generates a circular mask centered at a specified position (pos_x, pos_y) with a given radius. 
    The mask is applied within image bounds (img_width, img_height). 

    Args:
        pos_x       (int): The x-coordinate of the center of the circle.
        pos_y       (int): The y-coordinate of the center of the circle.
        radius      (int): The radius of the circle (in pixels).
        img_width   (int): The width of the image (in pixels), used to limit the mask within bounds.
        img_height  (int): The height of the image (in pixels), used to limit the mask within bounds.

    Returns:
        tuple: A tuple containing:
            - x (np.ndarray): The x-coordinates of mask region.
            - y (np.ndarray): The y-coordinates of mask region.
            - mask (np.ndarray): Bool array indicating whether coord is within the circle or not.
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
