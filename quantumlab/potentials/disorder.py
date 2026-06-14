import numpy as np
from quantumlab.potentials.base import Potential
from quantumlab.core.grid import Grid1D, Grid2D

class RandomDisorder(Potential):

    def __init__(self, W: float, cell_width: float, seed: int=None):
        self.W = W
        self.cell_width = cell_width
        self.seed = seed

    def evaluate(self, grid) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        if isinstance(grid, Grid2D):
            x_min, x_max = (grid.x_min, grid.x_max)
            num_cells = int(np.ceil((x_max - x_min) / self.cell_width))
            cell_values = rng.uniform(-self.W / 2.0, self.W / 2.0, num_cells)
            indices = np.clip(((grid.X - x_min) // self.cell_width).astype(int), 0, num_cells - 1)
            return cell_values[indices]
        else:
            x_min, x_max = (grid.x_min, grid.x_max)
            num_cells = int(np.ceil((x_max - x_min) / self.cell_width))
            cell_values = rng.uniform(-self.W / 2.0, self.W / 2.0, num_cells)
            indices = np.clip(((grid.x - x_min) // self.cell_width).astype(int), 0, num_cells - 1)
            return cell_values[indices]
