"""
User-defined custom potentials.
"""
import numpy as np
from quantumlab.potentials.base import Potential
from quantumlab.core.grid import Grid1D, Grid2D

class CustomPotential(Potential):
    """
    Potential wrapped around an arbitrary python callable function.
    """
    def __init__(self, func):
        """
        func: A callable function.
              For 1D: func(x)
              For 2D: func(x, y)
        """
        if not callable(func):
            raise TypeError("func must be callable")
        self.func = func

    def evaluate(self, grid) -> np.ndarray:
        if isinstance(grid, Grid2D):
            return self.func(grid.X, grid.Y)
        else:
            return self.func(grid.x)
