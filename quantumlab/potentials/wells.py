import numpy as np
from quantumlab.potentials.base import Potential
from quantumlab.core.grid import Grid1D, Grid2D

class InfiniteSquareWell(Potential):

    def __init__(self, width: float, position: float=0.0, V_inf: float=1000000000.0):
        self.width = width
        self.position = position
        self.V_inf = V_inf

    def evaluate(self, grid) -> np.ndarray:
        if isinstance(grid, Grid2D):
            mask = (grid.X >= self.position - self.width / 2.0) & (grid.X <= self.position + self.width / 2.0)
            return np.where(mask, 0.0, self.V_inf)
        else:
            mask = (grid.x >= self.position - self.width / 2.0) & (grid.x <= self.position + self.width / 2.0)
            return np.where(mask, 0.0, self.V_inf)

class FiniteSquareWell(Potential):

    def __init__(self, V0: float, width: float, position: float=0.0):
        self.V0 = V0
        self.width = width
        self.position = position

    def evaluate(self, grid) -> np.ndarray:
        if isinstance(grid, Grid2D):
            mask = (grid.X >= self.position - self.width / 2.0) & (grid.X <= self.position + self.width / 2.0)
            return np.where(mask, 0.0, self.V0)
        else:
            mask = (grid.x >= self.position - self.width / 2.0) & (grid.x <= self.position + self.width / 2.0)
            return np.where(mask, 0.0, self.V0)

class DoubleWell(Potential):

    def __init__(self, V0: float, barrier_distance: float, position: float=0.0):
        self.V0 = V0
        self.barrier_distance = barrier_distance
        self.position = position

    def evaluate(self, grid) -> np.ndarray:
        if isinstance(grid, Grid2D):
            term = (grid.X - self.position) / self.barrier_distance
            return self.V0 * (term ** 2 - 1.0) ** 2
        else:
            term = (grid.x - self.position) / self.barrier_distance
            return self.V0 * (term ** 2 - 1.0) ** 2
