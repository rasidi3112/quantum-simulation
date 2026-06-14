import numpy as np
from quantumlab.potentials.base import Potential
from quantumlab.core.grid import Grid1D, Grid2D

class CrystalPotential(Potential):

    def __init__(self, V0: float, lattice_spacing: float, position: float=0.0):
        self.V0 = V0
        self.lattice_spacing = lattice_spacing
        self.position = position

    def evaluate(self, grid) -> np.ndarray:
        if isinstance(grid, Grid2D):
            return self.V0 * np.sin(np.pi * (grid.X - self.position) / self.lattice_spacing) ** 2
        else:
            return self.V0 * np.sin(np.pi * (grid.x - self.position) / self.lattice_spacing) ** 2
