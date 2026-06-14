"""
Harmonic Oscillator potential model.
"""
import numpy as np
from quantumlab.potentials.base import Potential
from quantumlab.core.grid import Grid1D, Grid2D

class HarmonicOscillator(Potential):
    """
    Quantum Harmonic Oscillator potential:
    V(x) = 0.5 * m * (omega^2) * (x - position)^2
    """
    def __init__(self, omega: float, m: float = 1.0, position: float = 0.0):
        self.omega = omega
        self.m = m
        self.position = position

    def evaluate(self, grid) -> np.ndarray:
        if isinstance(grid, Grid2D):
            # Centered harmonic well in 2D (isotropic)
            return 0.5 * self.m * (self.omega ** 2) * ((grid.X - self.position) ** 2 + grid.Y ** 2)
        else:
            return 0.5 * self.m * (self.omega ** 2) * (grid.x - self.position) ** 2
