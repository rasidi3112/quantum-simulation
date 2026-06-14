"""
Various barrier potential models (Gaussian, Rectangular, Step, Multiple, RTD).
"""
import numpy as np
from quantumlab.potentials.base import Potential
from quantumlab.core.grid import Grid1D, Grid2D

class GaussianBarrier(Potential):
    """
    Gaussian potential barrier.
    V(x) = V0 * exp(-((x - position)/width)^2)
    """
    def __init__(self, V0: float, width: float, position: float = 0.0):
        self.V0 = V0
        self.width = width
        self.position = position

    def evaluate(self, grid) -> np.ndarray:
        if isinstance(grid, Grid2D):
            return self.V0 * np.exp(-((grid.X - self.position) / self.width) ** 2)
        else:
            return self.V0 * np.exp(-((grid.x - self.position) / self.width) ** 2)


class RectangularBarrier(Potential):
    """
    Rectangular (finite step) potential barrier.
    """
    def __init__(self, V0: float, width: float, position: float = 0.0):
        self.V0 = V0
        self.width = width
        self.position = position

    def evaluate(self, grid) -> np.ndarray:
        if isinstance(grid, Grid2D):
            mask = (grid.X >= self.position - self.width / 2.0) & (grid.X <= self.position + self.width / 2.0)
            return np.where(mask, self.V0, 0.0)
        else:
            mask = (grid.x >= self.position - self.width / 2.0) & (grid.x <= self.position + self.width / 2.0)
            return np.where(mask, self.V0, 0.0)


class PotentialStep(Potential):
    """
    Potential step. V(x) = V0 for x >= position, else 0.
    """
    def __init__(self, V0: float, position: float = 0.0):
        self.V0 = V0
        self.position = position

    def evaluate(self, grid) -> np.ndarray:
        if isinstance(grid, Grid2D):
            return np.where(grid.X >= self.position, self.V0, 0.0)
        else:
            return np.where(grid.x >= self.position, self.V0, 0.0)


class MultipleBarriers(Potential):
    """
    N rectangular barriers at specified positions.
    """
    def __init__(self, V0: float, width: float, positions: list):
        self.V0 = V0
        self.width = width
        self.positions = positions

    def evaluate(self, grid) -> np.ndarray:
        if isinstance(grid, Grid2D):
            V = np.zeros(grid.shape)
            for pos in self.positions:
                mask = (grid.X >= pos - self.width / 2.0) & (grid.X <= pos + self.width / 2.0)
                V[mask] = self.V0
            return V
        else:
            V = np.zeros(grid.shape)
            for pos in self.positions:
                mask = (grid.x >= pos - self.width / 2.0) & (grid.x <= pos + self.width / 2.0)
                V[mask] = self.V0
            return V


class ResonantTunnelingDiode(Potential):
    """
    Double barrier structure forming a quantum well in between.
    """
    def __init__(self, V0: float, barrier_width: float, well_width: float, position: float = 0.0):
        self.V0 = V0
        self.barrier_width = barrier_width
        self.well_width = well_width
        self.position = position

    def evaluate(self, grid) -> np.ndarray:
        half_w = self.well_width / 2.0
        w_b = self.barrier_width
        pos_left = self.position - half_w - w_b / 2.0
        pos_right = self.position + half_w + w_b / 2.0

        if isinstance(grid, Grid2D):
            mask_left = (grid.X >= pos_left - w_b / 2.0) & (grid.X <= pos_left + w_b / 2.0)
            mask_right = (grid.X >= pos_right - w_b / 2.0) & (grid.X <= pos_right + w_b / 2.0)
            V = np.zeros(grid.shape)
            V[mask_left] = self.V0
            V[mask_right] = self.V0
            return V
        else:
            mask_left = (grid.x >= pos_left - w_b / 2.0) & (grid.x <= pos_left + w_b / 2.0)
            mask_right = (grid.x >= pos_right - w_b / 2.0) & (grid.x <= pos_right + w_b / 2.0)
            V = np.zeros(grid.shape)
            V[mask_left] = self.V0
            V[mask_right] = self.V0
            return V
