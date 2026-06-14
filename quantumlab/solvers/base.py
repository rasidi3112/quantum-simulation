"""
Abstract base class for Time-Dependent Schrödinger Equation solvers.
"""
from abc import ABC, abstractmethod

class Solver(ABC):
    """
    Abstract base class for quantum solvers.
    """
    def __init__(self, grid, potential, dt: float, hbar: float = 1.0, m: float = 1.0):
        self.grid = grid
        self.potential = potential
        self.dt = dt
        self.hbar = hbar
        self.m = m

    @abstractmethod
    def step(self, wavefunction):
        """
        Evolve the wave function by a single time step dt.
        Returns the evolved WaveFunction object.
        """
        pass
