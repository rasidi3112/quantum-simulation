from abc import ABC, abstractmethod

class Solver(ABC):

    def __init__(self, grid, potential, dt: float, hbar: float=1.0, m: float=1.0):
        self.grid = grid
        self.potential = potential
        self.dt = dt
        self.hbar = hbar
        self.m = m

    @abstractmethod
    def step(self, wavefunction):
        pass
