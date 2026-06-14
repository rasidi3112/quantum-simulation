import numpy as np
from quantumlab.core.grid import Grid1D, Grid2D

class WaveFunction1D:

    def __init__(self, grid: Grid1D, psi: np.ndarray=None):
        self.grid = grid
        if psi is None:
            self.psi = np.zeros(grid.N, dtype=complex)
        else:
            if psi.shape != grid.shape:
                raise ValueError(f'psi shape {psi.shape} does not match grid shape {grid.shape}')
            self.psi = np.array(psi, dtype=complex)

    @property
    def probability_density(self) -> np.ndarray:
        return np.abs(self.psi) ** 2

    def norm(self) -> float:
        return float(np.sum(self.probability_density) * self.grid.dx)

    def normalize(self) -> 'WaveFunction1D':
        n = self.norm()
        if n > 0:
            self.psi /= np.sqrt(n)
        return self

    def fourier_transform(self) -> np.ndarray:
        return np.fft.fft(self.psi) * (self.grid.dx / np.sqrt(2 * np.pi))

    def copy(self) -> 'WaveFunction1D':
        return WaveFunction1D(self.grid, self.psi.copy())

    @classmethod
    def gaussian(cls, grid: Grid1D, x0: float, k0: float, sigma: float) -> 'WaveFunction1D':
        psi = np.exp(-(grid.x - x0) ** 2 / (4.0 * sigma ** 2)) * np.exp(1j * k0 * grid.x)
        wf = cls(grid, psi)
        wf.normalize()
        return wf

class WaveFunction2D:

    def __init__(self, grid: Grid2D, psi: np.ndarray=None):
        self.grid = grid
        if psi is None:
            self.psi = np.zeros(grid.shape, dtype=complex)
        else:
            if psi.shape != grid.shape:
                raise ValueError(f'psi shape {psi.shape} does not match grid shape {grid.shape}')
            self.psi = np.array(psi, dtype=complex)

    @property
    def probability_density(self) -> np.ndarray:
        return np.abs(self.psi) ** 2

    def norm(self) -> float:
        return float(np.sum(self.probability_density) * self.grid.dx * self.grid.dy)

    def normalize(self) -> 'WaveFunction2D':
        n = self.norm()
        if n > 0:
            self.psi /= np.sqrt(n)
        return self

    def fourier_transform(self) -> np.ndarray:
        return np.fft.fft2(self.psi) * (self.grid.dx * self.grid.dy / (2 * np.pi))

    def copy(self) -> 'WaveFunction2D':
        return WaveFunction2D(self.grid, self.psi.copy())

    @classmethod
    def gaussian(cls, grid: Grid2D, x0: float, y0: float, k0_x: float, k0_y: float, sigma_x: float, sigma_y: float) -> 'WaveFunction2D':
        psi_x = np.exp(-(grid.X - x0) ** 2 / (4.0 * sigma_x ** 2)) * np.exp(1j * k0_x * grid.X)
        psi_y = np.exp(-(grid.Y - y0) ** 2 / (4.0 * sigma_y ** 2)) * np.exp(1j * k0_y * grid.Y)
        psi = psi_x * psi_y
        wf = cls(grid, psi)
        wf.normalize()
        return wf
