"""
Wave function representations and operations in 1D and 2D.
"""
import numpy as np
from quantumlab.core.grid import Grid1D, Grid2D

class WaveFunction1D:
    """
    1D Wave Function representation.
    """
    def __init__(self, grid: Grid1D, psi: np.ndarray = None):
        self.grid = grid
        if psi is None:
            self.psi = np.zeros(grid.N, dtype=complex)
        else:
            if psi.shape != grid.shape:
                raise ValueError(f"psi shape {psi.shape} does not match grid shape {grid.shape}")
            self.psi = np.array(psi, dtype=complex)

    @property
    def probability_density(self) -> np.ndarray:
        """Calculate local probability density |psi|^2."""
        return np.abs(self.psi) ** 2

    def norm(self) -> float:
        """Calculate the total probability norm of the wave function."""
        return float(np.sum(self.probability_density) * self.grid.dx)

    def normalize(self) -> "WaveFunction1D":
        """Normalize the wave function in-place so that norm is 1."""
        n = self.norm()
        if n > 0:
            self.psi /= np.sqrt(n)
        return self

    def fourier_transform(self) -> np.ndarray:
        """
        Compute the momentum-space wave function phi(k).
        Normalized such that sum(|phi|^2) * dk = 1.
        """
        return np.fft.fft(self.psi) * (self.grid.dx / np.sqrt(2 * np.pi))

    def copy(self) -> "WaveFunction1D":
        """Return a copy of the wave function."""
        return WaveFunction1D(self.grid, self.psi.copy())

    @classmethod
    def gaussian(cls, grid: Grid1D, x0: float, k0: float, sigma: float) -> "WaveFunction1D":
        """
        Create and return a normalized Gaussian wave packet:
        psi(x) = exp(-(x-x0)^2 / (4*sigma^2)) * exp(i*k0*x)
        """
        psi = np.exp(-((grid.x - x0) ** 2) / (4.0 * sigma ** 2)) * np.exp(1j * k0 * grid.x)
        wf = cls(grid, psi)
        wf.normalize()
        return wf


class WaveFunction2D:
    """
    2D Wave Function representation.
    """
    def __init__(self, grid: Grid2D, psi: np.ndarray = None):
        self.grid = grid
        if psi is None:
            self.psi = np.zeros(grid.shape, dtype=complex)
        else:
            if psi.shape != grid.shape:
                raise ValueError(f"psi shape {psi.shape} does not match grid shape {grid.shape}")
            self.psi = np.array(psi, dtype=complex)

    @property
    def probability_density(self) -> np.ndarray:
        """Calculate local probability density |psi|^2."""
        return np.abs(self.psi) ** 2

    def norm(self) -> float:
        """Calculate the total probability norm of the wave function."""
        return float(np.sum(self.probability_density) * self.grid.dx * self.grid.dy)

    def normalize(self) -> "WaveFunction2D":
        """Normalize the wave function in-place so that norm is 1."""
        n = self.norm()
        if n > 0:
            self.psi /= np.sqrt(n)
        return self

    def fourier_transform(self) -> np.ndarray:
        """
        Compute the momentum-space wave function phi(kx, ky).
        Normalized such that sum(|phi|^2) * dkx * dky = 1.
        """
        return np.fft.fft2(self.psi) * (self.grid.dx * self.grid.dy / (2 * np.pi))

    def copy(self) -> "WaveFunction2D":
        """Return a copy of the wave function."""
        return WaveFunction2D(self.grid, self.psi.copy())

    @classmethod
    def gaussian(cls, grid: Grid2D, x0: float, y0: float, k0_x: float, k0_y: float, sigma_x: float, sigma_y: float) -> "WaveFunction2D":
        """
        Create and return a normalized 2D Gaussian wave packet.
        """
        psi_x = np.exp(-((grid.X - x0) ** 2) / (4.0 * sigma_x ** 2)) * np.exp(1j * k0_x * grid.X)
        psi_y = np.exp(-((grid.Y - y0) ** 2) / (4.0 * sigma_y ** 2)) * np.exp(1j * k0_y * grid.Y)
        psi = psi_x * psi_y
        wf = cls(grid, psi)
        wf.normalize()
        return wf
