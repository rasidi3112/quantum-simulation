"""
1D Time-Dependent Schrödinger Equation solver using the Split-Step Fourier Method.
"""
import numpy as np
from scipy.fft import fft, ifft
from quantumlab.solvers.base import Solver
from quantumlab.core.wavefunction import WaveFunction1D

class SplitStep1DSolver(Solver):
    """
    1D Split-Step Fourier Method Solver.
    """
    def __init__(self, grid, potential, dt: float, hbar: float = 1.0, m: float = 1.0):
        super().__init__(grid, potential, dt, hbar, m)
        # Evaluate potential V(x) on the grid
        self.V = self.potential.evaluate(self.grid)

        # Kinetic energy term in momentum space: T(k) = (hbar^2 * k^2) / (2 * m)
        self.T_k = (self.hbar**2 * self.grid.k**2) / (2.0 * self.m)

        # Precompute operators
        self.U_V = None
        self.U_T = None
        self.update_operators()

    def update_operators(self):
        """Precompute or update the evolution operators."""
        self.U_V = np.exp(-1j * self.V * self.dt / (2.0 * self.hbar))
        self.U_T = np.exp(-1j * self.T_k * self.dt / self.hbar)

    def step(self, wavefunction: WaveFunction1D) -> WaveFunction1D:
        """
        Evolve the wave function by a single time step dt.
        """
        psi = wavefunction.psi.copy()

        # Position space half-step
        psi *= self.U_V

        # Fourier transform to momentum space
        psi_k = fft(psi)

        # Momentum space full-step
        psi_k *= self.U_T

        # Inverse Fourier transform to position space
        psi = ifft(psi_k)

        # Position space half-step
        psi *= self.U_V

        return WaveFunction1D(self.grid, psi)
