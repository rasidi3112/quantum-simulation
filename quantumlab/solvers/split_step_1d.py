import numpy as np
from scipy.fft import fft, ifft
from quantumlab.solvers.base import Solver
from quantumlab.core.wavefunction import WaveFunction1D

class SplitStep1DSolver(Solver):

    def __init__(self, grid, potential, dt: float, hbar: float=1.0, m: float=1.0):
        super().__init__(grid, potential, dt, hbar, m)
        self.V = self.potential.evaluate(self.grid)
        self.T_k = self.hbar ** 2 * self.grid.k ** 2 / (2.0 * self.m)
        self.U_V = None
        self.U_T = None
        self.update_operators()

    def update_operators(self):
        self.U_V = np.exp(-1j * self.V * self.dt / (2.0 * self.hbar))
        self.U_T = np.exp(-1j * self.T_k * self.dt / self.hbar)

    def step(self, wavefunction: WaveFunction1D) -> WaveFunction1D:
        psi = wavefunction.psi.copy()
        psi *= self.U_V
        psi_k = fft(psi)
        psi_k *= self.U_T
        psi = ifft(psi_k)
        psi *= self.U_V
        return WaveFunction1D(self.grid, psi)
