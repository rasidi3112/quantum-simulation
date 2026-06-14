"""
Expectation values of quantum observables (position, momentum, energy, uncertainty).
"""
import numpy as np
from quantumlab.core.wavefunction import WaveFunction1D, WaveFunction2D

def position_expectation(wf) -> float:
    """Compute expectation value of position: ⟨x⟩."""
    prob = wf.probability_density
    if isinstance(wf, WaveFunction1D):
        return float(np.sum(wf.grid.x * prob) * wf.grid.dx)
    elif isinstance(wf, WaveFunction2D):
        return float(np.sum(wf.grid.X * prob) * wf.grid.dx * wf.grid.dy)
    else:
        raise TypeError("Unsupported wavefunction type")

def position_squared_expectation(wf) -> float:
    """Compute expectation value of position squared: ⟨x^2⟩."""
    prob = wf.probability_density
    if isinstance(wf, WaveFunction1D):
        return float(np.sum((wf.grid.x ** 2) * prob) * wf.grid.dx)
    elif isinstance(wf, WaveFunction2D):
        return float(np.sum((wf.grid.X ** 2) * prob) * wf.grid.dx * wf.grid.dy)
    else:
        raise TypeError("Unsupported wavefunction type")

def position_uncertainty(wf) -> float:
    """Compute position uncertainty along x: Δx = sqrt(⟨x^2⟩ - ⟨x⟩^2)."""
    mean_x = position_expectation(wf)
    mean_x2 = position_squared_expectation(wf)
    variance = mean_x2 - mean_x ** 2
    return float(np.sqrt(max(variance, 0.0)))

def y_expectation(wf) -> float:
    """Compute ⟨y⟩ (for 2D only)."""
    if isinstance(wf, WaveFunction2D):
        prob = wf.probability_density
        return float(np.sum(wf.grid.Y * prob) * wf.grid.dx * wf.grid.dy)
    else:
        raise ValueError("y expectation only defined for 2D wavefunctions")

def y_squared_expectation(wf) -> float:
    """Compute ⟨y^2⟩ (for 2D only)."""
    if isinstance(wf, WaveFunction2D):
        prob = wf.probability_density
        return float(np.sum((wf.grid.Y ** 2) * prob) * wf.grid.dx * wf.grid.dy)
    else:
        raise ValueError("y^2 expectation only defined for 2D wavefunctions")

def y_uncertainty(wf) -> float:
    """Compute position uncertainty along y (for 2D only)."""
    mean_y = y_expectation(wf)
    mean_y2 = y_squared_expectation(wf)
    variance = mean_y2 - mean_y ** 2
    return float(np.sqrt(max(variance, 0.0)))

def momentum_expectation(wf, direction: str = 'x', hbar: float = 1.0) -> float:
    """Compute expectation value of momentum: ⟨p⟩."""
    phi = wf.fourier_transform()
    phi_prob = np.abs(phi) ** 2
    if isinstance(wf, WaveFunction1D):
        if direction != 'x':
            raise ValueError("1D wavefunction only supports momentum in 'x' direction")
        p = hbar * wf.grid.k
        return float(np.sum(p * phi_prob) * wf.grid.dk)
    elif isinstance(wf, WaveFunction2D):
        if direction == 'x':
            p = hbar * wf.grid.K_x
        elif direction == 'y':
            p = hbar * wf.grid.K_y
        else:
            raise ValueError("2D wavefunction supports momentum direction 'x' or 'y'")
        return float(np.sum(p * phi_prob) * wf.grid.dk_x * wf.grid.dk_y)
    else:
        raise TypeError("Unsupported wavefunction type")

def momentum_squared_expectation(wf, direction: str = 'x', hbar: float = 1.0) -> float:
    """Compute expectation value of momentum squared: ⟨p^2⟩."""
    phi = wf.fourier_transform()
    phi_prob = np.abs(phi) ** 2
    if isinstance(wf, WaveFunction1D):
        if direction != 'x':
            raise ValueError("1D wavefunction only supports momentum in 'x' direction")
        p2 = (hbar * wf.grid.k) ** 2
        return float(np.sum(p2 * phi_prob) * wf.grid.dk)
    elif isinstance(wf, WaveFunction2D):
        if direction == 'x':
            p2 = (hbar * wf.grid.K_x) ** 2
        elif direction == 'y':
            p2 = (hbar * wf.grid.K_y) ** 2
        else:
            raise ValueError("2D wavefunction supports momentum direction 'x' or 'y'")
        return float(np.sum(p2 * phi_prob) * wf.grid.dk_x * wf.grid.dk_y)
    else:
        raise TypeError("Unsupported wavefunction type")

def momentum_uncertainty(wf, direction: str = 'x', hbar: float = 1.0) -> float:
    """Compute momentum uncertainty: Δp = sqrt(⟨p^2⟩ - ⟨p⟩^2)."""
    mean_p = momentum_expectation(wf, direction, hbar)
    mean_p2 = momentum_squared_expectation(wf, direction, hbar)
    variance = mean_p2 - mean_p ** 2
    return float(np.sqrt(max(variance, 0.0)))

def potential_energy_expectation(wf, potential) -> float:
    """Compute expectation value of potential energy: ⟨V⟩."""
    V = potential.evaluate(wf.grid)
    prob = wf.probability_density
    if isinstance(wf, WaveFunction1D):
        return float(np.sum(V * prob) * wf.grid.dx)
    elif isinstance(wf, WaveFunction2D):
        return float(np.sum(V * prob) * wf.grid.dx * wf.grid.dy)
    else:
        raise TypeError("Unsupported wavefunction type")

def kinetic_energy_expectation(wf, hbar: float = 1.0, m: float = 1.0) -> float:
    """Compute expectation value of kinetic energy: ⟨T⟩ = ⟨p^2⟩ / 2m."""
    if isinstance(wf, WaveFunction1D):
        mean_p2_x = momentum_squared_expectation(wf, 'x', hbar)
        return float(mean_p2_x / (2.0 * m))
    elif isinstance(wf, WaveFunction2D):
        mean_p2_x = momentum_squared_expectation(wf, 'x', hbar)
        mean_p2_y = momentum_squared_expectation(wf, 'y', hbar)
        return float((mean_p2_x + mean_p2_y) / (2.0 * m))
    else:
        raise TypeError("Unsupported wavefunction type")

def total_energy_expectation(wf, potential, hbar: float = 1.0, m: float = 1.0) -> float:
    """Compute total energy expectation: ⟨E⟩ = ⟨T⟩ + ⟨V⟩."""
    T_exp = kinetic_energy_expectation(wf, hbar, m)
    V_exp = potential_energy_expectation(wf, potential)
    return float(T_exp + V_exp)
