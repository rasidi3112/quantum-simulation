"""
Unit tests for physical potential models.
"""
import numpy as np
import pytest
from quantumlab.core.grid import Grid1D, Grid2D
from quantumlab.potentials.barriers import GaussianBarrier, RectangularBarrier, PotentialStep
from quantumlab.potentials.wells import InfiniteSquareWell, FiniteSquareWell, DoubleWell
from quantumlab.potentials.oscillator import HarmonicOscillator
from quantumlab.potentials.periodic import CrystalPotential
from quantumlab.potentials.disorder import RandomDisorder
from quantumlab.potentials.custom import CustomPotential
from quantumlab.potentials import create_potential

def test_gaussian_barrier():
    grid = Grid1D(256, -10.0, 10.0)
    V0 = 5.0
    width = 2.0
    pos = 1.0
    pot = GaussianBarrier(V0, width, pos)
    V = pot.evaluate(grid)
    assert V.shape == (256,)
    
    # Check peak value at center
    idx_peak = np.argmin(np.abs(grid.x - pos))
    assert V[idx_peak] == pytest.approx(V0, rel=1e-2)
    # Check potential drops off far away
    assert V[0] < 1e-4
    assert V[-1] < 1e-4

def test_rectangular_barrier():
    grid = Grid1D(256, -10.0, 10.0)
    V0 = 4.0
    width = 2.0
    pos = 0.0
    pot = RectangularBarrier(V0, width, pos)
    V = pot.evaluate(grid)
    
    # Outside barrier (abs(x) > 1.0) should be 0.0
    assert np.all(V[grid.x > 1.05] == 0.0)
    assert np.all(V[grid.x < -1.05] == 0.0)
    # Inside barrier (abs(x) < 1.0) should be V0
    assert np.all(V[np.abs(grid.x) < 0.95] == V0)

def test_harmonic_oscillator():
    grid = Grid1D(128, -5.0, 5.0)
    omega = 2.0
    m = 1.0
    pot = HarmonicOscillator(omega, m, position=0.0)
    V = pot.evaluate(grid)
    
    # V(x) = 0.5 * m * w^2 * x^2
    V_analytical = 0.5 * m * (omega ** 2) * (grid.x ** 2)
    assert np.allclose(V, V_analytical)

def test_custom_potential():
    grid = Grid1D(128, -5.0, 5.0)
    func = lambda x: np.sin(x)
    pot = CustomPotential(func)
    V = pot.evaluate(grid)
    assert np.allclose(V, np.sin(grid.x))

def test_potential_factory():
    pot = create_potential("finite_well", V0=10.0, width=3.0)
    assert isinstance(pot, FiniteSquareWell)
    assert pot.V0 == 10.0
    assert pot.width == 3.0
    
    with pytest.raises(ValueError):
        create_potential("nonexistent_potential")
