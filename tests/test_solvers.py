"""
Unit tests for quantum solvers (SplitStep1DSolver) and numerical diagnostics.
"""
import numpy as np
import pytest
from quantumlab.core.grid import Grid1D
from quantumlab.core.wavefunction import WaveFunction1D
from quantumlab.potentials.barriers import GaussianBarrier
from quantumlab.solvers.split_step_1d import SplitStep1DSolver
from quantumlab.observables.expectation import total_energy_expectation

def test_norm_conservation():
    """Verify that split-step Fourier solver conserves total probability norm to 1e-12."""
    grid = Grid1D(512, -25.0, 25.0)
    wf = WaveFunction1D.gaussian(grid, x0=-5.0, k0=2.0, sigma=1.0)
    potential = GaussianBarrier(V0=5.0, width=2.0, position=0.0)
    solver = SplitStep1DSolver(grid, potential, dt=0.02)

    assert wf.norm() == pytest.approx(1.0, abs=1e-12)

    wf_current = wf
    for _ in range(100):
        wf_current = solver.step(wf_current)

    assert wf_current.norm() == pytest.approx(1.0, abs=1e-12)

def test_energy_conservation():
    """Verify that total energy expectation remains constant during unitary time evolution."""
    grid = Grid1D(512, -25.0, 25.0)
    wf = WaveFunction1D.gaussian(grid, x0=-5.0, k0=2.0, sigma=1.0)
    potential = GaussianBarrier(V0=4.0, width=1.5, position=0.0)
    solver = SplitStep1DSolver(grid, potential, dt=0.01)

    initial_energy = total_energy_expectation(wf, potential)

    wf_current = wf
    for _ in range(50):
        wf_current = solver.step(wf_current)

    final_energy = total_energy_expectation(wf_current, potential)
    # SSFM has excellent energy conservation
    assert final_energy == pytest.approx(initial_energy, rel=1e-5)
