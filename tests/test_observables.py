"""
Unit tests for quantum mechanical observables and the uncertainty principle.
"""
import numpy as np
import pytest
from quantumlab.core.grid import Grid1D
from quantumlab.core.wavefunction import WaveFunction1D
from quantumlab.observables.expectation import (
    position_expectation, position_uncertainty,
    momentum_expectation, momentum_uncertainty
)

def test_gaussian_observables_and_uncertainty():
    """
    Test that a Gaussian wave packet satisfies analytical expectation values:
      ⟨x⟩ = x0, Δx = sigma
      ⟨p⟩ = hbar * k0, Δp = hbar / (2 * sigma)
      Δx * Δp = hbar / 2 (Heisenberg minimum uncertainty wave packet)
    """
    # Large domain and dense grid to minimize discrete truncation errors
    grid = Grid1D(1024, -50.0, 50.0)
    sigma = 3.0
    x0 = 2.0
    k0 = 4.5
    hbar = 1.0

    wf = WaveFunction1D.gaussian(grid, x0, k0, sigma)

    # 1. Position expectation values
    assert position_expectation(wf) == pytest.approx(x0, abs=1e-5)
    assert position_uncertainty(wf) == pytest.approx(sigma, abs=1e-5)

    # 2. Momentum expectation values
    assert momentum_expectation(wf, hbar=hbar) == pytest.approx(hbar * k0, abs=1e-5)
    expected_dp = hbar / (2.0 * sigma)
    assert momentum_uncertainty(wf, hbar=hbar) == pytest.approx(expected_dp, abs=1e-5)

    # 3. Heisenberg uncertainty relation
    dx = position_uncertainty(wf)
    dp = momentum_uncertainty(wf, hbar=hbar)
    assert dx * dp == pytest.approx(0.5 * hbar, abs=1e-5)
