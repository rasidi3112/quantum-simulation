import numpy as np
import pytest
from quantumlab.core.grid import Grid1D
from quantumlab.core.wavefunction import WaveFunction1D
from quantumlab.observables.expectation import position_expectation, position_uncertainty, momentum_expectation, momentum_uncertainty

def test_gaussian_observables_and_uncertainty():
    grid = Grid1D(1024, -50.0, 50.0)
    sigma = 3.0
    x0 = 2.0
    k0 = 4.5
    hbar = 1.0
    wf = WaveFunction1D.gaussian(grid, x0, k0, sigma)
    assert position_expectation(wf) == pytest.approx(x0, abs=1e-05)
    assert position_uncertainty(wf) == pytest.approx(sigma, abs=1e-05)
    assert momentum_expectation(wf, hbar=hbar) == pytest.approx(hbar * k0, abs=1e-05)
    expected_dp = hbar / (2.0 * sigma)
    assert momentum_uncertainty(wf, hbar=hbar) == pytest.approx(expected_dp, abs=1e-05)
    dx = position_uncertainty(wf)
    dp = momentum_uncertainty(wf, hbar=hbar)
    assert dx * dp == pytest.approx(0.5 * hbar, abs=1e-05)
