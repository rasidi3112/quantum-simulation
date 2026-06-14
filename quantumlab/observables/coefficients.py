import numpy as np
from quantumlab.core.wavefunction import WaveFunction1D, WaveFunction2D

def transmission_coefficient(wf, barrier_position: float=0.0) -> float:
    prob = wf.probability_density
    if isinstance(wf, WaveFunction1D):
        mask = wf.grid.x >= barrier_position
        return float(np.sum(prob[mask]) * wf.grid.dx)
    elif isinstance(wf, WaveFunction2D):
        mask = wf.grid.X >= barrier_position
        return float(np.sum(prob[mask]) * wf.grid.dx * wf.grid.dy)
    else:
        raise TypeError('Unsupported wavefunction type')

def reflection_coefficient(wf, barrier_position: float=0.0) -> float:
    prob = wf.probability_density
    if isinstance(wf, WaveFunction1D):
        mask = wf.grid.x < barrier_position
        return float(np.sum(prob[mask]) * wf.grid.dx)
    elif isinstance(wf, WaveFunction2D):
        mask = wf.grid.X < barrier_position
        return float(np.sum(prob[mask]) * wf.grid.dx * wf.grid.dy)
    else:
        raise TypeError('Unsupported wavefunction type')
