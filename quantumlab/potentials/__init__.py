from quantumlab.potentials.base import Potential
from quantumlab.potentials.barriers import GaussianBarrier, RectangularBarrier, PotentialStep, MultipleBarriers, ResonantTunnelingDiode
from quantumlab.potentials.wells import InfiniteSquareWell, FiniteSquareWell, DoubleWell
from quantumlab.potentials.oscillator import HarmonicOscillator
from quantumlab.potentials.periodic import CrystalPotential
from quantumlab.potentials.disorder import RandomDisorder
from quantumlab.potentials.custom import CustomPotential
POTENTIAL_REGISTRY = {'gaussian_barrier': GaussianBarrier, 'barrier': GaussianBarrier, 'rectangular_barrier': RectangularBarrier, 'potential_step': PotentialStep, 'multiple_barriers': MultipleBarriers, 'rtd': ResonantTunnelingDiode, 'infinite_well': InfiniteSquareWell, 'finite_well': FiniteSquareWell, 'double_well': DoubleWell, 'harmonic': HarmonicOscillator, 'crystal': CrystalPotential, 'disorder': RandomDisorder}

def create_potential(name: str, **kwargs) -> Potential:
    name_lower = name.lower()
    if name_lower not in POTENTIAL_REGISTRY:
        raise ValueError(f"Potential '{name}' not found. Available: {list(POTENTIAL_REGISTRY.keys())}")
    return POTENTIAL_REGISTRY[name_lower](**kwargs)
__all__ = ['Potential', 'GaussianBarrier', 'RectangularBarrier', 'PotentialStep', 'MultipleBarriers', 'ResonantTunnelingDiode', 'InfiniteSquareWell', 'FiniteSquareWell', 'DoubleWell', 'HarmonicOscillator', 'CrystalPotential', 'RandomDisorder', 'CustomPotential', 'create_potential']
