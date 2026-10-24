# Getting started

## Installation

QuantumLab needs Python ≥ 3.10 and is tested on 3.10–3.14. Clone the repository and install it in editable mode:

```bash
git clone https://github.com/rasidi3112/quantum-simulation.git
cd quantum-simulation
pip install -e .
```

Optional extras:

| Extra | Installs | Needed for |
|---|---|---|
| `viz3d` | scikit-image, Plotly | 3D isosurfaces and interactive HTML |
| `full` | same as `viz3d` | everything optional except the GPU |
| `gpu` | CuPy for CUDA 12 | running on an NVIDIA GPU |
| `dev` | pytest | running the test suite |
| `docs` | MkDocs, Material, mkdocstrings | building this documentation |

```bash
pip install -e ".[viz3d,dev]"
```

MP4 export also needs the `ffmpeg` executable (`brew install ffmpeg` or `apt install ffmpeg`). GIF export needs only Pillow.

## A first simulation

A Gaussian wave packet hits a barrier. Part of it is reflected and part tunnels through:

```python
from quantumlab.core.grid import Grid1D
from quantumlab.core.wavefunction import WaveFunction1D
from quantumlab.potentials import GaussianBarrier
from quantumlab.solvers.split_operator import SplitOperatorSolver
from quantumlab.observables import transmission_coefficient, reflection_coefficient

grid = Grid1D(N=1024, x_min=-50.0, x_max=50.0)
wf = WaveFunction1D.gaussian(grid, x0=-20.0, k0=3.5, sigma=4.0)
barrier = GaussianBarrier(V0=6.0, width=1.5, position=2.0)

solver = SplitOperatorSolver(grid, barrier, dt=0.04, order=4)
wf = solver.evolve(wf, 800)

R = reflection_coefficient(wf, barrier_position=2.0)
T = transmission_coefficient(wf, barrier_position=2.0)
print(f"R = {R:.4f}, T = {T:.4f}, R + T = {R + T:.6f}")
```

The same code works in 2D and 3D: use `Grid2D`/`Grid3D` and the matching `WaveFunction2D`/`WaveFunction3D`.

## Units

QuantumLab works in natural units with ℏ = m = 1 by default (both can be passed to every solver). Lengths, times and energies are then in units set by the problem, for example the oscillator length, 1/ω and ℏω for a harmonic trap.

## Running the examples and tests

```bash
python examples/01_gaussian_barrier.py      # outputs go to output/
pytest                                       # full test suite
python benchmarks/validation_suite.py        # analytic validation report
```

The 3D examples (07, 08, 11) take a few minutes on a laptop.
