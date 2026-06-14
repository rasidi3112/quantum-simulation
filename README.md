# QuantumLab: Research-Grade Quantum Simulation Framework

A high-performance, modular Python library for numerical quantum mechanics simulations. QuantumLab solves the 1D Time-Dependent Schrödinger Equation (TDSE) using the Split-Step Fourier Method (SSFM) to model quantum wave packet dynamics with high physical fidelity.

---

## Key Features

- **Unitary Time-Step Solver**: High-precision split-step solver conserving the total probability norm to machine precision ($< 10^{-12}$).
- **Physical Observables Module**: Real-time evaluation of expectation values ($\langle x \rangle$, $\langle p \rangle$, $\langle E \rangle$, $\langle T \rangle$, $\langle V \rangle$) and quantum uncertainties ($\Delta x$, $\Delta p$) proving Heisenberg's uncertainty principle.
- **Modular Potential Registry**:
  - **Barriers**: Gaussian, Rectangular, Step, Multiple, and Resonant Tunneling Diodes (RTD).
  - **Wells**: Infinite Square Well, Finite Square Well, and quartic Double Well.
  - **Oscillators**: Harmonic Oscillator.
  - **Periodic**: Sine-squared lattice (Crystal Potential).
  - **Disorder**: Cell-based random potential for Anderson localization.
  - **Custom**: User-defined Python callable functions.
- **Aesthetic Plotting**: Publication-ready scientific plotting with LaTeX markup, custom typography, light/dark themes, and dual-space (position and momentum side-by-side) analyses.
- **3D Space-Time Rendering**: Premium 3D surface visualizations of probability density evolution.

---

## Example Gallery

The framework includes several pre-built simulation scripts under the `examples/` directory. Running them generates high-resolution figures in the `output/` directory:

### 1. Gaussian Barrier Scattering (`examples/01_gaussian_barrier.py`)
Propagates a wave packet towards a Gaussian potential barrier, resolving reflection ($R$) and transmission ($T$) coefficients.
*   **Final State Plot**: `output/01_gaussian_barrier_final.png`
*   **3D Space-Time Surface**: `output/01_gaussian_barrier_spacetime.png`
*   **Dual Space Analysis**: `output/01_gaussian_barrier_dual.png`

### 2. Harmonic Oscillator (`examples/02_harmonic_oscillator.py`)
Simulates a coherent state wave packet oscillating back and forth in a parabolic well, demonstrating exact total energy conservation.
*   **Final State Plot**: `output/02_harmonic_oscillator_final.png`
*   **3D Space-Time Surface**: `output/02_harmonic_oscillator_spacetime.png`

### 3. Double Well Tunneling (`examples/03_double_well.py`)
Illustrates quantum tunneling and wave packet oscillations between two symmetric wells separated by a central potential barrier.
*   **Final State Plot**: `output/03_double_well_final.png`
*   **3D Space-Time Surface**: `output/03_double_well_spacetime.png`

### 4. Multiple Barrier Scattering (`examples/04_multiple_barriers.py`)
Models wave packet splitting and high-frequency interference fringes as the packet scatters off multiple rectangular barriers.
*   **Final State Plot**: `output/04_multiple_barriers_final.png`
*   **3D Space-Time Surface**: `output/04_multiple_barriers_spacetime.png`

---

## Installation & Usage

### 1. Install Dependencies
Install QuantumLab in development/editable mode:
```bash
pip install -e .
```

### 2. Run Simulations
Run any example script directly:
```bash
python examples/01_gaussian_barrier.py
python examples/02_harmonic_oscillator.py
python examples/03_double_well.py
python examples/04_multiple_barriers.py
```

### 3. Run Verification Tests
Execute the unit test suite to verify physical accuracy and energy conservation:
```bash
pytest tests/ -v
```
