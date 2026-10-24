# QuantumLab: Research-Grade Quantum Simulation Framework

[![tests](https://github.com/rasidi3112/quantum-simulation/actions/workflows/tests.yml/badge.svg)](https://github.com/rasidi3112/quantum-simulation/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![docs](https://github.com/rasidi3112/quantum-simulation/actions/workflows/docs.yml/badge.svg)](https://rasidi3112.github.io/quantum-simulation/)

A high-performance, modular Python library for numerical quantum mechanics simulations. QuantumLab solves the Time-Dependent Schrödinger Equation (TDSE) in **1D, 2D, and 3D** using the Split-Step Fourier Method (SSFM) to model quantum wave packet dynamics with high physical fidelity. It also solves the nonlinear **Gross–Pitaevskii equation** for Bose–Einstein condensates and computes **stationary states** with matrix-free eigensolvers.

<p align="center">
  <img src="images/07_double_slit_3d.gif" width="49%" alt="3D double-pinhole interference">
  <img src="images/08_bec_tof_3d.gif" width="49%" alt="3D BEC time-of-flight expansion">
</p>
<p align="center"><em>Left: Young's double-pinhole experiment in 3D (|ψ|² isosurfaces). Right: a Bose–Einstein condensate released from a cigar-shaped trap, coloured by the quantum phase arg ψ.</em></p>

Every new feature is checked against exact or independent results; see [Validation](#validation).

**Documentation:** [rasidi3112.github.io/quantum-simulation](https://rasidi3112.github.io/quantum-simulation/) has a user guide, the API reference, the examples gallery and the validation report.

---

## Key Features

- **Multi-Dimensional Unitary Solvers**: High-precision split-step solvers in **1D, 2D, and 3D** that conserve total probability norm to machine precision (< 10⁻¹²).
- **Physical Observables Module**: Real-time evaluation of expectation values (⟨x⟩, ⟨p⟩, ⟨E⟩, ⟨T⟩, ⟨V⟩) and quantum uncertainties (Δx, Δp) proving Heisenberg's uncertainty principle.
- **Modular Potential Registry**:
  - **Barriers**: Gaussian (1D/2D/3D), Rectangular, Step, Multiple, and Resonant Tunneling Diodes (RTD).
  - **Wells**: Infinite Square Well, Finite Square Well, and quartic Double Well.
  - **Oscillators**: Harmonic Oscillator.
  - **Periodic**: Sine-squared lattice (Crystal Potential).
  - **Disorder**: Cell-based random potential for Anderson localization.
  - **Custom**: User-defined Python callable functions.
- **Absorbing Boundary Layer (ABL)**: Prevents artificial reflections at domain edges in all dimensions.
- **Aesthetic Plotting**: Publication-ready scientific plotting with LaTeX markup, custom typography, light/dark themes, and dual-space (position and momentum side-by-side) analyses.
- **2D Visualization**: Probability density heatmaps, snapshot grids, and orthogonal slices for 3D wavefunctions.
- **3D Space-Time Rendering**: Premium 3D surface visualizations of 1D probability density evolution over time.

### New in v1.2

- **Rotating frame** (`rotation=Ω`): H → H − ΩL_z in 2D and 3D. The rotation term is applied exactly with axis-wise FFTs and keeps the 2nd/4th-order accuracy.
- **Dissipative dynamics** (`damping=γ`): (i − γ)ℏ∂ψ/∂t = Hψ at fixed norm, used to model vortex-lattice formation.
- **Vortex detection** (`vortex_positions`), from the phase winding around every grid cell.
- **GPU acceleration** (`backend='cupy'` or `'auto'`): the whole split-operator engine runs on a CUDA GPU through CuPy. During `evolve`, ψ stays on the device. Verified on an NVIDIA Tesla T4: all 130 tests pass, and the results match the CPU to ~10⁻¹⁵.
- **Continuous integration**: GitHub Actions runs the test suite on Python 3.10–3.14 and the validation suite on every push and pull request.

### New in v1.1

- **Generic split-operator engine** (`SplitOperatorSolver`, 1D/2D/3D): 2nd-order Strang and **4th-order Yoshida** splitting, measured convergence orders 2.00 and 4.00.
- **Time-dependent potentials** V(r, t): dipole driving with pulse envelopes (`DipoleDrive`, `sin2_envelope`) and user-defined `TimeDependentCustomPotential`.
- **Gross–Pitaevskii equation**: cubic nonlinearity g|ψ|²ψ for BEC dynamics and matter-wave solitons; GP energy and chemical potential.
- **Stationary states**:
  - imaginary-time ground and excited states, including nonlinear GP ground states, which converge with O(dτ²) accuracy;
  - direct eigensolvers built on a matrix-free FFT Hamiltonian: ARPACK Lanczos, or LOBPCG with a k-space preconditioner.
- **New potentials**: anisotropic `HarmonicTrap`, multi-slit / multi-pinhole `ApertureScreen` (2D/3D), chaotic `StadiumBilliard`.
- **New observables**: probability current **j**, ⟨L_z⟩, autocorrelation C(t) for Feit–Fleck spectra, full 3D support for all expectation values.
- **3D volumetric rendering** (`IsosurfaceRenderer`, `IsosurfaceAnimation`):
  - nested |ψ|² isosurfaces, coloured by the quantum phase arg ψ, with an orbiting camera, exported to MP4/GIF;
  - an interactive, rotatable HTML export (Plotly).
- **Validation suite** (`benchmarks/validation_suite.py`): every check is compared against analytic or independent references, and the measured errors are written to a report.

---

## Architecture

```
quantumlab/
├── core/
│   ├── grid.py              # Grid1D, Grid2D, Grid3D — discretized spatial domains
│   ├── wavefunction.py      # WaveFunction1D/2D/3D — state representation
│   ├── spectral.py          # Dimension-agnostic FFT helpers (k², dV, coordinates)
│   └── absorbing_boundary.py  # Absorbing Boundary Layer (ABL)
├── solvers/
│   ├── split_step_1d.py     # 1D Split-Step Fourier solver
│   ├── split_step_2d.py     # 2D Split-Step Fourier solver
│   ├── split_step_3d.py     # 3D Split-Step Fourier solver (CPU/SciPy)
│   ├── split_step_3d_gpu.py # 3D SSFM with CuPy GPU acceleration + CPU fallback
│   ├── split_operator.py    # Generic 1D/2D/3D engine: order 2/4, V(r,t), GPE, imaginary time, rotation, damping
│   └── backends.py          # NumPy/SciPy (CPU) and CuPy (CUDA GPU) array backends
├── stationary/
│   ├── eigensolver.py       # Lanczos / preconditioned LOBPCG on a matrix-free Hamiltonian
│   ├── imaginary_time.py    # Ground & excited states, GP ground states
│   └── hamiltonian.py       # Spectral application of H (incl. g|ψ|²)
├── potentials/
│   ├── base.py              # Abstract Potential base class
│   ├── barriers.py          # Gaussian (1D/2D/3D), Rectangular, Step, Multiple, RTD
│   ├── wells.py             # InfiniteSquareWell, FiniteSquareWell, DoubleWell
│   ├── oscillator.py        # HarmonicOscillator
│   ├── periodic.py          # CrystalPotential
│   ├── disorder.py          # RandomDisorder
│   ├── custom.py            # CustomPotential (user-defined callable, 1D/2D/3D)
│   ├── traps.py             # HarmonicTrap (anisotropic, 1D/2D/3D)
│   ├── apertures.py         # ApertureScreen (multi-slit/pinhole), StadiumBilliard
│   └── time_dependent.py    # DipoleDrive, TimeDependentCustomPotential, sin2_envelope
├── observables/
│   ├── expectation.py       # Position, momentum, energy expectation values & uncertainties
│   ├── coefficients.py      # Transmission & reflection coefficients
│   ├── momentum.py          # Momentum-space wavefunction
│   ├── current.py           # Probability current, ⟨L_z⟩, autocorrelation, vortex detection
│   └── nonlinear.py         # GP interaction energy, energy functional, chemical potential
├── visualization/
│   ├── plots_1d.py          # 1D wavefunction & dual-space plots
│   ├── plots_2d.py          # 2D density heatmaps, snapshots, orthogonal 3D slices
│   ├── plots_3d.py          # 3D space-time surface rendering
│   ├── animation.py         # Time-evolution animation exporter (MP4/GIF)
│   ├── volume_3d.py         # 3D isosurfaces with phase colouring, animations, HTML export
│   └── style.py             # Theme management (light/dark)
├── config.py                # Default config loader (YAML-based)
├── constants.py             # Physical constants (atomic units)
└── runner.py                # HPC batch simulation runner (YAML-driven)
```

---

## Example Gallery

The framework includes 11 pre-built simulation scripts under the `examples/` directory. Run any example to generate high-quality scientific visualizations saved to the `output/` directory.

---

### 1. Gaussian Barrier Scattering (`examples/01_gaussian_barrier.py`)

Propagates a 1D wave packet towards a Gaussian potential barrier, resolving reflection (*R*) and transmission (*T*) coefficients with R + T = 1.000000 conservation.

**Final State Plot:**
![Gaussian Barrier Final](images/01_gaussian_barrier_final.png)

**3D Space-Time Surface:**
![Gaussian Barrier Space-Time](images/01_gaussian_barrier_spacetime.png)

**Dual Space Analysis (Position & Momentum):**
![Gaussian Barrier Dual Space](images/01_gaussian_barrier_dual.png)

**Original visualizations from the first release:**
![1D Quantum Wave Packet Dispersion and Scattering](quantum_simulation.png)

![Quantum Wave Packet Space-Time Evolution](quantum_space_time.png)

---

### 2. Harmonic Oscillator (`examples/02_harmonic_oscillator.py`)

Simulates a coherent state wave packet oscillating back and forth in a parabolic well, demonstrating exact total energy conservation (fractional drift < 10⁻⁵).

**Final State Plot:**
![Harmonic Oscillator Final](images/02_harmonic_oscillator_final.png)

**3D Space-Time Surface:**
![Harmonic Oscillator Space-Time](images/02_harmonic_oscillator_spacetime.png)

---

### 3. Double Well Tunneling (`examples/03_double_well.py`)

Illustrates quantum tunneling and wave packet oscillations between two symmetric quartic wells separated by a central potential barrier.

**Final State Plot:**
![Double Well Final](images/03_double_well_final.png)

**3D Space-Time Surface:**
![Double Well Space-Time](images/03_double_well_spacetime.png)

---

### 4. Multiple Barrier Scattering (`examples/04_multiple_barriers.py`)

Models wave packet splitting and high-frequency interference fringes as the packet scatters off multiple rectangular barriers.

**Final State Plot:**
![Multiple Barriers Final](images/04_multiple_barriers_final.png)

**3D Space-Time Surface:**
![Multiple Barriers Space-Time](images/04_multiple_barriers_spacetime.png)

**Dual Space Analysis:**
![Multiple Barriers Dual Space](images/04_multiple_barriers_dual.png)

---

### 5. 2D Gaussian Barrier Scattering (`examples/05_2d_gaussian_barrier.py`)

A 2D Gaussian wave packet scatters off an isotropic radially-symmetric Gaussian barrier on a 256×256 grid. Demonstrates 2D quantum diffraction with a heatmap of final probability density and a 6-panel snapshot grid showing the full wave packet evolution.

**Final Probability Density (Light Theme):**
![2D Gaussian Barrier Final](images/05_2d_gaussian_barrier_final.png)

**Final Probability Density (Dark Theme):**
![2D Gaussian Barrier Final Dark](images/05_2d_gaussian_barrier_final_dark.png)

**Wave Packet Evolution Snapshots:**
![2D Gaussian Barrier Snapshots](images/05_2d_gaussian_barrier_snapshots.png)

---

### 6. 3D Gaussian Barrier Scattering (`examples/06_3d_gaussian_barrier.py`)

A 3D Gaussian wave packet propagates through a spherical Gaussian barrier on a 64³ grid. Orthogonal slice plots (XY, XZ, YZ planes) visualize the 3D probability density at multiple timesteps.

**Orthogonal Slices at t = 2.00 (Step 100):**
![3D Gaussian Barrier t=100](images/06_3d_slices_t0100.png)

**Orthogonal Slices at t = 4.00 (Step 200):**
![3D Gaussian Barrier t=200](images/06_3d_slices_t0200.png)

**Orthogonal Slices at t = 6.00 (Step 300):**
![3D Gaussian Barrier t=300](images/06_3d_slices_t0300.png)

---

### 7. 3D Double-Pinhole Interference (`examples/07_3d_double_slit.py`)

Young's experiment in full 3D on a 128 × 96 × 96 grid (1.2 million points). A Gaussian packet (λ ≈ 2.1) hits an opaque screen with two circular pinholes (separation d = 5). Only about 2.4% of the probability is transmitted, so in the 3D view the density behind the screen is amplified ×40 for visibility; the caption on every frame says so.

**Validation:** the momentum distribution of the transmitted wave along y has zeros at k_y = ±(2n+1)π/d in the Fraunhofer model. The measured zeros are 0.638 and 1.890, against predictions of 0.628 and 1.885 (deviations +1.6% and +0.3%). The ideal model assumes two identical point sources, which finite pinholes in a Gaussian beam only approximate.

![3D double pinhole](images/07_double_slit_3d_final.png)

![Fringes in momentum space](images/07_double_slit_fringes.png)

---

### 8. BEC Time of Flight (`examples/08_3d_bec_time_of_flight.py`)

A condensate is prepared in a cigar-shaped trap (ω = 0.7, 2.5, 2.5; g = 600) by imaginary-time relaxation of the Gross–Pitaevskii equation, starting from the Thomas–Fermi profile. The trap is then switched off. The tightly confined directions expand fastest, so the cloud turns from a cigar into a pancake: the **aspect-ratio inversion** that was one of the first signatures of Bose–Einstein condensation in 1995. The grid has 150 × 176 × 176 = 4.6 million points, wide enough that the expanded cloud stays clear of the periodic box faces (density there < 10⁻³ of the peak). Isosurfaces are coloured by the phase arg ψ; during the flight its gradient is the expansion velocity field.

**Validation:**
- The ground state satisfies the virial theorem 2⟨T⟩ − 2⟨V⟩ + 3E_int = 0 with a residual of 1.0 × 10⁻⁴.
- Norm is conserved to 12 digits through the flight.
- The rms widths follow the Castin–Dum scaling solution to within 0.58% along x and 0.23% along y and z.
- The inversion happens at t = 1.776, against the Castin–Dum prediction of 1.772. Castin–Dum is exact only in the Thomas–Fermi limit, and the kinetic energy here is about 5% of the total.

![BEC ground state](images/08_bec_ground_state.png)

![BEC after time of flight](images/08_bec_tof_final.png)

![Aspect ratio inversion](images/08_bec_aspect_ratio.png)

An interactive, rotatable version is written to `output/08_bec_tof_interactive.html`.

---

### 9. Bright Soliton Collisions (`examples/09_soliton_collision.py`)

Two exact sech solitons of the attractive 1D Gross–Pitaevskii equation (the focusing nonlinear Schrödinger equation) collide. The equation is integrable, so the solitons pass through each other or bounce off unchanged, depending on their relative phase. With the 4th-order solver, norm and GP energy are conserved to 10 digits, and the peak heights after the collision equal the initial κ² = 1 to within grid sampling (0.9999).

![Soliton collisions](images/09_soliton_collisions.png)

![Soliton space-time](images/09_soliton_spacetime_3d.png)

---

### 10. Quantum Chaos in the Stadium Billiard (`examples/10_quantum_chaos_stadium.py`)

The first 60 eigenstates of a soft-walled Bunimovich stadium, which is classically fully chaotic, computed with the k-space preconditioned LOBPCG solver. The low states are regular. Higher states show the irregular patterns typical of chaotic billiards, alongside "bouncing-ball" modes between the flat walls (e.g. n = 17 and n = 53).

**Validation:**
- The lowest levels from LOBPCG and from imaginary-time propagation agree to about 10⁻⁶.
- The level count N(E = 15.25) = 60 compares with 60.3 from Weyl's law, A·E/2π − P·√(2E)/4π. The walls are soft, so the boundary term is only approximate.

![Stadium eigenstates](images/10_stadium_eigenstates.png)

![Weyl law](images/10_stadium_weyl.png)

### 11. Vortex Lattice in a Rotating BEC (`examples/11_rotating_bec_vortex_lattice.py`)

A condensate at rest is placed in a slightly anisotropic trap rotating at Ω = 0.8ω. In the rotating frame the dissipative Gross–Pitaevskii equation (i − γ)ℏ∂ψ/∂t = (H − ΩL_z)ψ is integrated with γ = 0.05, following Tsubota, Kasamatsu & Ueda, *Phys. Rev. A* **65**, 023603 (2002). The cloud elongates and its surface becomes unstable. Vortices then enter from the edge (t ≈ 100) and crystallise into a triangular lattice of 18 singly quantised vortices.

<p align="center">
  <img src="images/11_vortex_lattice_formation.gif" width="80%" alt="Vortex lattice formation">
</p>

The 2D lattice then seeds a 3D condensate in a cigar-shaped rotating trap (128 × 128 × 64 grid). After 3000 imaginary-time steps in the rotating frame (residual ‖(H − ΩL_z)ψ − μψ‖ = 1.6 × 10⁻³), 18 vortex lines run through the cloud. Below they are drawn as tubes where the density falls below 30% of its local average.

<p align="center">
  <img src="images/11_vortex_lines_3d.gif" width="60%" alt="3D vortex lines">
</p>

**Validation:**
- Feynman's rule for a rigidly rotating superfluid, n_v = mΩ/(πℏ), predicts a triangular-lattice spacing of 2.129.
- The measured mean nearest-neighbour spacing is 2.034 ± 0.065 in 2D (−4.5%) and 2.044 ± 0.069 in the z = 0 plane of the 3D state (−4.0%).
- Feynman's rule is exact for an infinite uniform lattice. This one has only 18 vortices in an inhomogeneous cloud, so deviations of a few percent are expected.
- The rotating-frame solver itself is checked exactly in the validation suite.

![Final 2D vortex lattice](images/11_vortex_lattice_2d.png)

![Vortex count and angular momentum](images/11_vortex_count.png)

![3D vortex lines](images/11_vortex_lines_3d.png)

---

## Validation

`python benchmarks/validation_suite.py` runs a battery of checks against exact, analytic or independent numerical references and writes the measured errors to [`benchmarks/validation_report.md`](benchmarks/validation_report.md).

| Check | Reference | Measured error |
|---|---|---|
| Free Gaussian spreading (1D) | σ(t) = σ₀√(1+(t/2σ₀²)²), exact density | 1.37e-15 |
| Temporal convergence, Strang splitting | global error ∝ dt^2 | 9.10e-04 |
| Temporal convergence, Yoshida splitting | global error ∝ dt^4 | 3.47e-03 |
| Driven harmonic oscillator, t ≤ 20 | Ehrenfest: ⟨x⟩ = classical forced trajectory | 5.60e-09 |
| Norm conservation, 3D, 500 steps | unitary evolution: ‖ψ‖ = 1 up to roundoff | 2.55e-15 |
| Energy conservation, 3D, 500 steps (order 4) | E(t) = E(0) | 2.18e-10 |
| Minimum-uncertainty Gaussian (3D) | Δr_j Δp_j = ℏ/2 for each axis | 4.60e-12 |
| Angular momentum of a vortex state | ⟨L_z⟩ = ℏ for ψ ∝ (x+iy)e^{−r²/2} | 2.22e-16 |
| Harmonic spectrum 1D, 10 levels (Lanczos) | E_n = n + ½ | 7.02e-14 |
| Anisotropic 3D harmonic spectrum, 6 levels (LOBPCG) | E = Σ ω_j (n_j + ½) | 1.42e-13 |
| 3D ground state by imaginary time | E₀ = ½(ω_x+ω_y+ω_z) | 6.95e-10 |
| Spectrum from autocorrelation (Feit–Fleck) | peaks of FFT[C(t)] at n + ½ | 1.62e-03 |
| Moving bright soliton, t = 20 | exact sech solution of the 1D NLS | 1.46e-09 |
| Virial theorem, 3D GP ground state | 2⟨T⟩ − 2⟨V⟩ + 3E_int = 0 | 2.32e-06 |
| Imaginary-time GP ground state accuracy | residual ∝ dτ² (ratio 4 when halving dτ) | 2.23e-03 |
| Coherent state in a frame rotating at Ω = 0.6 | ⟨r'⟩ = R(−Ωt)·(x₀ cos ωt, 0) | 1.89e-08 |
| Vortex eigenstate in the rotating frame | C(t) = e^{−i(2ℏω − Ωℏ)t/ℏ} | 1.15e-08 |

All 17 checks pass. A few rows report something other than a plain error:
- The two convergence rows give the deviation of the observed order (2.001 and 3.997) from the theoretical one.
- The Feit–Fleck row is limited by the spectral resolution 2π/T ≈ 0.031.
- The last row gives the relative deviation of the observed residual ratio (3.99) from 4.

---

## Installation & Usage

### 1. Install Dependencies

Install QuantumLab in development/editable mode (requires Python ≥ 3.10):

```bash
pip install -e .
```

**Core dependencies** (installed automatically):
- `numpy >= 1.22`
- `scipy >= 1.8`
- `matplotlib >= 3.5`
- `pyyaml >= 6.0`

**Optional extras** for full features (animation, GPU, interactive):

```bash
# 3D isosurface rendering (scikit-image) and interactive HTML export (Plotly)
pip install -e ".[viz3d]"

# Everything optional except GPU
pip install -e ".[full]"

# GPU acceleration via CuPy (requires CUDA)
pip install -e ".[gpu]"
```

MP4 export uses the `ffmpeg` executable (e.g. `brew install ffmpeg` or `apt install ffmpeg`). GIF export needs only Pillow.

**GPU usage.** Any `SplitOperatorSolver` (and `ground_state`) can run on a CUDA GPU:

```python
solver = SplitOperatorSolver(grid, trap, dt=0.005, nonlinearity=500.0, rotation=0.8, backend="cupy")
wf = solver.evolve(wf, 10_000)   # ψ stays on the GPU; one copy back at the end
```

`backend="auto"` picks the GPU when CuPy and a CUDA device are present and falls back to the CPU otherwise. The GPU path runs in double precision and is tested in three ways:
- In CI, without a GPU, it runs through a strict emulated device. Any accidental host/device array mixing raises an error there, and the results must match NumPy to 10⁻¹³.
- On machines with a CUDA device, `tests/test_backends.py` additionally compares real CuPy results with the CPU.
- `benchmarks/gpu_split_operator.py` reports timings and the GPU–CPU difference.

GitHub's hosted CI runners have no GPU, so the real-CuPy test is skipped there. To run it yourself on a free cloud GPU, open the validation notebook: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rasidi3112/quantum-simulation/blob/main/notebooks/gpu_validation.ipynb). Choose a GPU runtime and *Run all*. It runs the full test suite including the real-CuPy tests, benchmarks CPU vs GPU, and prints a summary.

**Measured on real hardware** (Google Colab, NVIDIA Tesla T4, CuPy 14.0.1, Python 3.13):
- The full test suite passes, **130 passed with 0 skipped**, including the 14 real-GPU tests.
- The benchmark below is a 3D rotating Gross–Pitaevskii run (order 2, g = 200, Ω = 0.7), timed over 50 steps.

| Grid | Points | Colab CPU (2 vCPU) ms/step | T4 GPU ms/step | Speed-up | max \|ψ_GPU − ψ_CPU\| |
|---|---|---|---|---|---|
| 64³ | 262,144 | 33.1 | 4.4 | 7.6× | 2.0e-15 |
| 96³ | 884,736 | 132.9 | 12.3 | 10.8× | 2.1e-15 |
| 128³ | 2,097,152 | 363.6 | 29.2 | 12.5× | 2.6e-15 |
| 160³ | 4,096,000 | 721.5 | 78.2 | 9.2× | 1.5e-15 |

The speed-up depends on the CPU it is compared with. Colab's CPU has only 2 vCPUs. On an 8-core Apple M1 the same 128³ run takes about 87 ms/step, so a T4 is roughly 3× faster than that machine.

### 2. Run Simulations

Run any example script from the project root:

```bash
# 1D simulations
python examples/01_gaussian_barrier.py
python examples/02_harmonic_oscillator.py
python examples/03_double_well.py
python examples/04_multiple_barriers.py

# 2D simulation (256×256 grid)
python examples/05_2d_gaussian_barrier.py

# 3D simulation (64³ grid)
python examples/06_3d_gaussian_barrier.py

# New in v1.1 (3D examples take a few minutes on a laptop)
python examples/07_3d_double_slit.py
python examples/08_3d_bec_time_of_flight.py
python examples/09_soliton_collision.py
python examples/10_quantum_chaos_stadium.py
python examples/11_rotating_bec_vortex_lattice.py

# CPU vs GPU timing (GPU columns appear only with CuPy and a CUDA device)
python benchmarks/gpu_split_operator.py

# Validation report
python benchmarks/validation_suite.py
```

All outputs are saved to the `output/` directory.

### 3. Run Verification Tests

Execute the full unit test suite to verify physical accuracy and energy conservation:

```bash
pytest tests/ -v
```

Test coverage includes:
- `test_solvers.py` — 1D SSFM norm conservation
- `test_2d_solver.py` — 2D SSFM norm conservation
- `test_3d_solver.py` — 3D SSFM norm conservation
- `test_observables.py` — Expectation value correctness
- `test_potentials.py` — Potential evaluation correctness
- `test_3d_observables.py` — 3D expectation values, Heisenberg bound, T + R, Parseval
- `test_runner.py` — YAML runner in 1D/2D/3D, GPU fallback, split-operator option
- `test_split_operator.py` — agreement with legacy solvers, convergence orders 2 and 4, driven oscillator, free spreading
- `test_stationary.py` — harmonic spectra in 1D/2D/3D, Lanczos vs LOBPCG, imaginary-time accuracy
- `test_gross_pitaevskii.py` — bright soliton, GP energy conservation, virial theorem in 1D/2D/3D
- `test_new_potentials_and_observables.py` — traps, apertures, stadium, current, ⟨L_z⟩, Feit–Fleck spectrum
- `test_volume_3d.py` — isosurface geometry, phase sampling, PNG/GIF/HTML export, overlays
- `test_rotation.py` — rotating-frame coherent state, vortex eigenstate, E − ΩL_z conservation, damping, vortex lattice energy
- `test_backends.py` — GPU code path against NumPy through a strict emulated device; a real CuPy comparison runs where a CUDA device exists

---

## Quick Start

```python
import numpy as np
from quantumlab.core.grid import Grid1D
from quantumlab.core.wavefunction import WaveFunction1D
from quantumlab.potentials import GaussianBarrier
from quantumlab.solvers import SplitStep1DSolver
from quantumlab.observables import transmission_coefficient, reflection_coefficient

# Setup
grid      = Grid1D(N=1024, x_min=-50.0, x_max=50.0)
wf        = WaveFunction1D.gaussian(grid, x0=-20.0, k0=3.5, sigma=4.0)
potential = GaussianBarrier(V0=6.0, width=1.5, position=2.0)
solver    = SplitStep1DSolver(grid, potential, dt=0.04)

# Evolve
for _ in range(800):
    wf = solver.step(wf)

# Analyze
R = reflection_coefficient(wf, barrier_position=2.0)
T = transmission_coefficient(wf, barrier_position=2.0)
print(f"R={R:.4f}, T={T:.4f}, R+T={R+T:.6f}")
```

### Advanced: GPE ground state, 4th-order dynamics and 3D rendering

```python
import numpy as np
from quantumlab.core.grid import Grid3D
from quantumlab.potentials import HarmonicTrap
from quantumlab.stationary import ground_state, eigenstates
from quantumlab.solvers.split_operator import SplitOperatorSolver
from quantumlab.visualization.volume_3d import IsosurfaceRenderer

grid = Grid3D(64, -8, 8, 64, -8, 8, 64, -8, 8)
trap = HarmonicTrap((1.0, 1.0, 2.0))

# Linear spectrum (exact: E = Σ ω_j (n_j + ½))
energies, states = eigenstates(grid, trap, n_states=4)

# Gross–Pitaevskii ground state by imaginary time (ψ normalised to 1, g includes N)
wf, info = ground_state(grid, trap, dtau=0.005, nonlinearity=200.0)
print(info.chemical_potential, info.residual)

# Imprint a vortex and evolve with the 4th-order solver
wf.psi *= np.exp(1j * np.arctan2(grid.Y, grid.X))
solver = SplitOperatorSolver(grid, trap, dt=0.005, order=4, nonlinearity=200.0)
wf = solver.evolve(wf, 200)

IsosurfaceRenderer(grid, levels=(0.5, 0.15, 0.03)).save(wf.psi, "vortex.png", t=solver.t)
```

---

## Citation

If you use QuantumLab in your work, please cite it. GitHub's **"Cite this repository"** button (generated from [`CITATION.cff`](CITATION.cff)) provides APA and BibTeX entries:

```bibtex
@software{quantumlab,
  author  = {Rasidi, Ahmad and Kbro1989},
  title   = {QuantumLab: Quantum Simulation Framework},
  version = {1.2.1},
  year    = {2026},
  url     = {https://github.com/rasidi3112/quantum-simulation}
}
```

## License

QuantumLab is released under the [MIT License](LICENSE).
