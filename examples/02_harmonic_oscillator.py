import os
import numpy as np
from quantumlab.core.grid import Grid1D
from quantumlab.core.wavefunction import WaveFunction1D
from quantumlab.potentials.oscillator import HarmonicOscillator
from quantumlab.solvers.split_step_1d import SplitStep1DSolver
from quantumlab.observables.expectation import position_expectation, total_energy_expectation
from quantumlab.visualization.plots_1d import plot_wavefunction_1d
from quantumlab.visualization.plots_3d import plot_space_time_3d

def main():
    L = 60.0
    N = 512
    dt = 0.02
    num_steps = 600
    omega = 1.0
    m = 1.0
    x0 = -6.0
    k0 = 0.0
    sigma = 1.0 / np.sqrt(2.0)
    os.makedirs('output', exist_ok=True)
    print('--- Example 02: Quantum Harmonic Oscillator ---')
    print('Initializing grid and wave function...')
    grid = Grid1D(N, -L / 2, L / 2)
    wf = WaveFunction1D.gaussian(grid, x0, k0, sigma)
    print('Setting up Harmonic Oscillator potential and SSFM solver...')
    potential = HarmonicOscillator(omega, m)
    solver = SplitStep1DSolver(grid, potential, dt, hbar=1.0, m=m)
    space_time = np.zeros((num_steps + 1, N))
    space_time[0, :] = wf.probability_density
    times = [0.0]
    positions = [position_expectation(wf)]
    energies = [total_energy_expectation(wf, potential, hbar=1.0, m=m)]
    print(f'Running simulation for {num_steps} steps...')
    wf_current = wf
    for step in range(1, num_steps + 1):
        wf_current = solver.step(wf_current)
        space_time[step, :] = wf_current.probability_density
        t = step * dt
        x_mean = position_expectation(wf_current)
        energy = total_energy_expectation(wf_current, potential, hbar=1.0, m=m)
        times.append(t)
        positions.append(x_mean)
        energies.append(energy)
        if step % 150 == 0:
            print(f'  Step {step}/{num_steps}: Time={t:.2f}s, ⟨x⟩={x_mean:.4f}, ⟨E⟩={energy:.6f}')
    print('\nSimulation complete. Plotting results...')
    plot_wavefunction_1d(wf_current, potential=potential, title='Harmonic Oscillator Coherent State (Final State)', save_path='output/02_harmonic_oscillator_final.png', show=False, theme='light')
    t_arr = np.array(times)
    plot_space_time_3d(grid, t_arr, space_time, x_range=(-15, 15), title='Quantum Harmonic Oscillator Space-Time Evolution (3D)', save_path='output/02_harmonic_oscillator_spacetime.png', show=False, theme='light')
    e_initial = energies[0]
    e_final = energies[-1]
    e_drift = np.abs(e_final - e_initial) / e_initial
    print(f'Energy Conservation Diagnostics:')
    print(f'  Initial Energy (t=0) : {e_initial:.8f}')
    print(f'  Final Energy (t=12.0): {e_final:.8f}')
    print(f'  Fractional Energy Drift: {e_drift:.2e}')
    print("Outputs saved in the 'output/' directory.")
if __name__ == '__main__':
    main()
