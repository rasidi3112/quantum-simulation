import os
import numpy as np
from quantumlab.core.grid import Grid1D
from quantumlab.core.wavefunction import WaveFunction1D
from quantumlab.potentials.wells import DoubleWell
from quantumlab.solvers.split_step_1d import SplitStep1DSolver
from quantumlab.observables.expectation import position_expectation, total_energy_expectation
from quantumlab.visualization.plots_1d import plot_wavefunction_1d
from quantumlab.visualization.plots_3d import plot_space_time_3d

def main():
    L = 30.0
    N = 512
    dt = 0.05
    num_steps = 1500
    V0 = 3.0
    barrier_distance = 3.0
    x0 = -3.0
    k0 = 0.0
    sigma = 0.7
    os.makedirs('output', exist_ok=True)
    print('--- Example 03: Quantum Double Well Tunneling ---')
    print('Initializing grid and wave function...')
    grid = Grid1D(N, -L / 2, L / 2)
    wf = WaveFunction1D.gaussian(grid, x0, k0, sigma)
    print('Setting up Double Well potential (quartic) and solver...')
    potential = DoubleWell(V0, barrier_distance)
    solver = SplitStep1DSolver(grid, potential, dt, hbar=1.0, m=1.0)
    space_time = np.zeros((num_steps + 1, N))
    space_time[0, :] = wf.probability_density
    times = [0.0]
    positions = [position_expectation(wf)]
    print(f'Running simulation for {num_steps} steps (T_max = {num_steps * dt:.1f}s)...')
    wf_current = wf
    for step in range(1, num_steps + 1):
        wf_current = solver.step(wf_current)
        space_time[step, :] = wf_current.probability_density
        t = step * dt
        x_mean = position_expectation(wf_current)
        times.append(t)
        positions.append(x_mean)
        if step % 300 == 0:
            energy = total_energy_expectation(wf_current, potential, hbar=1.0, m=1.0)
            print(f'  Step {step}/{num_steps}: Time={t:.1f}s, ⟨x⟩={x_mean:.4f}, Energy={energy:.6f}')
    print('\nSimulation complete. Plotting results...')
    plot_wavefunction_1d(wf_current, potential=potential, title='Double Well Tunneling (Final State)', save_path='output/03_double_well_final.png', show=False, theme='light')
    t_arr = np.array(times)
    plot_space_time_3d(grid, t_arr, space_time, x_range=(-8, 8), title='Quantum Double Well Space-Time Evolution (3D)', save_path='output/03_double_well_spacetime.png', show=False, theme='light')
    pos_arr = np.array(positions)
    num_crossings = np.sum((pos_arr[:-1] < 0) & (pos_arr[1:] >= 0)) + np.sum((pos_arr[:-1] > 0) & (pos_arr[1:] <= 0))
    print(f'Tunneling Diagnostics:')
    print(f'  Initial position ⟨x⟩(0): {positions[0]:.4f} (Left Well)')
    print(f'  Final position ⟨x⟩(T)  : {positions[-1]:.4f}')
    print(f'  Number of well crossings detected: {num_crossings}')
    print("Outputs saved in the 'output/' directory.")
if __name__ == '__main__':
    main()
