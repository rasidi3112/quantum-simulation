import os
import numpy as np
from quantumlab.core.grid import Grid1D
from quantumlab.core.wavefunction import WaveFunction1D
from quantumlab.potentials.barriers import MultipleBarriers
from quantumlab.solvers.split_step_1d import SplitStep1DSolver
from quantumlab.observables.expectation import position_expectation
from quantumlab.observables.coefficients import transmission_coefficient, reflection_coefficient
from quantumlab.visualization.plots_1d import plot_wavefunction_1d, plot_dual_space_1d
from quantumlab.visualization.plots_3d import plot_space_time_3d

def main():
    L = 100.0
    N = 1024
    dt = 0.04
    num_steps = 700
    x0 = -20.0
    sigma = 3.0
    k0 = 4.0
    V0 = 8.0
    width = 0.8
    positions = [-4.0, -1.5, 1.0, 3.5]
    os.makedirs('output', exist_ok=True)
    print('--- Example 04: Multiple Barrier Scattering ---')
    print('Initializing grid and wave function...')
    grid = Grid1D(N, -L / 2, L / 2)
    wf = WaveFunction1D.gaussian(grid, x0, k0, sigma)
    print('Setting up Multiple Barriers potential and solver...')
    potential = MultipleBarriers(V0, width, positions)
    solver = SplitStep1DSolver(grid, potential, dt, hbar=1.0, m=1.0)
    space_time = np.zeros((num_steps + 1, N))
    space_time[0, :] = wf.probability_density
    print(f'Running simulation for {num_steps} steps...')
    wf_current = wf
    for step in range(1, num_steps + 1):
        wf_current = solver.step(wf_current)
        space_time[step, :] = wf_current.probability_density
        if step % 200 == 0:
            norm = wf_current.norm()
            x_mean = position_expectation(wf_current)
            print(f'  Step {step}/{num_steps}: Norm={norm:.6f}, ⟨x⟩={x_mean:.3f}')
    print('\nSimulation complete. Plotting results...')
    boundary = max(positions) + width / 2.0
    R = reflection_coefficient(wf_current, boundary)
    T = transmission_coefficient(wf_current, boundary)
    print(f'Scattering Diagnostics:')
    print(f'  Reflection Coefficient R : {R:.6f}')
    print(f'  Transmission Coefficient T: {T:.6f}')
    print(f'  Total Probability R + T  : {R + T:.6f}')
    plot_wavefunction_1d(wf_current, potential=potential, title='Multiple Barrier Scattering (Final State)', save_path='output/04_multiple_barriers_final.png', show=False, theme='light')
    t_arr = np.linspace(0, num_steps * dt, num_steps + 1)
    plot_space_time_3d(grid, t_arr, space_time, x_range=(-30, 30), title='Multiple Barrier Space-Time Evolution (3D)', save_path='output/04_multiple_barriers_spacetime.png', show=False, theme='light')
    plot_dual_space_1d(wf_current, title='Multiple Barrier Scattering: Position vs Momentum Space', save_path='output/04_multiple_barriers_dual.png', show=False, theme='light')
    print("Outputs saved in the 'output/' directory.")
if __name__ == '__main__':
    main()
