import numpy as np
import matplotlib
import os

headless = False
try:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    plt.close(fig)
except Exception:
    matplotlib.use('Agg')
    headless = True

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.fft import fft, ifft
import time

# Set professional serif font style (looks like LaTeX/scientific publication)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'dejavuserif'

L = 100.0
N = 1024
dx = L / N
x = np.linspace(-L/2, L/2, N, endpoint=False)

hbar = 1.0
m = 1.0

dt = 0.04
num_steps = 800

x0 = -25.0
sigma = 4.0
k0 = 3.5

V0 = 6.0
barrier_width = 1.5
barrier_center = 2.0
V = V0 * np.exp(-((x - barrier_center) / barrier_width)**2)

psi = np.exp(-((x - x0) ** 2) / (4.0 * sigma ** 2)) * np.exp(1j * k0 * x)
psi /= np.sqrt(np.sum(np.abs(psi)**2) * dx)

k = 2 * np.pi * np.fft.fftfreq(N, dx)
T_k = (hbar**2 * k**2) / (2.0 * m)

U_V = np.exp(-1j * V * dt / (2.0 * hbar))
U_T = np.exp(-1j * T_k * dt / hbar)

space_time_data = np.zeros((num_steps + 1, N))

if not headless:
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.style.use('default')
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    ax.set_title("1D Quantum Wave Packet Dispersion and Scattering", fontsize=12, fontweight='bold', color='black', pad=15)
    ax.set_xlabel("Position (x)", fontsize=10, color='black')
    ax.set_ylabel("Amplitude / Probability Density", fontsize=10, color='black')
    ax.set_xlim(-40, 40)
    ax.set_ylim(-0.35, 0.45)
    ax.grid(color='#e0e0e0', linestyle=':', linewidth=0.8)

    scale_factor = 0.05
    ax.fill_between(x, 0, V * scale_factor, color='#d62728', alpha=0.15, label='Potential Barrier V(x)')
    ax.plot(x, V * scale_factor, color='#d62728', linewidth=1.2, alpha=0.7)

    line_prob, = ax.plot([], [], color='#1f77b4', linewidth=1.8, label=r'Probability Density $|\Psi(x)|^2$')
    fill_prob = [ax.fill_between([], [], [], color='#1f77b4', alpha=0.1)]
    line_real, = ax.plot([], [], color='#7f7f7f', linestyle='--', linewidth=0.8, alpha=0.6, label=r'Real Part $\mathrm{Re}(\Psi)$')

    ax.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='#d3d3d3')
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, color='black', fontsize=9, 
                        bbox=dict(facecolor='white', edgecolor='#d3d3d3', boxstyle='round,pad=0.4'))

    is_running = True
    def on_close(event):
        global is_running
        is_running = False
    fig.canvas.mpl_connect('close_event', on_close)
else:
    is_running = True
    print("Running in headless mode. Saving visuals to files.")

psi_t = psi.copy()

for step in range(num_steps + 1):
    if not is_running:
        break

    prob_density = np.abs(psi_t)**2
    space_time_data[step, :] = prob_density
    
    if not headless:
        line_prob.set_data(x, prob_density)
        fill_prob[0].remove()
        fill_prob[0] = ax.fill_between(x, 0, prob_density, color='#1f77b4', alpha=0.1)
        line_real.set_data(x, np.real(psi_t))
        t = step * dt
        norm = np.sum(prob_density) * dx
        time_text.set_text(f"Time: {t:.2f} s\nStep: {step}/{num_steps}\nTotal Prob: {norm:.4f}")
        
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        time.sleep(0.002)

    if step < num_steps:
        psi_t = U_V * psi_t
        psi_k = fft(psi_t)
        psi_k = U_T * psi_k
        psi_t = ifft(psi_k)
        psi_t = U_V * psi_t

if not headless:
    plt.ioff()
    if is_running:
        output_image = "quantum_simulation.png"
        plt.savefig(output_image, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved final state plot to '{output_image}'.")
        plt.close(fig)
else:
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    ax.set_title("Quantum Wave Packet Scattering - Final State", fontsize=12, fontweight='bold', color='black', pad=15)
    ax.set_xlabel("Position (x)", fontsize=10)
    ax.set_ylabel("Amplitude / Probability Density", fontsize=10)
    ax.set_xlim(-40, 40)
    ax.set_ylim(-0.35, 0.45)
    ax.grid(color='#e0e0e0', linestyle=':', linewidth=0.8)
    
    ax.fill_between(x, 0, V * 0.05, color='#d62728', alpha=0.15, label='Potential Barrier V(x)')
    ax.plot(x, V * 0.05, color='#d62728', linewidth=1.2, alpha=0.7)
    
    ax.plot(x, np.abs(psi_t)**2, color='#1f77b4', linewidth=1.8, label=r'Probability Density $|\Psi(x)|^2$')
    ax.fill_between(x, 0, np.abs(psi_t)**2, color='#1f77b4', alpha=0.1)
    ax.plot(x, np.real(psi_t), color='#7f7f7f', linestyle='--', linewidth=0.8, alpha=0.6, label=r'Real Part $\mathrm{Re}(\Psi)$')
    
    ax.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='#d3d3d3')
    output_image = "quantum_simulation.png"
    plt.savefig(output_image, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved final state plot to '{output_image}'.")

# --- Generate 3D Space-Time Surface Plot ---
plt.style.use('default')
fig_3d = plt.figure(figsize=(11, 8))
fig_3d.patch.set_facecolor('white')
ax_3d = fig_3d.add_subplot(111, projection='3d')
ax_3d.set_facecolor('white')

x_min, x_max = -40, 40
indices = (x >= x_min) & (x <= x_max)
x_cropped = x[indices]
st_cropped = space_time_data[:, indices]

t_arr = np.linspace(0, num_steps * dt, num_steps + 1)
X, T_grid = np.meshgrid(x_cropped, t_arr)

# Plot surface with standard scientific colormap (viridis)
surf = ax_3d.plot_surface(X, T_grid, st_cropped, cmap='viridis', linewidth=0, antialiased=True, alpha=0.95)

# Style 3D grid and panes to be clean white/light gray
ax_3d.xaxis.pane.fill = False
ax_3d.yaxis.pane.fill = False
ax_3d.zaxis.pane.fill = False
ax_3d.xaxis.pane.set_edgecolor('white')
ax_3d.yaxis.pane.set_edgecolor('white')
ax_3d.zaxis.pane.set_edgecolor('white')

ax_3d.set_xlabel("Position (x)", fontsize=10, labelpad=10)
ax_3d.set_ylabel("Time (seconds)", fontsize=10, labelpad=10)
ax_3d.set_zlabel(r"Probability Density $|\Psi(x, t)|^2$", fontsize=10, labelpad=10)
ax_3d.set_title("Quantum Wave Packet Space-Time Evolution (3D Surface)", fontsize=12, fontweight='bold', pad=15)

# View angle adjustment for clear perspective of the barrier collision
ax_3d.view_init(elev=32, azim=-62)

cbar = fig_3d.colorbar(surf, ax=ax_3d, shrink=0.5, aspect=10, pad=0.1)
cbar.set_label(r"Probability Density $|\Psi|^2$", fontsize=9)

output_st_image = "quantum_space_time.png"
plt.savefig(output_st_image, dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig_3d)
print(f"Saved 3D space-time surface plot to '{output_st_image}'.")
