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
from scipy.fft import fft, ifft
import time

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
    fig, ax = plt.subplots(figsize=(11, 6))
    plt.style.use('dark_background')
    fig.patch.set_facecolor('#0b0e14')
    ax.set_facecolor('#0f131a')

    ax.set_title("1D Quantum Wave Packet Live Simulation (Split-Step Fourier Method)", 
                 fontsize=14, fontweight='bold', color='#00e5ff', pad=15)
    ax.set_xlabel("Position (x)", fontsize=11, color='#e0e0e0')
    ax.set_ylabel("Amplitude / Probability Density", fontsize=11, color='#e0e0e0')
    ax.set_xlim(-40, 40)
    ax.set_ylim(-0.35, 0.45)
    ax.grid(color='#21262d', linestyle='--', linewidth=0.5)

    scale_factor = 0.05
    ax.fill_between(x, 0, V * scale_factor, color='#ff3333', alpha=0.2, label='Potential Barrier V(x) (scaled)')
    ax.plot(x, V * scale_factor, color='#ff5555', linewidth=1.5, alpha=0.8)

    line_prob, = ax.plot([], [], color='#00ffcc', linewidth=2, label=r'Probability Density $|\Psi(x)|^2$')
    fill_prob = [ax.fill_between([], [], [], color='#00ffcc', alpha=0.15)]
    line_real, = ax.plot([], [], color='#bf55ec', linestyle='--', linewidth=1.0, alpha=0.7, label=r'Real Part $\mathrm{Re}(\Psi)$')

    ax.legend(loc='upper left', frameon=True, facecolor='#0f131a', edgecolor='#21262d')
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, color='#e0e0e0', fontsize=10, 
                        bbox=dict(facecolor='#0f131a', edgecolor='#21262d', boxstyle='round,pad=0.5'))

    is_running = True
    def on_close(event):
        global is_running
        is_running = False
    fig.canvas.mpl_connect('close_event', on_close)
else:
    is_running = True
    print("Running in headless mode. Live window disabled. Visuals will be saved directly to files.")

psi_t = psi.copy()

for step in range(num_steps + 1):
    if not is_running:
        break

    prob_density = np.abs(psi_t)**2
    space_time_data[step, :] = prob_density
    
    if not headless:
        line_prob.set_data(x, prob_density)
        fill_prob[0].remove()
        fill_prob[0] = ax.fill_between(x, 0, prob_density, color='#00ffcc', alpha=0.15)
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
        plt.savefig(output_image, dpi=300, bbox_inches='tight', facecolor='#0b0e14')
        print(f"Simulation completed! Saved final state plot to '{output_image}'.")
        plt.close(fig)
else:
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor('#0b0e14')
    ax.set_facecolor('#0f131a')
    ax.set_title("Quantum Wave Packet Scattering - Final State", fontsize=14, fontweight='bold', color='#00e5ff', pad=15)
    ax.set_xlabel("Position (x)", fontsize=11)
    ax.set_ylabel("Amplitude / Probability Density", fontsize=11)
    ax.set_xlim(-40, 40)
    ax.set_ylim(-0.35, 0.45)
    ax.grid(color='#21262d', linestyle='--', linewidth=0.5)
    
    ax.fill_between(x, 0, V * 0.05, color='#ff3333', alpha=0.2, label='Potential Barrier V(x) (scaled)')
    ax.plot(x, V * 0.05, color='#ff5555', linewidth=1.5, alpha=0.8)
    
    ax.plot(x, np.abs(psi_t)**2, color='#00ffcc', linewidth=2, label=r'Probability Density $|\Psi(x)|^2$')
    ax.fill_between(x, 0, np.abs(psi_t)**2, color='#00ffcc', alpha=0.15)
    ax.plot(x, np.real(psi_t), color='#bf55ec', linestyle='--', linewidth=1.0, alpha=0.7, label=r'Real Part $\mathrm{Re}(\Psi)$')
    
    ax.legend(loc='upper left', frameon=True, facecolor='#0f131a', edgecolor='#21262d')
    output_image = "quantum_simulation.png"
    plt.savefig(output_image, dpi=300, bbox_inches='tight', facecolor='#0b0e14')
    plt.close(fig)
    print(f"Saved final state plot to '{output_image}'.")

plt.style.use('dark_background')
fig_st, ax_st = plt.subplots(figsize=(10, 8))
fig_st.patch.set_facecolor('#0b0e14')
ax_st.set_facecolor('#0f131a')

x_min, x_max = -40, 40
indices = (x >= x_min) & (x <= x_max)
x_cropped = x[indices]
st_cropped = space_time_data[:, indices]

t_arr = np.linspace(0, num_steps * dt, num_steps + 1)
X, T_grid = np.meshgrid(x_cropped, t_arr)
im = ax_st.pcolormesh(X, T_grid, st_cropped, shading='auto', cmap='inferno')

ax_st.axvline(barrier_center, color='#ff3333', linestyle=':', linewidth=1.5, label='Barrier Center')
ax_st.axvline(barrier_center - barrier_width, color='#ff3333', linestyle='--', linewidth=0.8, alpha=0.5)
ax_st.axvline(barrier_center + barrier_width, color='#ff3333', linestyle='--', linewidth=0.8, alpha=0.5)

ax_st.set_title("Quantum Wave Packet Space-Time Evolution Map", fontsize=14, fontweight='bold', color='#00e5ff', pad=15)
ax_st.set_xlabel("Position (x)", fontsize=11)
ax_st.set_ylabel("Time (seconds)", fontsize=11)
ax_st.legend(loc='upper left')

cbar = fig_st.colorbar(im, ax=ax_st, label=r'Probability Density $|\Psi(x, t)|^2$')
cbar.ax.yaxis.label.set_color('#e0e0e0')
cbar.ax.tick_params(colors='#e0e0e0')

output_st_image = "quantum_space_time.png"
plt.savefig(output_st_image, dpi=300, bbox_inches='tight', facecolor='#0b0e14')
plt.close(fig_st)
print(f"Saved space-time evolution plot to '{output_st_image}'.")
