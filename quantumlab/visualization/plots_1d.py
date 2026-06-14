import numpy as np
import matplotlib.pyplot as plt
from quantumlab.core.wavefunction import WaveFunction1D
from quantumlab.visualization.style import set_style
from quantumlab.observables.momentum import get_momentum_probability_density

def plot_wavefunction_1d(wf: WaveFunction1D, potential=None, title: str='Wave Function State', save_path: str=None, show: bool=True, theme: str='light'):
    set_style(theme)
    fig, ax = plt.subplots(figsize=(10, 6))
    x = wf.grid.x
    prob = wf.probability_density
    real_psi = np.real(wf.psi)
    if potential is not None:
        V = potential.evaluate(wf.grid)
        max_v = np.max(np.abs(V))
        max_prob = np.max(prob)
        if max_v > 0 and max_prob > 0:
            scale_factor = 0.5 * max_prob / max_v
            fill_color = '#d62728' if theme == 'light' else '#ff4a4a'
            ax.fill_between(x, 0, V * scale_factor, color=fill_color, alpha=0.15, label='Potential V(x) (scaled)')
            ax.plot(x, V * scale_factor, color=fill_color, linewidth=1.2, alpha=0.7)
    prob_color = '#1f77b4' if theme == 'light' else '#5ab3ff'
    real_color = '#7f7f7f' if theme == 'light' else '#a0a0a0'
    ax.plot(x, prob, color=prob_color, linewidth=1.8, label='Probability Density $|\\Psi(x)|^2$')
    ax.fill_between(x, 0, prob, color=prob_color, alpha=0.1)
    ax.plot(x, real_psi, color=real_color, linestyle='--', linewidth=0.8, alpha=0.6, label='Real Part $\\mathrm{Re}(\\Psi)$')
    ax.set_title(title, fontsize=12, fontweight='bold', pad=15)
    ax.set_xlabel('Position (x)', fontsize=10)
    ax.set_ylabel('Amplitude / Probability Density', fontsize=10)
    ax.grid(True)
    ax.legend(loc='upper left')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)

def plot_dual_space_1d(wf: WaveFunction1D, title: str='Dual Space Analysis', save_path: str=None, show: bool=True, theme: str='light'):
    set_style(theme)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    x = wf.grid.x
    prob_x = wf.probability_density
    prob_color = '#1f77b4' if theme == 'light' else '#5ab3ff'
    ax1.plot(x, prob_x, color=prob_color, linewidth=1.8)
    ax1.fill_between(x, 0, prob_x, color=prob_color, alpha=0.1)
    ax1.set_title('Position Space Probability Density', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Position (x)')
    ax1.set_ylabel('$|\\Psi(x)|^2$')
    ax1.grid(True)
    k, prob_k = get_momentum_probability_density(wf)
    mom_color = '#2ca02c' if theme == 'light' else '#4afc4a'
    ax2.plot(k, prob_k, color=mom_color, linewidth=1.8)
    ax2.fill_between(k, 0, prob_k, color=mom_color, alpha=0.1)
    ax2.set_title('Momentum Space Probability Density', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Momentum (k)')
    ax2.set_ylabel('$|\\Phi(k)|^2$')
    ax2.grid(True)
    fig.suptitle(title, fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)
