"""
3D surface plotting utilities for space-time evolution.
"""
import numpy as np
import matplotlib.pyplot as plt
from quantumlab.visualization.style import set_style

def plot_space_time_3d(grid, t_arr: np.ndarray, space_time_data: np.ndarray, 
                       x_range: tuple = None, title: str = "Quantum Wave Packet Space-Time Evolution",
                       save_path: str = None, show: bool = True, theme: str = 'light'):
    """
    Plot space-time evolution of the probability density in a premium 3D surface plot.
    """
    set_style(theme)
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Crop along x if a sub-range is provided (avoids boundary artifacts and focuses on the action)
    x = grid.x
    if x_range is not None:
        x_min, x_max = x_range
        indices = (x >= x_min) & (x <= x_max)
        x_plot = x[indices]
        data_plot = space_time_data[:, indices]
    else:
        x_plot = x
        data_plot = space_time_data

    X, T = np.meshgrid(x_plot, t_arr)

    # Choose color map based on theme
    cmap = 'viridis' if theme == 'light' else 'plasma'
    surf = ax.plot_surface(X, T, data_plot, cmap=cmap, linewidth=0, antialiased=True, alpha=0.95)

    # Aesthetic styling for modern clean 3D grid
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    pane_color = '#444444' if theme == 'dark' else '#e0e0e0'
    ax.xaxis.pane.set_edgecolor(pane_color)
    ax.yaxis.pane.set_edgecolor(pane_color)
    ax.zaxis.pane.set_edgecolor(pane_color)

    ax.set_xlabel("Position (x)", fontsize=10, labelpad=10)
    ax.set_ylabel("Time (seconds)", fontsize=10, labelpad=10)
    ax.set_zlabel(r"Probability Density $|\Psi(x, t)|^2$", fontsize=10, labelpad=10)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=15)

    ax.view_init(elev=32, azim=-62)

    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1)
    cbar.set_label(r"Probability Density $|\Psi|^2$", fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)
