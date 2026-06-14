import matplotlib.pyplot as plt

def set_style(theme: str='light'):
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['mathtext.fontset'] = 'dejavuserif'
    if theme == 'dark':
        plt.style.use('dark_background')
        plt.rcParams.update({'axes.facecolor': '#121212', 'figure.facecolor': '#121212', 'grid.color': '#2c2c2c', 'grid.linestyle': ':', 'grid.linewidth': 0.8, 'axes.edgecolor': '#444444', 'xtick.color': '#bbbbbb', 'ytick.color': '#bbbbbb', 'axes.labelcolor': '#ffffff', 'text.color': '#ffffff', 'legend.facecolor': '#1e1e1e', 'legend.edgecolor': '#333333'})
    else:
        plt.style.use('default')
        plt.rcParams.update({'axes.facecolor': 'white', 'figure.facecolor': 'white', 'grid.color': '#e0e0e0', 'grid.linestyle': ':', 'grid.linewidth': 0.8, 'axes.edgecolor': '#cccccc', 'xtick.color': '#333333', 'ytick.color': '#333333', 'axes.labelcolor': '#000000', 'text.color': '#000000', 'legend.facecolor': 'white', 'legend.edgecolor': '#d3d3d3'})
