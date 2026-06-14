import numpy as np

class Grid1D:

    def __init__(self, N: int, x_min: float, x_max: float):
        self.N = N
        self.x_min = x_min
        self.x_max = x_max
        self.L = x_max - x_min
        self.dx = self.L / N
        self.x = np.linspace(x_min, x_max, N, endpoint=False)
        self.k = 2 * np.pi * np.fft.fftfreq(N, self.dx)
        self.dk = 2 * np.pi / self.L

    @property
    def shape(self):
        return (self.N,)

class Grid2D:

    def __init__(self, N_x: int, x_min: float, x_max: float, N_y: int, y_min: float, y_max: float):
        self.N_x = N_x
        self.x_min = x_min
        self.x_max = x_max
        self.L_x = x_max - x_min
        self.dx = self.L_x / N_x
        self.x = np.linspace(x_min, x_max, N_x, endpoint=False)
        self.N_y = N_y
        self.y_min = y_min
        self.y_max = y_max
        self.L_y = y_max - y_min
        self.dy = self.L_y / N_y
        self.y = np.linspace(y_min, y_max, N_y, endpoint=False)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='ij')
        self.k_x = 2 * np.pi * np.fft.fftfreq(N_x, self.dx)
        self.k_y = 2 * np.pi * np.fft.fftfreq(N_y, self.dy)
        self.K_x, self.K_y = np.meshgrid(self.k_x, self.k_y, indexing='ij')
        self.dk_x = 2 * np.pi / self.L_x
        self.dk_y = 2 * np.pi / self.L_y

    @property
    def shape(self):
        return (self.N_x, self.N_y)
