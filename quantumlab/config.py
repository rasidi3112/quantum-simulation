import yaml
import os
DEFAULT_CONFIG = {'units': 'atomic', 'grid': {'N': 512, 'x_min': -20.0, 'x_max': 20.0, 'N_y': 256, 'y_min': -10.0, 'y_max': 10.0}, 'wavefunction': {'type': 'gaussian', 'x0': -5.0, 'k0': 3.0, 'sigma': 1.0, 'y0': 0.0, 'k0_y': 0.0, 'sigma_y': 1.0}, 'potential': {'type': 'barrier', 'V0': 5.0, 'width': 1.0, 'position': 0.0}, 'solver': {'dt': 0.01, 't_max': 10.0, 'absorbing_boundary': False, 'boundary_strength': 1.0, 'boundary_width': 2.0}, 'visualization': {'style': 'scientific', 'live_plot': True, 'save_plots': True, 'save_animation': False, 'animation_fps': 30, 'theme': 'dark'}, 'export': {'enabled': False, 'format': 'hdf5', 'output_dir': 'output'}}

def load_config(config_path=None):
    config = {}
    for k, v in DEFAULT_CONFIG.items():
        if isinstance(v, dict):
            config[k] = v.copy()
        else:
            config[k] = v
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            user_config = yaml.safe_load(f)
            if user_config:
                for key, val in user_config.items():
                    if isinstance(val, dict) and key in config and isinstance(config[key], dict):
                        config[key].update(val)
                    else:
                        config[key] = val
    return config
