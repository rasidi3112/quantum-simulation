"""
YAML configuration loader for quantum simulation parameters.
"""
import yaml
import os

DEFAULT_CONFIG = {
    "units": "atomic",  # "atomic" or "si"
    "grid": {
        "N": 512,
        "x_min": -20.0,
        "x_max": 20.0,
        "N_y": 256,         # for 2D
        "y_min": -10.0,
        "y_max": 10.0,
    },
    "wavefunction": {
        "type": "gaussian",
        "x0": -5.0,
        "k0": 3.0,
        "sigma": 1.0,
        "y0": 0.0,          # for 2D
        "k0_y": 0.0,        # for 2D
        "sigma_y": 1.0,     # for 2D
    },
    "potential": {
        "type": "barrier",  # "free", "barrier", "well", "harmonic", etc.
        "V0": 5.0,
        "width": 1.0,
        "position": 0.0,
    },
    "solver": {
        "dt": 0.01,
        "t_max": 10.0,
        "absorbing_boundary": False,
        "boundary_strength": 1.0,
        "boundary_width": 2.0,
    },
    "visualization": {
        "style": "scientific",
        "live_plot": True,
        "save_plots": True,
        "save_animation": False,
        "animation_fps": 30,
        "theme": "dark",
    },
    "export": {
        "enabled": False,
        "format": "hdf5",  # "hdf5", "csv", "json"
        "output_dir": "output",
    }
}

def load_config(config_path=None):
    """
    Loads configuration from a YAML file, merging with defaults.
    """
    config = {}
    # Make a deep copy of DEFAULT_CONFIG
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
