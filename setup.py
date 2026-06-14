from setuptools import setup, find_packages

setup(
    name="quantumlab",
    version="1.0.0",
    description="Research-grade quantum simulation framework",
    author="rasidi3112",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.22",
        "scipy>=1.8",
        "matplotlib>=3.5",
        "pyyaml>=6.0",
    ],
    extras_require={
        "full": [
            "h5py>=3.7",
            "plotly>=5.10",
            "numba>=0.56",
            "Pillow>=9.0",
            "imageio>=2.20",
            "imageio-ffmpeg>=0.4",
            "PyQt6>=6.4",
        ],
        "gpu": [
            "cupy>=11.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "quantumlab-gui=gui.app:main",
        ],
    },
)
