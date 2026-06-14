"""
Abstract base class for physical potentials.
"""
from abc import ABC, abstractmethod
import numpy as np

class Potential(ABC):
    """
    Abstract base class for potential models.
    """
    @abstractmethod
    def evaluate(self, grid) -> np.ndarray:
        """
        Evaluate the potential on the given grid.
        Returns a numpy array of shape matching grid.shape.
        """
        pass
