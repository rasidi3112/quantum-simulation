from abc import ABC, abstractmethod
import numpy as np

class Potential(ABC):

    @abstractmethod
    def evaluate(self, grid) -> np.ndarray:
        pass
