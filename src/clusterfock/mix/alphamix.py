from clusterfock.mix import Mixer
import numpy as np


class AlphaMixer(Mixer):
    def __init__(self, alpha: float):
        self.alpha = alpha

    def __call__(self, old: np.ndarray, new: np.ndarray) -> np.ndarray:
        return (1 - self.alpha) * new + self.alpha * old
