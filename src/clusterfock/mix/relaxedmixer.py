from clusterfock.mix import Mixer
import numpy as np


class RelaxedMixer(Mixer):
    def __init__(self, alpha: float):
        self.alpha = alpha

    def __call__(self, t: np.ndarray, dt: np.ndarray) -> np.ndarray:
        t_next = t + dt
        return (1 - self.alpha) * t_next + self.alpha * t
