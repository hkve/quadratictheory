from quadratictheory.mix import Mixer
import numpy as np


class RelaxedMixer(Mixer):
    def __init__(self, alpha: float):
        self.alpha = alpha

    def reset(self):
        pass

    def __call__(self, p: np.ndarray, dp: np.ndarray) -> np.ndarray:
        p_next = p + dp
        return (1 - self.alpha) * p_next + self.alpha * p
