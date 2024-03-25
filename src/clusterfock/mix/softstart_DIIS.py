from clusterfock.mix import Mixer
from clusterfock.mix import RelaxedMixer
from clusterfock.mix import DIISMixer
import numpy as np


class SoftStartDIISMixer(Mixer):
    def __init__(self, alpha: float = 0, start_DIIS_after: int = 5, n_vectors: int = 4):
        self.start_DIIS_after = start_DIIS_after

        self.alpha = alpha
        self.n_vectors = n_vectors
        self.relaxed_mixer = RelaxedMixer(alpha=alpha)
        self.diis_mixer = DIISMixer(n_vectors=n_vectors)

        self.iter = 0

    def reset(self):
        self.relaxed_mixer.reset()
        self.diis_mixer.reset()

    def __call__(self, p: np.ndarray, dp: np.ndarray) -> np.ndarray:
        current_iter = self.iter
        self.iter += 1

        if self.iter <= self.start_DIIS_after:
            print("MIX STEP")
            return self.relaxed_mixer(p, dp)
        else:
            print("DIIS STEP")
            return self.diis_mixer(p, dp)
