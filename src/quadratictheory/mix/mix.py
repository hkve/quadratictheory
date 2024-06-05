import numpy as np
from abc import ABC, abstractmethod


class Mixer(ABC):
    def __init__(*args, **kwargs):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def __call__(self, p: np.ndarray, dp: np.ndarray):
        pass
