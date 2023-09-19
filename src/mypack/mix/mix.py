from typing import Any
import numpy as np
from abc import ABC, abstractmethod


class Mixer(ABC):
    def __init__(*args, **kwargs):
        pass

    @abstractmethod
    def __call__(self, old: np.ndarray, new: np.ndarray) -> Any:
        pass
