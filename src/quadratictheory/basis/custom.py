from quadratictheory.basis import Basis
# from functools import cached_property
import numpy as np

class CustomBasis(Basis):
    def __init__(self, L: int, N: int, restricted: bool = True, dtype=float, **kwargs):
        super().__init__(L=L, N=N, restricted=restricted, dtype=dtype)

    @property
    def r(self) -> float:
        return self._r
    
    @r.setter
    def r(self, r: np.array):
        self._r = r