import numpy as np

class Basis():
    def __init__(self, L, N) -> None:
        self._L = L
        self._N = N
        self._M = L - N

        self._o = slice(0, L)
        self._v = slice(N, L)

        self._h = None
        self._f = None
        self._u = None
