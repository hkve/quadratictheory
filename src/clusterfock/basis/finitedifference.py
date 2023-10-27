from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np

from clusterfock.basis import Basis


class FiniteDifferenceBasisFunctions(ABC):
    def __init__(self, L: int, orthogonal: bool = False):
        self._L = L

        self.orthogonal = orthogonal

    @abstractmethod
    def _raw(self, p, x):
        raise NotImplementedError

    @abstractmethod
    def _normalization(self, p):
        raise NotImplementedError

    @abstractmethod
    def _potential(self, x):
        raise NotImplementedError

    def __getitem__(self, p):
        func = lambda x: self._normalization(p) * self._raw(p, x)
        return func


class FiniteDifferenceBasis(Basis):
    def __init__(
        self,
        L: int,
        N: int,
        phi: FiniteDifferenceBasisFunctions,
        restricted: bool = False,
        dtype=float,
    ):
        self._x = None
        self._phi = phi
        super().__init__(L, N, restricted, dtype)

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, x):
        self._x = x
