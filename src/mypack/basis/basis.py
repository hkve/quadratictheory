from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class Basis(ABC):
    def __init__(self, L: int, N: int, restricted: bool = False, dtype=float) -> None:
        # If restricted scheme is used, all sp states are doubly occupied
        if restricted:
            self._degeneracy = 2
            assert L % 2 == 0, "#Basis function must be even in restricted scheme"
            assert N % 2 == 0, "#Particles must be even in restricted scheme"
        else:
            self._degeneracy = 1

        self._restricted = restricted
        self._orthonormal = True
        self._antisymmetric = False

        self.dtype = dtype
        self._L = L // self._degeneracy
        self._N = N // self._degeneracy
        self._M = L - N

        self._o = slice(0, self.L)
        self._v = slice(self.N, self.L)

        self._h = np.zeros(shape=(self.L, self.L), dtype=dtype)
        self._u = np.zeros(shape=(self.L, self.L, self.L, self.L), dtype=dtype)
        self._f = np.zeros(shape=(self.L, self.L), dtype=dtype)
        self._s = np.eye(self.L, dtype=dtype)

    def _calculate_fock_matrix(self):
        h, u, o, v = self.h, self.u, self.o, self.v
        self.f = self.h + np.einsum("piqi->pq", u[:, o, :, o])

    def _change_basis_one_body(self, operator, C):
        return np.einsum("ai,bj,ab->ij", C.conj(), C, operator, optimize=True)

    def _change_basis_two_body(self, operator, C):
        return np.einsum(
            "ai,bj,gk,dl,abgd->ijkl", C.conj(), C.conj(), C, C, operator, optimize=True
        )

    def copy(self):
        new_basis = Basis(L=self.L, N=self.N, restricted=self._restricted, dtype=self.dtype)

        new_basis._orthonormal = self._orthonormal
        new_basis._antisymmetric = self._antisymmetric

        new_basis.h = self.h.copy()
        new_basis.u = self.u.copy()
        new_basis.s = self.s.copy()

        return new_basis

    def change_basis(
        self, C: np.ndarray, inplace: bool = False, inverse: bool = False
    ) -> Optional[Basis]:
        if inverse:
            C = C.conj().T

        h, u, f, s = None, None, None, None

        obj = self if inplace else self.copy()

        h = obj._change_basis_one_body(h, C)
        u = obj._change_basis_two_body(h, C)
        s = obj._change_basis_one_body(s, C)
        obj._calculate_fock_matrix()

        return obj

    @property
    def L(self) -> int:
        return self._L

    @property
    def o(self) -> slice:
        return self._o

    @property
    def v(self) -> slice:
        return self._v

    @property
    def N(self) -> int:
        return self._N

    @N.setter
    def N(self, N: int) -> int:
        N = N // self._degeneracy
        assert N <= self.L, f"#Particles = {N} must be larger than #Basis functions = {self.L}"
        self._N = N // self._degeneracy

        self._o = slice(0, self.N)
        self._v = slice(self.N, self.L)
        self._calculate_fock_matrix()

    @property
    def h(self) -> np.ndarray:
        return self._h

    @h.setter
    def h(self, h: np.ndarray):
        self._h = h.astype(self.dtype)

    @property
    def u(self) -> np.ndarray:
        return self._u

    @u.setter
    def u(self, u: np.ndarray):
        self._u = u.astype(self.dtype)

    @property
    def f(self) -> np.ndarray:
        return self._f

    @f.setter
    def f(self, f: np.ndarray):
        self._f = f.astype(self.dtype)

    @property
    def s(self) -> np.ndarray:
        return self._s

    @s.setter
    def s(self, s: np.ndarray) -> np.ndarray:
        self._s = s.astype(self.dtype)
