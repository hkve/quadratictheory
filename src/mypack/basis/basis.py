from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class Basis(ABC):
    def __init__(self, L: int, N: int, restricted: bool = False, dtype=float):
        # If restricted scheme is used, all sp states are doubly occupied
        if restricted:
            self._degeneracy = 2
            assert L % 2 == 0, "#Basis function must be even in restricted scheme"
            assert N % 2 == 0, "#Particles must be even in restricted scheme"
        else:
            self._degeneracy = 1

        self.restricted = restricted
        self.orthonormal = True
        self.antisymmetric = False

        self.dtype = dtype
        self._L = L // self._degeneracy
        self._N = N // self._degeneracy
        self._M = L - N

        self._o = slice(0, self.L)
        self._v = slice(self.N, self.L)

        self._energy_shift = 0
        self._one_body_shape = (self.L, self.L)
        self._two_body_shape = (self.L, self.L, self.L, self.L)

        # self._h = np.zeros(shape=(self.L, self.L), dtype=dtype)
        # self._u = np.zeros(shape=(self.L, self.L, self.L, self.L), dtype=dtype)
        # self._f = np.zeros(shape=(self.L, self.L), dtype=dtype)
        # self._s = np.eye(self.L, dtype=dtype)
        self._h = None
        self._u = None
        self._f = None
        self._s = None

    def calculate_fock_matrix(self):
        h, u, o, v = self.h, self.u, self.o, self.v
        self.f = self.h + np.einsum("piqi->pq", u[:, o, :, o])

    def _change_basis_one_body(self, operator: np.ndarray, C: np.ndarray) -> np.ndarray:
        return np.einsum("ai,bj,ab->ij", C.conj(), C, operator, optimize=True)

    def _change_basis_two_body(self, operator: np.ndarray, C: np.ndarray) -> np.ndarray:
        return np.einsum(
            "ai,bj,gk,dl,abgd->ijkl", C.conj(), C.conj(), C, C, operator, optimize=True
        )

    def copy(self) -> Basis:
        L, N = self._degeneracy * self.L, self._degeneracy * self.N
        new_basis = Basis(L=L, N=N, restricted=self.restricted, dtype=self.dtype)
        new_basis.orthonormal = self.orthonormal
        new_basis.antisymmetric = self.antisymmetric

        new_basis.h = self.h.copy()
        new_basis.u = self.u.copy()
        new_basis.s = self.s.copy()
        new_basis.calculate_fock_matrix()

        return new_basis

    def change_basis(
        self, C: np.ndarray, inplace: bool = True, inverse: bool = False
    ) -> Optional[Basis]:
        if inverse:
            C = C.conj().T

        obj = self if inplace else self.copy()

        obj.h = obj._change_basis_one_body(obj.h, C)
        obj.u = obj._change_basis_two_body(obj.u, C)
        obj.s = obj._change_basis_one_body(obj.s, C)
        obj.calculate_fock_matrix()

        return obj

    def _antisymmetrize(self):
        self.antisymmetric = True
        self.u = self.u - self.u.transpose(1, 0, 3, 2)

    def _add_spin(self):
        self.restricted = False
        self._degeneracy = 1

        I = np.eye(2)
        I2 = np.einsum("pr, qs -> pqrs", I, I)

        self.h = np.kron(self.h, I)
        self.u = np.kron(self.u, I2)
        self.f = np.kron(self.f, I)
        self.s = np.kron(self.s, I)

        self._L = 2 * self._L
        self.N = 2 * self.N

    def from_restricted(self, inplace: bool = True):
        obj = self if inplace else self.copy()

        obj._add_spin()
        obj._antisymmetrize()

        return obj

    # Getters and setters for ints (L,N) and slices (o,v)
    @property
    def L(self) -> int:
        return self._L

    @property
    def N(self) -> int:
        return self._N

    @N.setter
    def N(self, N: int) -> int:
        assert N % 2 == 0, "#Particles must be even in restricted scheme"
        N = N // self._degeneracy
        assert N <= self.L, f"#Particles = {N} must be larger than #Basis functions = {self.L}"
        self._N = N
        self._M = self.L - self.N

        self._o = slice(0, self.N)
        self._v = slice(self.N, self.L)
        self.calculate_fock_matrix()

    @property
    def M(self) -> int:
        return self._M

    @property
    def o(self) -> slice:
        return self._o

    @property
    def v(self) -> slice:
        return self._v

    # Getters and setters for matricies h, u, f and s
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
