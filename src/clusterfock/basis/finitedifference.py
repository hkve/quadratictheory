from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np

from clusterfock.basis import Basis


class FiniteDifferenceBasisFunctions(ABC):
    def __init__(self, eigenfunction: bool, orthonormal: bool):
        self._eigenfunction = eigenfunction
        self._orthonormal = orthonormal

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
        self._L_spatial = self.L if self.restricted else self.L // 2

    def setup(self):
        orthonormal = self._phi._orthonormal
        eigenfunction = self._phi._eigenfunction
        normalization = self._fill_normalization()

        if not orthonormal:
            s = self._fill_s()
        else:
            s = np.eye(self._L_spatial)

        if eigenfunction:
            h = self._fill_h_diagonal()
        else:
            h = self._fill_h_all()

        h = h * normalization
        s = s if orthonormal else s * normalization

        self.s = s
        self.h = h

        self.from_restricted()

    def _double_derivative(self, y: np.ndarray) -> np.ndarray:
        return np.gradient(np.gradient(y, self.x), self.x)

    def _calculate_h_diagonal_element(self, phi_p):
        kin_integrand = -0.5 * phi_p.conj() * self._double_derivative(phi_p)
        pot_integrand = self._phi._potential(self.x) * np.abs(phi_p) ** 2

        return np.trapz(kin_integrand + pot_integrand, self.x)

    def _calculate_h_offdiagonal_element(self, phi_p, phi_q):
        kin_integrand = -0.5 * phi_p.conj() * self._double_derivative(phi_q)
        pot_integrand = self._phi._potential(self.x) * phi_p.conj() * phi_q

        return np.trapz(kin_integrand + pot_integrand, self.x)

    def _fill_normalization(self):
        L_spatial = self._L_spatial

        normalization = np.zeros((L_spatial, L_spatial), dtype=self.dtype)
        phi = self._phi

        for p in range(L_spatial):
            p_norm = phi._normalization(p)
            normalization[p, p] = np.abs(p_norm) ** 2
            for q in range(p + 1, L_spatial):
                q_norm = phi._normalization(q)
                normalization[p, q] = p_norm.conj() * q_norm
                normalization[q, p] = normalization[p, q].conj()

        return normalization

    def _fill_s(self):
        L_spatial = self._L_spatial

        s = np.zeros((L_spatial, L_spatial), dtype=self.dtype)

        for p in range(L_spatial):
            phi_p = self._phi._raw(p, self.x)
            s[p, p] = np.trapz(phi_p.conj() * phi_p, self.x)
            for q in range(p + 1, L_spatial):
                phi_q = self._phi._raw(q, self.x)
                s[p, q] = np.trapz(phi_p.conj() * phi_q, self.x)
                s[q, p] = s[p, q]

        return s

    def _fill_h_diagonal(self):
        L = self._L
        L_spatial = self._L_spatial

        h = np.zeros((L_spatial, L_spatial), dtype=self.dtype)

        for p in range(L_spatial):
            phi_p = self._phi._raw(p, self.x)
            h[p, p] = self._calculate_h_diagonal_element(phi_p)

        return h

    def _fill_h_all(self):
        L = self._L
        L_spatial = self._L_spatial

        h = np.zeros((L_spatial, L_spatial), dtype=self.dtype)

        for p in range(L_spatial):
            phi_p = self._phi._raw(p, self.x)
            h[p, p] = self._calculate_h_diagonal_element(phi_p)

            for q in range(p + 1, L_spatial):
                phi_q = self._phi._raw(q, self.x)
                h[p, q] = self._calculate_h_offdiagonal_element(phi_p, phi_q)
                h[q, p] = h[p, q].conj()

        return h

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, x):
        self._x = x
