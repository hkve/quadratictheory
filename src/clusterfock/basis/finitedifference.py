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

    def __getitem__(self, p):
        func = lambda x: self._normalization(p) * self._raw(p, x)
        return func


class FiniteDifferenceBasis(Basis):
    def __init__(
        self,
        L: int,
        N: int,
        phi: FiniteDifferenceBasisFunctions,
        x=(-10, 10, 5000),
        restricted: bool = False,
        dtype=float,
    ):
        self.x = x
        self._phi = phi
        self._L_spatial = L // 2
        self._restricted_dummy = restricted
        super().__init__(L, N, True, dtype)

        self._r = None

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

        u = self._fill_u_all()

        h = self._add_normalization_one_body(h, normalization)
        u = self._add_normalization_two_body(u, normalization)

        s = s if orthonormal else self._add_normalization_one_body(s, normalization)

        self.h = h
        self.u = u
        self.s = s

        if not self._restricted_dummy:
            self.from_restricted()

    def _add_normalization_one_body(self, operator, n):
        return np.einsum("p,q,pq->pq", n.conj(), n, operator)

    def _add_normalization_two_body(self, operator, n):
        return np.einsum("p,q,r,s,pqrs->pqrs", n.conj(), n.conj(), n, n, operator)

    def _double_derivative(self, y: np.ndarray) -> np.ndarray:
        return np.gradient(np.gradient(y, self.x), self.x)

    def _calculate_h_diagonal_element(self, phi_p):
        kin_integrand = -0.5 * phi_p.conj() * self._double_derivative(phi_p)
        pot_integrand = self._potential(self.x) * np.abs(phi_p) ** 2

        return np.trapz(kin_integrand + pot_integrand, self.x)

    def _calculate_h_offdiagonal_element(self, phi_p, phi_q):
        kin_integrand = -0.5 * phi_p.conj() * self._double_derivative(phi_q)
        pot_integrand = self._potential(self.x) * phi_p.conj() * phi_q

        return np.trapz(kin_integrand + pot_integrand, self.x)
    
    def _calculate_r_offdiagonal_element(self, phi_p, phi_q):
        integrand = phi_p.conj() * self.x * phi_q

        return np.trapz(integrand, self.x)

    def _fill_normalization(self):
        L_spatial = self._L_spatial

        normalization = np.zeros(L_spatial, dtype=self.dtype)
        phi = self._phi

        for p in range(L_spatial):
            normalization[p] = phi._normalization(p)

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
    
    def _fill_r_all(self):
        L = self._L
        L_spatial = self._L_spatial

        r = np.zeros((L_spatial, L_spatial), dtype=self.dtype)

        for p in range(L_spatial):
            phi_p = self._phi._raw(p, self.x)
            r[p, p] = self._calculate_r_offdiagonal_element(phi_p, phi_p)

            for q in range(p + 1, L_spatial):
                phi_q = self._phi._raw(q, self.x)
                r[p, q] = self._calculate_r_offdiagonal_element(phi_p, phi_q)
                r[q, p] = r[p, q].conj()

        return r

    def _fill_u_all(self):
        L = self._L_spatial
        u = np.zeros((L, L, L, L), dtype=self.dtype)
        X, Y = np.meshgrid(self.x, self.x)
        v_tilde = self._interaction(X, Y)

        phi = self._phi

        for q in range(L):
            phi_q = phi._raw(q, self.x[:, None]).conj()
            for s in range(L):
                phi_s = phi._raw(s, self.x[:, None])
                inner = np.trapz(phi_q * v_tilde * phi_s, dx=self._dx, axis=0)

                for p in range(q, L):
                    phi_p = phi._raw(p, self.x).conj()
                    for r in range(L):
                        if p == q and r > s:
                            continue

                        phi_r = phi._raw(r, self.x)
                        u[p, q, r, s] = np.trapz(phi_p * inner * phi_r, dx=self._dx, axis=0)
                        u[q, p, s, r] = u[p, q, r, s]

        return u

    def density(self, rho, x=None):
        if x is None:
            x = self.x

        phi = self._phi
        phi_x = np.zeros((self._L_spatial, *x.shape))
        
        for i in range(self._L_spatial):
            phi_x[i] = phi[i](x)
        if not self.restricted:
            phi_x = np.repeat(phi_x, repeats=2, axis=0)

        return np.einsum("px,pq,qx->x", phi_x.conj(), rho, phi_x)        

    @property
    def r(self):
        if self._r is None:
            r = self._fill_r_all()
            normalization = self._fill_normalization()
            r = self._add_normalization_one_body(r, normalization)
            if not self.restricted:
                r = self._add_spin_one_body(r)
            if np.linalg.norm(self.C - np.eye(self.L)) > 0.01:
                r = self._change_basis_one_body(r, self.C)

            self.r = r

        return self._r

    @r.setter
    def r(self, r):
        self._r = r

    @abstractmethod
    def _potential(self, x):
        raise NotImplementedError

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, x):
        if type(x) == tuple:
            self._x = np.linspace(*x)
            self._dx = self._x[1] - self._x[0]
        elif type(x) == np.ndarray:
            self._x = x
            if np.all(np.isclose(x, x[0])):
                self._dx = x[1] - x[0]
            else:
                self._dx = None
        else:
            raise ValueError("The grid must either be a tuple for linspace or an array")
