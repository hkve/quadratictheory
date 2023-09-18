from __future__ import annotations
from mypack.basis.basis import Basis
from abc import ABC, abstractmethod
import numpy as np
from scipy.linalg import eigh


class HartreeFock(ABC):
    def __init__(self, basis: Basis):
        self.basis = basis
        self.has_run = False
        self.converged = False

    def density_matrix(self, C: np.ndarray) -> np.ndarray:
        d = self.basis._degeneracy
        o, v = self.basis.o, self.basis.v
        return d * np.einsum("ai,bi->ab", C[:, o].conj(), C[:, o])

    @abstractmethod
    def evaluate_energy_scheme(self) -> float:
        pass

    @abstractmethod
    def evaluate_fock_matrix(self, rho: np.ndarray, h: np.ndarray, u: np.ndarray) -> np.ndarray:
        pass

    def run(self, tol: float = 1e-8, maxiters: int = 1000, vocal: bool = False) -> HartreeFock:
        if self.has_run:
            self.has_run = False
            self.converged = False

        basis = self.basis

        L, N = basis.L, basis.N

        C = np.eye(L, L)
        rho = self.density_matrix(C)

        eps_hf_old = np.zeros_like(np.diag(basis.h))
        eps_hf_new = np.zeros_like(np.diag(basis.h))

        iters = 0
        diff = 1

        while (iters < maxiters) and (diff > tol):
            HFmat = self.evaluate_fock_matrix(rho)
            eps_hf_new, C = eigh(HFmat, basis.s)

            rho = self.density_matrix(C)
            diff = np.mean(np.abs(eps_hf_new - eps_hf_old))
            eps_hf_old = eps_hf_new
            iters += 1

            if vocal:
                print(f"i = {iters}, mo = {eps_hf_new}")

        self.has_run = True
        if iters < maxiters:
            self.converged = True
            self._iters = iters
            self._diff = diff

        self.rho = rho
        self.C = C

        return self

    def energy(self):
        self._check_state()
        return self.evaluate_energy_scheme()

    def _check_state(self):
        if not self.has_run:
            raise RuntimeError("No Hartree-Fock calculation has been run. Perform .run() first.")
        if not self.converged:
            raise RuntimeWarning("Hartree-Fock calculation has not converged")
