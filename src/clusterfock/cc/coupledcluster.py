from __future__ import annotations
from clusterfock.basis import Basis
from clusterfock.mix import AlphaMixer
from abc import ABC, abstractmethod
import numpy as np


class CoupledCluster(ABC):
    def __init__(self, basis: Basis, t_amp: list, l_amp: list | None):
        basis.calculate_fock_matrix()
        self.basis = basis

        self.mixer = AlphaMixer(alpha=0)
        self.has_run = False
        self.converged = False

        self._t_amplitudes_orders = t_amp
        self._l_amplitudes_orders = l_amp

    def run(self, tol: float = 1e-8, maxiters: int = 1000, vocal: bool = False) -> CoupledCluster:
        basis = self.basis

        self._t_amplitudes = self._allocate_amplitudes(self._t_amplitudes_orders)
        self._l_amplitudes = self._allocate_amplitudes(self._l_amplitudes_orders)

        self._iterate_t(tol, maxiters, vocal)
        # Reset mixer(s) here
        if self._l_amplitudes is not None:
            self._iterate_l(tol, maxiters, vocal)

        return self

    def _iterate(self, tol: float, maxiters: int, vocal: bool):
        iters, diff = 0, 1000
        corr_energy = 0

        t = self._t_amplitudes
        while (iters < maxiters) and (diff > tol):
            t_next = self._next_t_iteration(t)

            # Mix here
            corr_energy_next = self._evaluate_cc_energy(t_next)
            diff = np.abs(corr_energy_next - corr_energy)

            corr_energy = corr_energy_next
            t = t_next
            iters += 1

    @abstractmethod
    def _next_t_iteration(self, t: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def _next_l_iteration(self, l: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def _evaluate_cc_energy(self, t: np.ndarray) -> float:
        pass

    def _allocate_amplitudes(self, amps: dict | None):
        if amps is None:
            return None

        N, M = self.basis.N, self.basis.M

        amplitudes = [0] * len(amps)
        for i, order in enumerate(amps):
            shape = tuple([M] * order + [N] * order)
            amplitudes[i] = np.zeros(shape)

        return amplitudes

    def energy(self, t: np.ndarray = None) -> float:
        if t is None:
            t = self.t

        return self._evaluate_cc_energy(t)
