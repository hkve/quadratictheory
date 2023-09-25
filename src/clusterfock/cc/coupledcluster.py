from __future__ import annotations
from clusterfock.basis import Basis
from clusterfock.mix import DIISMixer
from abc import ABC, abstractmethod
from functools import reduce
import operator
import numpy as np


class CoupledCluster(ABC):
    def __init__(self, basis: Basis, t_amp: list):
        self.basis = basis
        basis.calculate_fock_matrix()
        self._f = self.basis.f

        self.has_run = False
        self.converged = False
        self.mixer = DIISMixer(n_vectors=8)

        self._order_map = {"S": 1, "D": 2}
        self._t_amplitudes_orders = t_amp
        self.eps_diag = np.diag(self._f).copy()

    def run(self, tol: float = 1e-8, maxiters: int = 1000, vocal: bool = False) -> CoupledCluster:
        basis = self.basis

        self._t_amplitudes, self._t_shapes = self._allocate_amplitudes(self._t_amplitudes_orders)
        self._epsinv = self._allocate_epsinv(self._t_amplitudes_orders)

        self._iterate(tol, maxiters, vocal)

        return self

    def _iterate(self, tol: float, maxiters: int, vocal: bool):
        iters, diff = 0, 1000
        corr_energy = 0

        t = self._t_amplitudes
        while (iters < maxiters) and (diff > tol):
            t_next = self._next_t_iteration(t)

            t_next_flat = self.mixer(self._flatten_amplitudes(t), self._flatten_amplitudes(t_next))

            t_next = self._deflatten_amplitudes(t_next_flat, self._t_shapes)

            corr_energy_next = self._evaluate_cc_energy(t_next)
            diff = np.abs(corr_energy_next - corr_energy)

            corr_energy = corr_energy_next
            t = t_next
            iters += 1

            if vocal:
                print(f"i = {iters}, {corr_energy = :.4e}, {diff = :.4e}")

        self._t_amplitudes = t_next
        self.has_run = True
        self.iters = iters
        if iters < maxiters:
            self.converged = True

    @abstractmethod
    def _next_t_iteration(self, t: np.ndarray) -> np.ndarray:
        pass

    # @abstractmethod
    # def _next_l_iteration(self, l: np.ndarray) -> np.ndarray:
    #     pass

    @abstractmethod
    def _evaluate_cc_energy(self, t: np.ndarray) -> float:
        pass

    def _allocate_amplitudes(self, amps: dict | None) -> dict:
        if amps is None:
            return None

        N, M = self.basis.N, self.basis.M

        amplitudes, amplitudes_shape = {}, {}
        for order_name in amps:
            order = self._order_map[order_name]
            shape = tuple([M] * order + [N] * order)
            amplitudes_shape[order_name] = shape
            amplitudes[order_name] = np.zeros(shape, dtype=self.basis.dtype)

        return amplitudes, amplitudes_shape

    def _allocate_epsinv(self, amps: dict) -> dict:
        N, M = self.basis.N, self.basis.M

        eps_v = self.eps_diag[N:]
        eps_o = self.eps_diag[:N]
        epsinv = {}

        if "S" in amps:
            eps = -eps_v[:, None] + eps_o[None, :]
            epsinv["S"] = 1 / eps
        if "D" in amps:
            eps = (
                -eps_v[:, None, None, None]
                - eps_v[None, :, None, None]
                + eps_o[None, None, :, None]
                + eps_o[None, None, None, :]
            )
            epsinv["D"] = 1 / eps

        return epsinv

    def _flatten_amplitudes(self, amplitudes):
        return np.concatenate(tuple(t.ravel() for t in amplitudes.values()))

    def _deflatten_amplitudes(self, amp_array, amp_shapes):
        prod = lambda shape: reduce(operator.mul, shape, 1)
        sizes = [0] + [prod(shape) for shape in amp_shapes.values()]

        amplitudes = {}
        offset = 0
        for i, order in enumerate(amp_shapes.keys()):
            order_slice = slice(sizes[i], offset + sizes[i + 1])
            offset += sizes[i + 1]
            order_shape = amp_shapes[order]
            amplitudes[order] = amp_array[order_slice].reshape(order_shape)

        return amplitudes

    def energy(self, t_amplitudes: np.ndarray = None) -> float:
        if t_amplitudes is None:
            t_amplitudes = self._t_amplitudes

        return self._evaluate_cc_energy(t_amplitudes) + self.basis.energy()
