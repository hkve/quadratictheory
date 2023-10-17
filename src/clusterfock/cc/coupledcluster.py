from __future__ import annotations
from clusterfock.basis import Basis
from clusterfock.mix import DIISMixer
from clusterfock.cc.parameter import CoupledClusterParameter
from abc import ABC, abstractmethod
from functools import reduce
import operator
import numpy as np


class CoupledCluster(ABC):
    @abstractmethod
    def __init__(self, basis: Basis, t_orders: list, l_orders: list = None):
        self.basis = basis
        basis.calculate_fock_matrix()
        self._f = self.basis.f

        self.has_run = False
        self.converged = False
        self.mixer = DIISMixer(n_vectors=8)

        self._t = CoupledClusterParameter(t_orders, basis.N, basis.M)
        self._l = CoupledClusterParameter(l_orders, basis.N, basis.M)
        self._epsinv = CoupledClusterParameter(t_orders, basis.N, basis.M)

        self._t_info = {"run": None, "converged": False, "iters": 0}
        self._l_info = self._t_info.copy()

        self.rho_ob = None
        self.rho_tb = None

    def run(
        self, tol: float = 1e-8, maxiters: int = 1000, include_l: bool = False, vocal: bool = False
    ) -> CoupledCluster:
        basis = self.basis

        self._t.initialize_zero(dtype=self.basis.dtype)
        self._epsinv.initialize_epsilon(epsilon=np.diag(self._f), inv=True)

        self._iterate_t(tol, maxiters, vocal)

        if include_l:
            assert (
                self._l is not None
            ), f"This scheme does not implment lambda equations, {self._l = }"

            self._l.initialize_zero(dtype=self.basis.dtype)
            self.mixer.reset()
            self._iterate_l(tol, maxiters, vocal)

        return self

    def _iterate_t(self, tol: float, maxiters: int, vocal: bool):
        iters, diff = 0, 1000
        corr_energy = 0

        t, epsinv = self._t, self._epsinv
        converged = False

        while (iters < maxiters) and not converged:
            rhs = self._next_t_iteration(t)

            rhs_norms = rhs.norm()

            if np.all(np.array(list(rhs_norms.values())) < tol):
                converged = True

            t_next_flat = self.mixer(t.to_flat(), (rhs * epsinv).to_flat())
            t.from_flat(t_next_flat)

            corr_energy = self._evaluate_cc_energy(t)
            iters += 1

            if vocal:
                print(f"i = {iters}, {corr_energy = :.6e}, rhs_norms = {rhs_norms}")

        self._t = t
        self._t_info["run"] = True
        self._t_info["iters"] = iters
        if iters < maxiters:
            self._t_info["converged"] = True

    def _iterate_l(self, tol: float, maxiters: int, vocal: bool):
        iters, diff = 0, 1000

        t, l, epsinv = self._t, self._l, self._epsinv
        converged = False

        while (iters < maxiters) and not converged:
            rhs = self._next_l_iteration(t, l)

            rhs_norms = rhs.norm()

            if np.all(np.array(list(rhs_norms.values())) < tol):
                converged = True

            l_next_flat = self.mixer(l.to_flat(), (rhs * epsinv).to_flat())
            l.from_flat(l_next_flat)

            iters += 1

            if vocal:
                print(f"i = {iters}, rhs_norms = {rhs_norms}")

        self._l = l
        self._l_info["run"] = True
        self._l_info["iters"] = iters
        if iters < maxiters:
            self._l_info["converged"] = True

    @abstractmethod
    def _next_t_iteration(self, t: CoupledClusterParameter) -> CoupledClusterParameter:
        pass

    def _next_l_iteration(
        self, t: CoupledClusterParameter, l: CoupledClusterParameter
    ) -> CoupledClusterParameter:
        pass

    @abstractmethod
    def _evaluate_cc_energy(self, t: CoupledClusterParameter) -> float:
        pass

    def _calculated_one_body_density(self) -> np.ndarray:
        raise NotImplementedError("This scheme does not implement one body densities")

    def _calculated_one_body_density(self) -> np.ndarray:
        raise NotImplementedError("This scheme does not implement two body densities")

    def _check_valid_for_densities(self):
        if self._l is None:
            raise RuntimeError("Expectation values without lambdas has not been implemented")
        if not self.l_info["run"]:
            raise RuntimeError("No lambda computation has been run. Perform .run() first.")
        if not self.l_info["converged"]:
            raise RuntimeWarning("Lambda computation did not converge")

    def one_body_density(self):
        self.rho_ob = self._calculated_one_body_density()

    def two_body_density(self):
        self.rho_tb = self._calculated_two_body_density()

    def densities(self):
        self._check_valid_for_densities()

        self.one_body_density()
        self.two_body_density()

    def energy(self, t: CoupledClusterParameter = None) -> float:
        if t is None:
            t = self._t

        return self._evaluate_cc_energy(t) + self.basis.energy()

    @property
    def t_info(self):
        return self._t_info

    @property
    def l_info(self):
        return self._l_info

    @property
    def info(self):
        merged = {
            **{"t_" + k: v for k, v in self.t_info.items()},
            **{"l_" + k: v for k, v in self.l_info.items()},
        }
        return merged
