from __future__ import annotations
from clusterfock.basis import Basis
from clusterfock.cc.parameter import CoupledClusterParameter, merge_to_flat
from clusterfock.cc.coupledcluster import CoupledCluster

import numpy as np


class QuadraticCoupledCluster(CoupledCluster):
    def __call__(self, basis: Basis, t_orders: list, l_orders: list):
        super().__init__(basis, t_orders, l_orders)

    def run(self, tol: float = 1e-8, maxiters: int = 1000, vocal: bool = False) -> CoupledCluster:
        basis = self.basis
        t, l, epsinv = self._t, self._l, self._epsinv

        if t.is_empty():
            t.initialize_zero()
        if l.is_empty():
            l.initialize_zero()

        epsinv.initialize_epsilon(epsilon=np.diag(self._f), inv=True)

        iters, diff = 0, 1000
        converged = False

        while (iters < maxiters) and not converged:
            rhs_t = self._next_t_iteration(t, l)
            rhs_l = self._next_l_iteration(t, l)

            rhs_norms_t = rhs_t.norm()
            rhs_norms_l = rhs_l.norm()

            rhs_t_converged = np.all(np.array(list(rhs_norms_t.values())) < tol)
            rhs_l_converged = np.all(np.array(list(rhs_norms_l.values())) < tol)
            converged = rhs_t_converged and rhs_l_converged

            tl_flat, t_slice, l_slice = merge_to_flat(t, l)
            delta_tl_flat, _, _ = merge_to_flat(rhs_t * epsinv, rhs_l * epsinv)

            tl_next_flat = self.mixer(tl_flat, delta_tl_flat)

            t.from_flat(tl_next_flat[t_slice])
            l.from_flat(tl_next_flat[l_slice])

            iters += 1
            if vocal:
                print(f"i = {iters}, rhs_norms_t = {rhs_norms_t}, rhs_norms_l = {rhs_norms_l}")

        self._t = t
        self._l = l

        self._t_info["run"] = True
        self._t_info["iters"] = iters
        if iters < maxiters:
            self._t_info["converged"] = True

        self._l_info = self._t_info

        return self

    def initialize_amplitudes(self, t, l):
        self._t = t.copy()
        self._l = l.copy()

    def time_dependent_energy(self):
        return self.energy()
