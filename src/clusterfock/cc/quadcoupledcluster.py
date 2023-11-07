from __future__ import annotations
from clusterfock.basis import Basis
from clusterfock.mix import DIISMixer
from clusterfock.cc.parameter import CoupledClusterParameter
from clusterfock.cc.coupledcluster import CoupledCluster

import numpy as np

class QuadraticCoupledCluster(CoupledCluster):
    def __call__(self, basis: Basis, t_orders: list, l_orders: list):
        super().__init__(basis, t_orders, l_orders)

    def run(
        self, tol: float = 1e-8, maxiters: int = 1000, include_l: bool = False, vocal: bool = False
    ) -> CoupledCluster:
    
        basis = self.basis

        self._t.initialize_zero(dtype=self.basis.dtype)
        self._l.initialize_zero(dtype=self.basis.dtype)
        self._epsinv.initialize_epsilon(epsilon=np.diag(self._f), inv=True)

        iters, diff = 0, 1000
        t, l, epsinv = self._t, self._l, self._epsinv
        converged = False
        self.mixer_t = DIISMixer(n_vectors=8)
        self.mixer_l = DIISMixer(n_vectors=8)

        while (iters < maxiters) and not converged:
            rhs_t = self._next_t_iteration(t, l)
            rhs_l = self._next_l_iteration(t, l)

            rhs_norms_t = rhs_t.norm()
            rhs_norms_l = rhs_l.norm()

            rhs_t_converged = np.all(np.array(list(rhs_norms_t.values())) < tol)
            rhs_l_converged = np.all(np.array(list(rhs_norms_l.values())) < tol)
            converged = rhs_t_converged and rhs_l_converged

            # Here i should merge instead but just testing with two mixers
            t_next_flat = self.mixer_t(t.to_flat(), (rhs_norms_t * epsinv).to_flat())
            l_next_flat = self.mixer_l(l.to_flat(), (rhs_norms_l * epsinv).to_flat())

            t.from_flat(t_next_flat)
            l.from_flat(l_next_flat)

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