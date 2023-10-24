import numpy as np
from clusterfock.basis import Basis
from clusterfock.cc.coupledcluster import CoupledCluster
from clusterfock.cc.parameter import CoupledClusterParameter

from clusterfock.cc.rhs.t_CCSD import amplitudes_ccsd
from clusterfock.cc.rhs.l_CCSD import lambda_amplitudes_ccsd
from clusterfock.cc.rhs.t_inter_CCSD import amplitudes_intermediates_ccsd
from clusterfock.cc.rhs.l_inter_CCSD import lambda_amplitudes_intermediates_ccsd

from clusterfock.cc.densities.l_CCSD import one_body_density, two_body_density


class GCCSD(CoupledCluster):
    def __init__(self, basis: Basis, intermediates: bool = True):
        assert not basis.restricted, "CCD can not deal with restricted basis"

        t_orders = [1, 2]
        l_orders = [1, 2]
        super().__init__(basis, t_orders, l_orders)

        self.t_rhs = amplitudes_intermediates_ccsd if intermediates else amplitudes_ccsd
        self.l_rhs = (
            lambda_amplitudes_intermediates_ccsd if intermediates else lambda_amplitudes_ccsd
        )

    def _next_t_iteration(self, t: CoupledClusterParameter) -> dict:
        basis = self.basis

        rhs1, rhs2 = self.t_rhs(
            t1=t[1],
            t2=t[2],
            u=basis.u,
            f=self._f,
            v=basis.v,
            o=basis.o,
        )

        rhs = CoupledClusterParameter(t.orders, t.N, t.M)
        rhs.initialize_dicts({1: rhs1, 2: rhs2})

        return rhs

    def _next_l_iteration(
        self, t: CoupledClusterParameter, l: CoupledClusterParameter
    ) -> CoupledClusterParameter:
        basis = self.basis

        rhs1, rhs2 = self.l_rhs(
            t1=t[1],
            t2=t[2],
            l1=l[1],
            l2=l[2],
            u=basis.u,
            f=self._f,
            v=basis.v,
            o=basis.o,
        )

        rhs = CoupledClusterParameter(l.orders, l.N, l.M)
        rhs.initialize_dicts({1: rhs1, 2: rhs2})

        return rhs

    def _evaluate_cc_energy(self, t: CoupledClusterParameter) -> float:
        u, o, v = self.basis.u, self.basis.o, self.basis.v
        t1, t2 = t[1], t[2]

        e = 0
        e += np.einsum("ia,ai->", self._f[o, v], t1, optimize=True)
        e += np.einsum("abij,ijab->", t2, u[o, o, v, v], optimize=True) / 4
        e += np.einsum("ai,bj,ijab->", t1, t1, u[o, o, v, v], optimize=True) / 2

        return e

    def _calculate_one_body_density(self) -> np.ndarray:
        basis = self.basis
        rho = np.zeros((basis.L, basis.L), dtype=basis.dtype)

        l1, t1 = self._l[1], self._t[1]
        l2, t2 = self._l[2], self._t[2]
        o, v = basis.o, basis.v

        rho = one_body_density(rho, t1, t2, l1, l2, o, v)

        return rho

    def _calculate_two_body_density(self) -> np.ndarray:
        basis = self.basis
        rho = np.zeros((basis.L, basis.L, basis.L, basis.L), dtype=basis.dtype)

        l1, t1 = self._l[1], self._t[1]
        l2, t2 = self._l[2], self._t[2]
        o, v = basis.o, basis.v

        rho = two_body_density(rho, t1, t2, l1, l2, o, v)

        return rho
