import numpy as np
from clusterfock.basis import Basis
from clusterfock.cc.coupledcluster import CoupledCluster
from clusterfock.cc.parameter import CoupledClusterParameter

from clusterfock.cc.rhs.t_CCSD import amplitudes_ccsd
from clusterfock.cc.rhs.l_CCSD import lambda_amplitudes_ccsd
from clusterfock.cc.rhs.t_inter_CCSD import amplitudes_intermediates_ccsd
from clusterfock.cc.rhs.l_inter_CCSD import lambda_amplitudes_intermediates_ccsd
from clusterfock.cc.rhs.ccsd_Gauss_Stanton import ccsd_t_Gauss_Stanton, ccsd_l_Gauss_Stanton

from clusterfock.cc.densities.l_CCSD import one_body_density, two_body_density
from clusterfock.cc.energies.e_inter_ccsd import td_energy_addition
from clusterfock.cc.weights.ccsd import (
    reference_ccsd, ket_singles_ccsd, bra_singles_ccsd, ket_doubles_ccsd, bra_doubles_ccsd
)

from clusterfock.cc.rhs.t_inter_RCCSD import amplitudes_intermediates_rccsd
from clusterfock.cc.rhs.l_inter_RCCSD import lambda_amplitudes_intermediates_rccsd

from clusterfock.cc.densities.l_RCCSD import one_body_density_restricted
from clusterfock.cc.energies.e_inter_rccsd import td_energy_addition_restricted


class GCCSD(CoupledCluster):
    def __init__(self, basis: Basis, intermediates: bool = True):
        assert not basis.restricted, "General CCSD can not deal with restricted basis"

        t_orders = [1, 2]
        l_orders = [1, 2]
        super().__init__(basis, t_orders, l_orders)

        self.t_rhs = ccsd_t_Gauss_Stanton if intermediates else amplitudes_ccsd
        # self.t_rhs = ccsd_t_Gauss_Stanton
        self.l_rhs = (
            ccsd_l_Gauss_Stanton if intermediates else lambda_amplitudes_ccsd
        )

        self.td_energy_addition = td_energy_addition

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

        rhs = CoupledClusterParameter(t.orders, t.N, t.M, dtype=t.dtype)
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

        rhs = CoupledClusterParameter(l.orders, l.N, l.M, dtype=l.dtype)
        rhs.initialize_dicts({1: rhs1, 2: rhs2})

        return rhs

    def _evaluate_cc_energy(self) -> float:
        u, o, v = self.basis.u, self.basis.o, self.basis.v
        t1, t2 = self._t[1], self._t[2]

        e = 0
        e += np.einsum("ia,ai->", self._f[o, v], t1, optimize=True)
        e += np.einsum("abij,ijab->", t2, u[o, o, v, v], optimize=True) / 4
        e += np.einsum("ai,bj,ijab->", t1, t1, u[o, o, v, v], optimize=True) / 2

        return e

    def _evaluate_tdcc_energy(self) -> float:
        t1, t2 = self._t[1], self._t[2]
        l1, l2 = self._l[1], self._l[2]
        u, o, v = self.basis.u, self.basis.o, self.basis.v
        f = self._f

        return self.td_energy_addition(t1, t2, l1, l2, u, f, o, v)

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

    def _overlap(self, t0, l0, t, l):
        psitilde_t = 1
        psitilde_t += np.einsum("ai,ai->", l[1], t0[1])
        psitilde_t -= np.einsum("ai,ai->", l[1], t[1])

        psitilde_t += 0.25 * np.einsum("abij,abij->", l[2], t0[2])
        psitilde_t -= 0.5 * np.einsum("abij,aj,bi->", l[2], t0[1], t0[1])
        psitilde_t -= np.einsum("abij,ai,bj->", l[2], t[1], t0[1])
        psitilde_t -= 0.5 * np.einsum("abij,aj,bi->", l[2], t[1], t[1])
        psitilde_t -= 0.25 * np.einsum("abij,abij->", l[2], t[2])

        psit = 1
        psit += np.einsum("ai,ai->", l0[1], t[1])
        psit -= np.einsum("ai,ai->", l0[1], t0[1])

        psit += 0.25 * np.einsum("abij,abij->", l0[2], t[2])
        psit -= 0.5 * np.einsum("abij,aj,bi->", l0[2], t[1], t[1])
        psit -= np.einsum("abij,ai,bj->", l0[2], t0[1], t[1])
        psit -= 0.5 * np.einsum("abij,aj,bi->", l0[2], t0[1], t0[1])
        psit -= 0.25 * np.einsum("abij,abij->", l0[2], t0[2])

        return psit * psitilde_t
    
    def _if_missing_use_stored(self, t1, t2, l1, l2):
        if not t1: t1 = self._t[1]
        if not t2: t2 = self._t[2]
        if not l1: l1 = self._l[1]
        if not l2: l2 = self._l[2]

        return t1, t2, l1, l2

    def reference_weights(self, t1=None, t2=None, l1=None, l2=None):
        t1, t2, l1, l2 = self._if_missing_use_stored(t1,t2,l1,l2)
        
        return reference_ccsd(t1, t2, l1, l2)
    
    def singles_weights(self, t1=None, t2=None, l1=None, l2=None):
        t1, t2, l1, l2 = self._if_missing_use_stored(t1,t2,l1,l2)
        bra = bra_singles_ccsd(t1, t2, l1, l2)
        ket = ket_singles_ccsd(t1, t2, l1, l2)

        return np.multiply(bra, ket)

    def doubles_weights(self, t1=None, t2=None, l1=None, l2=None):
        t1, t2, l1, l2 = self._if_missing_use_stored(t1,t2,l1,l2)
        bra = bra_doubles_ccsd(t1, t2, l1, l2)
        ket = ket_doubles_ccsd(t1, t2,l1,l2)

        return np.multiply(bra, ket)
    
class RCCSD(CoupledCluster):
    def __init__(self, basis: Basis, intermediates=True):
        assert basis.restricted, f"Restricted CCSD requires restricted basis"

        t_orders = [1, 2]
        l_orders = [1, 2]
        super().__init__(basis, t_orders, l_orders)

        self.t_rhs = amplitudes_intermediates_rccsd
        self.l_rhs = lambda_amplitudes_intermediates_rccsd
        self.td_energy_addition = td_energy_addition_restricted

    def _evaluate_cc_energy(self) -> float:
        t1, t2 = self._t[1], self._t[2]
        u, o, v = self.basis.u, self.basis.o, self.basis.v
        f = self._f

        E = 2 * np.einsum("ia,ai->", f[o, v], t1, optimize=True)
        E -= np.einsum("abij,ijba->", t2, u[o, o, v, v], optimize=True)
        E += 2 * np.einsum("abij,ijab->", t2, u[o, o, v, v], optimize=True)
        E -= np.einsum("ai,bj,ijba->", t1, t1, u[o, o, v, v], optimize=True)
        E += 2 * np.einsum("ai,bj,ijab->", t1, t1, u[o, o, v, v], optimize=True)

        return E

    def _evaluate_tdcc_energy(self) -> float:
        t1, t2 = self._t[1], self._t[2]
        l1, l2 = self._l[1], self._l[2]
        u, o, v = self.basis.u, self.basis.o, self.basis.v
        f = self._f

        return self.td_energy_addition(t1, t2, l1, l2, u, f, o, v)
    
    def _calculate_one_body_density(self) -> np.ndarray:
        basis = self.basis
        rho = np.zeros((basis.L, basis.L), dtype=basis.dtype)

        l1, t1 = self._l[1], self._t[1]
        l2, t2 = self._l[2], self._t[2]
        o, v = basis.o, basis.v

        rho = one_body_density_restricted(rho, t1, t2, l1, l2, o, v)

        return rho

    def _next_t_iteration(self, t: CoupledClusterParameter) -> CoupledClusterParameter:
        basis = self.basis

        rhs1, rhs2 = self.t_rhs(
            t1=t[1],
            t2=t[2],
            u=basis.u,
            f=self._f,
            v=basis.v,
            o=basis.o,
        )

        rhs = CoupledClusterParameter(t.orders, t.N, t.M, dtype=t.dtype)
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

        rhs = CoupledClusterParameter(l.orders, l.N, l.M, dtype=l.dtype)
        rhs.initialize_dicts({1: rhs1, 2: rhs2})

        return rhs
