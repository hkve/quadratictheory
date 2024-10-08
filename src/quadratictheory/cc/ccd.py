import numpy as np
from quadratictheory.basis import Basis
from quadratictheory.cc.coupledcluster import CoupledCluster
from quadratictheory.cc.parameter import CoupledClusterParameter

from quadratictheory.cc.rhs.t_CCD import amplitudes_ccd
from quadratictheory.cc.rhs.l_CCD import lambda_amplitudes_ccd
from quadratictheory.cc.rhs.t_inter_CCD import amplitudes_intermediates_ccd
from quadratictheory.cc.rhs.l_inter_CCD import lambda_amplitudes_intermediates_ccd
from quadratictheory.cc.rhs.t_RCCD import amplitudes_ccd_restricted

from quadratictheory.cc.densities.l_CCD import one_body_density, two_body_density
from quadratictheory.cc.energies.e_inter_ccd import td_energy_addition

from quadratictheory.cc.weights.ccd import reference_ccd, ket_doubles_ccd, bra_doubles_ccd

from quadratictheory.cc.rhs.t_inter_RCCD import amplitudes_intermediates_rccd
from quadratictheory.cc.rhs.l_inter_RCCD import lambda_amplitudes_intermediates_rccd
from quadratictheory.cc.rhs.l_RCCD import lambda_amplitudes_rccd

from quadratictheory.cc.densities.l_RCCD import one_body_density_restricted
from quadratictheory.cc.energies.e_inter_rccd import td_energy_addition_opti_restricted
from quadratictheory.cc.energies.e_rccd import td_energy_addition_restricted


class GCCD(CoupledCluster):
    def __init__(self, basis: Basis, intermediates: bool = True):
        assert not basis.restricted, "CCD can not deal with restricted basis"

        t_orders = [2]
        l_orders = [2]
        super().__init__(basis, t_orders, l_orders)

        self.t_rhs = amplitudes_intermediates_ccd if intermediates else amplitudes_ccd
        self.l_rhs = lambda_amplitudes_intermediates_ccd if intermediates else lambda_amplitudes_ccd
        self.td_energy_addition = td_energy_addition

    def _next_t_iteration(self, t: CoupledClusterParameter) -> CoupledClusterParameter:
        basis = self.basis

        rhs2 = self.t_rhs(
            t2=t[2],
            u=basis.u,
            f=self._f,
            v=basis.v,
            o=basis.o,
        )

        rhs = CoupledClusterParameter(t.orders, t.N, t.M, dtype=t.dtype)
        rhs.initialize_dicts({2: rhs2})

        return rhs

    def _next_l_iteration(
        self, t: CoupledClusterParameter, l: CoupledClusterParameter
    ) -> CoupledClusterParameter:
        basis = self.basis

        rhs2 = self.l_rhs(
            t2=t[2],
            l2=l[2],
            u=basis.u,
            f=self._f,
            v=basis.v,
            o=basis.o,
        )

        rhs = CoupledClusterParameter(l.orders, l.N, l.M, dtype=l.dtype)
        rhs.initialize_dicts({2: rhs2})

        return rhs

    def _evaluate_cc_energy(self) -> float:
        t2 = self._t[2]
        u, o, v = self.basis.u, self.basis.o, self.basis.v
        return 0.25 * np.einsum("ijab,abij", u[o, o, v, v], t2)

    def _evaluate_tdcc_energy(self) -> float:
        t2, l2 = self._t[2], self._l[2]
        u, o, v = self.basis.u, self.basis.o, self.basis.v
        f = self._f
        return self.td_energy_addition(t2, l2, u, f, o, v)

    def _calculate_one_body_density(self) -> np.ndarray:
        basis = self.basis
        rho = np.zeros((basis.L, basis.L), dtype=basis.dtype)

        l, t = self._l[2], self._t[2]
        o, v = basis.o, basis.v

        rho = one_body_density(rho, t, l, o, v)

        return rho

    def _calculate_two_body_density(self) -> np.ndarray:
        basis = self.basis
        rho = np.zeros((basis.L, basis.L, basis.L, basis.L), dtype=basis.dtype)

        l, t = self._l[2], self._t[2]
        o, v = basis.o, basis.v

        rho = two_body_density(rho, t, l, o, v)

        return rho

    def _overlap(self, t0, l0, t, l):
        psitilde_t = 1
        psitilde_t += 0.25 * np.einsum("abij,abij->", t0[2], l[2])
        psitilde_t -= 0.25 * np.einsum("abij,abij->", t[2], l[2])

        psit = 1
        psit += 0.25 * np.einsum("abij,abij->", t[2], l0[2])
        psit -= 0.25 * np.einsum("abij,abij->", t0[2], l0[2])

        return psit * psitilde_t

    def _if_missing_use_stored(self, t2, l2):
        if t2 is None:
            t2 = self._t[2]
        if l2 is None:
            l2 = self._l[2]

        return t2, l2

    def reference_weights(self, t2=None, l2=None):
        t2, l2 = self._if_missing_use_stored(t2, l2)
        det = reference_ccd(t2, l2)

        return det

    def doubles_weights(self, t2=None, l2=None):
        t2, l2 = self._if_missing_use_stored(t2, l2)
        ket = ket_doubles_ccd(t2, l2)
        bra = bra_doubles_ccd(t2, l2)

        return np.multiply(bra, ket)


class RCCD(CoupledCluster):
    def __init__(self, basis: Basis, intermediates=True):
        assert basis.restricted, f"Restricted CCD requires restricted basis"

        t_orders = [2]
        l_orders = [2]
        super().__init__(basis, t_orders, l_orders)

        o, v = basis.o, basis.v
        self.f_pp_o = self._f[v, v].copy()
        self.f_hh_o = self._f[o, o].copy()

        self.t_rhs = amplitudes_intermediates_rccd if intermediates else amplitudes_ccd_restricted
        self.l_rhs = (
            lambda_amplitudes_intermediates_rccd if intermediates else lambda_amplitudes_rccd
        )
        self.td_energy_addition = (
            td_energy_addition_opti_restricted if intermediates else td_energy_addition_restricted
        )

    def _evaluate_cc_energy(self) -> float:
        t2 = self._t[2]
        u, o, v = self.basis.u, self.basis.o, self.basis.v
        D = np.einsum("ijab,abij", u[o, o, v, v], t2, optimize=True)
        E = np.einsum("ijba,abij", u[o, o, v, v], t2, optimize=True)

        return 2 * D - E

    def _evaluate_tdcc_energy(self) -> float:
        t2, l2 = self._t[2], self._l[2]
        u, o, v = self.basis.u, self.basis.o, self.basis.v
        f = self._f
        return self.td_energy_addition(t2, l2, u, f, o, v)

    def _calculate_one_body_density(self) -> np.ndarray:
        basis = self.basis
        rho = np.zeros((basis.L, basis.L), dtype=basis.dtype)

        l, t = self._l[2], self._t[2]
        o, v = basis.o, basis.v

        rho = one_body_density_restricted(rho, t, l, o, v)

        return rho

    def _next_t_iteration(self, t: CoupledClusterParameter):
        basis = self.basis

        rhs2 = self.t_rhs(
            t2=t[2],
            u=basis.u,
            f=self._f,
            v=basis.v,
            o=basis.o,
        )

        rhs = CoupledClusterParameter(t.orders, t.N, t.M, dtype=t.dtype)
        rhs.initialize_dicts({2: rhs2})

        return rhs

    def _next_l_iteration(self, t: CoupledClusterParameter, l: CoupledClusterParameter):
        basis = self.basis

        rhs2 = self.l_rhs(
            t2=t[2],
            l2=l[2],
            u=basis.u,
            f=self._f,
            v=basis.v,
            o=basis.o,
        )

        rhs = CoupledClusterParameter(l.orders, t.N, t.M, dtype=t.dtype)
        rhs.initialize_dicts({2: rhs2})

        return rhs
