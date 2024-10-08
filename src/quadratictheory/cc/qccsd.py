import numpy as np
from quadratictheory.basis import Basis
from quadratictheory.cc.quadcoupledcluster import QuadraticCoupledCluster
from quadratictheory.cc.parameter import CoupledClusterParameter
from quadratictheory.mix import DIISMixer

from quadratictheory.cc.rhs.t_QCCSD import amplitudes_qccsd
from quadratictheory.cc.rhs.l_QCCSD import lambda_amplitudes_qccsd
from quadratictheory.cc.rhs.t_inter_QCCSD import amplitudes_intermediates_qccsd
from quadratictheory.cc.rhs.l_inter_QCCSD import lambda_amplitudes_intermediates_qccsd
from quadratictheory.cc.energies.e_qccsd import energy_qccsd
from quadratictheory.cc.energies.e_inter_qccsd import energy_intermediates_qccsd
from quadratictheory.cc.weights.ccsd import (
    reference_ccsd,
    ket_singles_ccsd,
    bra_singles_ccsd,
    ket_doubles_ccsd,
    bra_doubles_ccsd,
)
from quadratictheory.cc.weights.qccsd import (
    reference_addition_qccsd,
    bra_singles_addition_qccsd,
    bra_doubles_addition_qccsd,
    triples_weigth_qccsd,
    quadruple_weigth_qccsd,
)
from quadratictheory.cc.densities.l_CCSD import one_body_density, two_body_density
from quadratictheory.cc.densities.l_QCCSD import (
    one_body_density_addition,
    two_body_density_addition,
)


class QCCSD(QuadraticCoupledCluster):
    def __init__(self, basis: Basis, intermediates: bool = True):
        assert not basis.restricted, "QCCSD can not deal with restricted basis"

        t_orders = [1, 2]
        l_orders = [1, 2]
        super().__init__(basis, t_orders, l_orders)

        self.mixer = DIISMixer(n_vectors=8)
        self.t_rhs = amplitudes_intermediates_qccsd if intermediates else amplitudes_qccsd
        self.l_rhs = (
            lambda_amplitudes_intermediates_qccsd if intermediates else lambda_amplitudes_qccsd
        )
        self.energy_expression = energy_intermediates_qccsd if intermediates else energy_qccsd

    def _next_t_iteration(
        self, t: CoupledClusterParameter, l: CoupledClusterParameter
    ) -> CoupledClusterParameter:
        basis = self.basis

        rhs1, rhs2 = self.t_rhs(
            t1=t[1],
            t2=t[2],
            l1=l[1],
            l2=l[2],
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

    def _t_rhs_timedependent(
        self, t: CoupledClusterParameter, l: CoupledClusterParameter
    ) -> CoupledClusterParameter:
        """
        Adds additional calculations to '_next_t_iteration' if this is required by the
        time evelution equations for the specific CC scheme. For standard coupled cluster, this
        adds nothing

        Args:
            t (CoupledClusterParameter): The amplitude at this iteration

        Returns:
            rhs_t (CoupledClusterParameter): The rhs of the time-dependent equation
        """
        rhs_t = self._next_t_iteration(t, l)
        mixing_term = np.einsum("bj,abij->ai", l[1], rhs_t[2])
        rhs_t.add(1, -mixing_term)

        return rhs_t

    def _l_rhs_timedependent(
        self, t: CoupledClusterParameter, l: CoupledClusterParameter
    ) -> CoupledClusterParameter:
        """
        Adds additional calculations to '_next_l_iteration' if this is required by the
        time evelution equations for the specific CC scheme. For standard coupled cluster, this
        adds nothing

        Args:
            t (CoupledClusterParameter): The amplitude at this iteration
            l (CoupledClusterParameter): The Lambda-amplitude at this iteration

        Returns:
            rhs_l (CoupledClusterParameter): The rhs of the time-dependent equation
        """

        rhs_l = self._next_l_iteration(t, l)

        mixing_term = np.einsum("ai,bj->abij", rhs_l[1], l[1])
        mixing_term = (
            mixing_term
            - mixing_term.transpose(0, 1, 3, 2)
            - mixing_term.transpose(1, 0, 2, 3)
            + mixing_term.transpose(1, 0, 3, 2)
        )
        rhs_l.add(2, -mixing_term)

        return rhs_l

    def _evaluate_cc_energy(self) -> float:
        t1, t2 = self._t[1], self._t[2]
        l1, l2 = self._l[1], self._l[2]

        u, o, v = self.basis.u, self.basis.o, self.basis.v
        f = self._f
        return self.energy_expression(t1, t2, l1, l2, u, f, o, v)

    def _calculate_one_body_density(self) -> np.ndarray:
        basis = self.basis
        rho = np.zeros((basis.L, basis.L), dtype=basis.dtype)

        t1, t2 = self._t[1], self._t[2]
        l1, l2 = self._l[1], self._l[2]
        o, v = basis.o, basis.v

        rho = one_body_density(rho, t1, t2, l1, l2, o, v)
        rho = one_body_density_addition(rho, t1, t2, l1, l2, o, v)

        return rho

    def _calculate_two_body_density(self) -> np.ndarray:
        basis = self.basis
        rho = np.zeros((basis.L, basis.L, basis.L, basis.L), dtype=basis.dtype)

        t1, t2 = self._t[1], self._t[2]
        l1, l2 = self._l[1], self._l[2]
        o, v = basis.o, basis.v

        rho = two_body_density(rho, t1, t2, l1, l2, o, v)
        rho = two_body_density_addition(rho, t1, t2, l1, l2, o, v)

        return rho

    def _overlap(self, t0, l0, t, l):
        return 0

    def _if_missing_use_stored(self, t1, t2, l1, l2):
        if t1 is None:
            t1 = self._t[1]
        if t2 is None:
            t2 = self._t[2]
        if l1 is None:
            l1 = self._l[1]
        if l2 is None:
            l2 = self._l[2]

        return t1, t2, l1, l2

    def reference_weights(self, t1=None, t2=None, l1=None, l2=None):
        t1, t2, l1, l2 = self._if_missing_use_stored(t1, t2, l1, l2)
        det = reference_ccsd(t1, t2, l1, l2)
        det += reference_addition_qccsd(t1, t2, l1, l2)

        return det

    def singles_weights(self, t1=None, t2=None, l1=None, l2=None):
        t1, t2, l1, l2 = self._if_missing_use_stored(t1, t2, l1, l2)
        ket = ket_singles_ccsd(t1, t2, l1, l2)
        bra = bra_singles_ccsd(t1, t2, l1, l2)
        bra += bra_singles_addition_qccsd(t1, t2, l1, l2)

        return np.multiply(bra, ket)

    def doubles_weights(self, t1=None, t2=None, l1=None, l2=None):
        t1, t2, l1, l2 = self._if_missing_use_stored(t1, t2, l1, l2)
        ket = ket_doubles_ccsd(t1, t2, l1, l2)
        bra = bra_doubles_ccsd(t1, t2, l1, l2)
        bra += bra_doubles_addition_qccsd(t1, t2, l1, l2)

        return np.multiply(bra, ket)

    def triples_weight(self, t1=None, t2=None, l1=None, l2=None):
        t1, t2, l1, l2 = self._if_missing_use_stored(t1, t2, l1, l2)

        return triples_weigth_qccsd(t1, t2, l1, l2)

    def quadruple_weight(self, t1=None, t2=None, l1=None, l2=None):
        t1, t2, l1, l2 = self._if_missing_use_stored(t1, t2, l1, l2)

        return quadruple_weigth_qccsd(t1, t2, l1, l2)
