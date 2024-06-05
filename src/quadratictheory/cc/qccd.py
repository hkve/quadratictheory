import numpy as np
from quadratictheory.basis import Basis
from quadratictheory.cc.quadcoupledcluster import QuadraticCoupledCluster
from quadratictheory.cc.parameter import CoupledClusterParameter
from quadratictheory.mix import DIISMixer

# General
from quadratictheory.cc.rhs.t_QCCD import amplitudes_qccd
from quadratictheory.cc.rhs.l_QCCD import lambda_amplitudes_qccd
from quadratictheory.cc.rhs.t_inter_QCCD import amplitudes_intermediates_qccd
from quadratictheory.cc.rhs.l_inter_QCCD import lambda_amplitudes_intermediates_qccd
from quadratictheory.cc.energies.e_qccd import energy_qccd
from quadratictheory.cc.energies.e_inter_qccd import energy_intermediates_qccd
from quadratictheory.cc.densities.l_CCD import one_body_density, two_body_density
from quadratictheory.cc.densities.l_QCCD import two_body_density_addition
from quadratictheory.cc.weights.ccd import reference_ccd, ket_doubles_ccd, bra_doubles_ccd
from quadratictheory.cc.weights.qccd import (
    reference_addition_qccd,
    bra_doubles_addition_qccd,
    quadruple_weigth_qccd,
)


# Restricted
from quadratictheory.cc.rhs.t_RQCCD import t_qccd_restricted
from quadratictheory.cc.rhs.l_RQCCD import l_qccd_restricted
from quadratictheory.cc.energies.e_rqccd import energy_qccd_restricted

from quadratictheory.cc.rhs.t_inter_RQCCD import t_intermediates_qccd_restricted
from quadratictheory.cc.rhs.l_inter_RQCCD import l_intermediates_qccd_restricted
from quadratictheory.cc.energies.e_inter_rqccd import energy_intermediates_qccd_restricted


class QCCD(QuadraticCoupledCluster):
    def __init__(self, basis: Basis, intermediates: bool = True):
        assert not basis.restricted, "QCCD can not deal with restricted basis"

        t_orders = [2]
        l_orders = [2]
        super().__init__(basis, t_orders, l_orders)

        self.mixer = DIISMixer(n_vectors=8)
        self.t_rhs = amplitudes_intermediates_qccd if intermediates else amplitudes_qccd
        self.l_rhs = (
            lambda_amplitudes_intermediates_qccd if intermediates else lambda_amplitudes_qccd
        )
        self.energy_expression = energy_intermediates_qccd if intermediates else energy_qccd

    def _next_t_iteration(
        self, t: CoupledClusterParameter, l: CoupledClusterParameter
    ) -> CoupledClusterParameter:
        basis = self.basis

        rhs2 = self.t_rhs(
            t2=t[2],
            l2=l[2],
            u=basis.u,
            f=self._f,
            v=basis.v,
            o=basis.o,
        )

        rhs = CoupledClusterParameter(t.orders, t.N, t.M, dtype=t.dtype)
        rhs.initialize_dicts({2: rhs2})

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
        return self._next_t_iteration(t, l)

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
        l2 = self._l[2]

        u, o, v = self.basis.u, self.basis.o, self.basis.v
        f = self._f

        return self.energy_expression(t2, l2, u, f, o, v)

    def _calculate_one_body_density(self) -> np.ndarray:
        basis = self.basis
        rho = np.zeros((basis.L, basis.L), dtype=basis.dtype)

        l2, t2 = self._l[2], self._t[2]
        o, v = basis.o, basis.v

        rho = one_body_density(rho, t2, l2, o, v)

        return rho

    def _calculate_two_body_density(self) -> np.ndarray:
        basis = self.basis
        rho = np.zeros((basis.L, basis.L, basis.L, basis.L), dtype=basis.dtype)

        l2, t2 = self._l[2], self._t[2]
        o, v = basis.o, basis.v

        rho = two_body_density(rho, t2, l2, o, v)
        rho = two_body_density_addition(rho, t2, l2, o, v)

        return rho

    def _overlap(self, t0, l0, t, l):
        return 0

    def _if_missing_use_stored(self, t2, l2):
        if t2 is None:
            t2 = self._t[2]
        if l2 is None:
            l2 = self._l[2]

        return t2, l2

    def reference_weights(self, t2=None, l2=None):
        t2, l2 = self._if_missing_use_stored(t2, l2)
        det = reference_ccd(t2, l2)
        det += reference_addition_qccd(t2, l2)

        return det

    def doubles_weights(self, t2=None, l2=None):
        t2, l2 = self._if_missing_use_stored(t2, l2)
        ket = ket_doubles_ccd(t2, l2)
        bra = bra_doubles_ccd(t2, l2)
        bra += bra_doubles_addition_qccd(t2, l2)

        return np.multiply(bra, ket)

    def quadruple_weight(self, t2=None, l2=None):
        t2, l2 = self._if_missing_use_stored(t2, l2)

        return quadruple_weigth_qccd(t2, l2)


class RQCCD(QuadraticCoupledCluster):
    def __init__(self, basis: Basis, intermediates: bool = True):
        assert basis.restricted, "RQCCD must deal with restricted basis"

        t_orders = [2]
        l_orders = [2]
        super().__init__(basis, t_orders, l_orders)

        self.mixer = DIISMixer(n_vectors=8)
        self.t_rhs = t_intermediates_qccd_restricted if intermediates else t_qccd_restricted
        self.l_rhs = l_intermediates_qccd_restricted if intermediates else t_qccd_restricted
        self.energy_expression = (
            energy_intermediates_qccd_restricted if intermediates else energy_qccd_restricted
        )

    def _next_t_iteration(
        self, t: CoupledClusterParameter, l: CoupledClusterParameter
    ) -> CoupledClusterParameter:
        basis = self.basis

        rhs2 = self.t_rhs(
            t2=t[2],
            l2=l[2],
            u=basis.u,
            f=self._f,
            v=basis.v,
            o=basis.o,
        )

        rhs = CoupledClusterParameter(t.orders, t.N, t.M, dtype=t.dtype)
        rhs.initialize_dicts({2: rhs2})

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
        return self._next_t_iteration(t, l)

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
        l2 = self._l[2]

        u, o, v = self.basis.u, self.basis.o, self.basis.v
        f = self._f

        return self.energy_expression(t2, l2, u, f, o, v)

    def _calculate_one_body_density(self) -> np.ndarray:
        pass

    def _calculate_two_body_density(self) -> np.ndarray:
        pass

    def _overlap(self, t0, l0, t, l):
        return 0
