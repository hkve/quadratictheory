import numpy as np
from clusterfock.basis import Basis
from clusterfock.cc.quadcoupledcluster import QuadraticCoupledCluster
from clusterfock.cc.parameter import CoupledClusterParameter
from clusterfock.mix import DIISMixer

from clusterfock.cc.rhs.t_QCCD import amplitudes_qccd
from clusterfock.cc.rhs.l_QCCD import lambda_amplitudes_qccd
from clusterfock.cc.rhs.t_inter_QCCD import amplitudes_intermediates_qccd
from clusterfock.cc.rhs.l_inter_QCCD import lambda_amplitudes_intermediates_qccd
from clusterfock.cc.energies.e_qccd import energy_qccd
from clusterfock.cc.energies.e_inter_qccd import energy_intermediates_qccd


class QCCD(QuadraticCoupledCluster):
    def __init__(self, basis: Basis, intermediates: bool = False):
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

    def _evaluate_cc_energy(self, t: CoupledClusterParameter) -> float:
        t2 = t[2]
        l2 = self._l[2]

        u, f, o, v = self.basis.u, self.basis.f, self.basis.o, self.basis.v
        return self.energy_expression(t2, l2, u, f, o, v)
