import numpy as np
from clusterfock.basis import Basis
from clusterfock.cc.quadcoupledcluster import QuadraticCoupledCluster
from clusterfock.cc.parameter import CoupledClusterParameter
from clusterfock.mix import DIISMixer

from clusterfock.cc.rhs.t_QCCSD import amplitudes_qccsd
from clusterfock.cc.rhs.l_QCCSD import lambda_amplitudes_qccsd
from clusterfock.cc.rhs.t_inter_QCCSD import amplitudes_intermediates_qccsd
from clusterfock.cc.rhs.l_inter_QCCSD import lambda_amplitudes_intermediates_qccsd
from clusterfock.cc.energies.e_qccsd import energy_qccsd
from clusterfock.cc.energies.e_inter_qccsd import energy_intermediates_qccsd

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

    def _evaluate_cc_energy(self, t: CoupledClusterParameter) -> float:
        t1, t2 = t[1], t[2]
        l1, l2 = self._l[1], self._l[2]

        u, f, o, v = self.basis.u, self.basis.f, self.basis.o, self.basis.v
        return self.energy_expression(t1, t2, l1, l2, u, f, o, v)
