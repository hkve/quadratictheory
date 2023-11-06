import numpy as np
from clusterfock.basis import Basis
from clusterfock.cc.coupledcluster import CoupledCluster
from clusterfock.cc.parameter import CoupledClusterParameter

from clusterfock.cc.rhs.t_uCCD2 import amplitudes_uccd2
from clusterfock.cc.energies.e_uccd2 import energy_uccd2


class UCCD2(CoupledCluster):
    def __init__(self, basis: Basis, intermediates: bool = True):
        assert not basis.restricted, "CCD can not deal with restricted basis"

        t_orders = [2]
        super().__init__(basis, t_orders)

        # self.t_rhs = amplitudes_intermediates_ccd if intermediates else amplitudes_ccd
        # self.l_rhs = lambda_amplitudes_intermediates_ccd if intermediates else lambda_amplitudes_ccd
        self.t_rhs = amplitudes_uccd2

    def _next_t_iteration(self, t: CoupledClusterParameter) -> CoupledClusterParameter:
        basis = self.basis

        rhs2 = self.t_rhs(
            t2=t[2],
            u=basis.u,
            f=self._f,
            v=basis.v,
            o=basis.o,
        )

        rhs = CoupledClusterParameter(t.orders, t.N, t.M)
        rhs.initialize_dicts({2: rhs2})

        return rhs

    def _evaluate_cc_energy(self, t: CoupledClusterParameter) -> float:
        t2 = t[2]
        u, f, o, v = self.basis.u, self.basis.f, self.basis.o, self.basis.v
        return energy_uccd2(t2, u, f, o, v)
