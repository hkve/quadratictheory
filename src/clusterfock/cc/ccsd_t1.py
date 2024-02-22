import numpy as np
from clusterfock.basis import Basis
from clusterfock.cc.coupledcluster_t1 import CoupledCluster_T1
from clusterfock.cc.parameter import CoupledClusterParameter

from clusterfock.cc.rhs.t_CCSD import amplitudes_ccsd
from clusterfock.cc.rhs.l_CCSD import lambda_amplitudes_ccsd
from clusterfock.cc.rhs.t_inter_CCD import amplitudes_intermediates_ccd
from clusterfock.cc.rhs.t_inter_CCSD import amplitudes_intermediates_ccsd
from clusterfock.cc.rhs.l_inter_CCSD import lambda_amplitudes_intermediates_ccsd

from clusterfock.cc.densities.l_CCSD import one_body_density, two_body_density
from clusterfock.cc.energies.e_inter_ccsd import td_energy_addition


class GCCSD_T1(CoupledCluster_T1):
    def __init__(self, basis: Basis, intermediates: bool = True):
        assert not basis.restricted, "CCD can not deal with restricted basis"

        t_orders = [1,2]
        # l_orders = [1, 2]
        super().__init__(basis, t_orders)
        self._t1 = np.zeros((basis.M, basis.N), dtype=basis.dtype)
        self._u = basis.u.copy()
        self._f = basis.f.copy()

        self.t1_rhs = amplitudes_intermediates_ccsd
        self.t2_rhs = amplitudes_intermediates_ccd if intermediates else amplitudes_ccsd
        self.l_rhs = (
            lambda_amplitudes_intermediates_ccsd if intermediates else lambda_amplitudes_ccsd
        )

        self.td_energy_addition = td_energy_addition

    def _next_t_iteration(self, t: CoupledClusterParameter) -> dict:
        basis = self.basis

        self._f, self._u = self.perform_t1_transform(t[1], basis.h, basis.u)

        rhs1, _ = self.t1_rhs(
            t1=np.zeros((basis.M, basis.N)),
            t2=t[2],
            u=self._u,
            f=self._f,
            v=basis.v,
            o=basis.o,
        )

        rhs2 = self.t2_rhs(
            t2=t[2],
            u=self._u,
            f=self._f,
            v=basis.v,
            o=basis.o,
        )
        
        rhs = CoupledClusterParameter(t.orders, t.N, t.M, dtype=t.dtype)
        rhs.initialize_dicts({1: rhs1, 2: rhs2})

        return rhs

    def _evaluate_cc_energy(self) -> float:
        o, v = self.basis.o, self.basis.v
        t2 = self._t[2]

        e = 0
        e += np.einsum("abij,ijab->", t2, self._u[o, o, v, v], optimize=True) / 4

        return e