import numpy as np
from clusterfock.basis import Basis
from clusterfock.cc.coupledcluster import CoupledCluster
from clusterfock.cc.parameter import CoupledClusterParameter

from clusterfock.cc.rhs.t_CCSD import amplitudes_ccsd
from clusterfock.cc.rhs.t_inter_CCSD import amplitudes_intermediates_ccsd


class GCCSD(CoupledCluster):
    def __init__(self, basis: Basis, intermediates: bool = True):
        assert not basis.restricted, "CCD can not deal with restricted basis"

        orders = [1, 2]
        super().__init__(basis, orders)

        self.rhs = amplitudes_intermediates_ccsd if intermediates else amplitudes_ccsd

    def _next_t_iteration(self, t: CoupledClusterParameter) -> dict:
        basis = self.basis

        rhs1, rhs2 = self.rhs(
            t1=t[1],
            t2=t[2],
            u=basis.u,
            f=self._f,
            v=basis.v,
            o=basis.o,
        )

        rhs = CoupledClusterParameter(t.orders, t.N, t.M)
        rhs.initialize_dicts({
            1: rhs1,
            2: rhs2
        })

        return rhs

    def _evaluate_cc_energy(self, t: CoupledClusterParameter) -> float:
        u, o, v = self.basis.u, self.basis.o, self.basis.v
        t1, t2 = t[1], t[2]

        e = 0
        e += np.einsum("ia,ai->", self._f[o, v], t1, optimize=True)
        e += np.einsum("abij,ijab->", t2, u[o, o, v, v], optimize=True) / 4
        e += np.einsum("ai,bj,ijab->", t1, t1, u[o, o, v, v], optimize=True) / 2

        return 0.25 * np.einsum("ijab,abij", u[o, o, v, v], t2)
