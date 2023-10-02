import numpy as np
from clusterfock.basis import Basis
from clusterfock.cc.coupledcluster import CoupledCluster
from clusterfock.cc.parameter import CoupledClusterParameter

from clusterfock.cc.rhs.t_CCD import amplitudes_ccd
from clusterfock.cc.rhs.t_inter_CCD import amplitudes_intermediates_ccd
from clusterfock.cc.rhs.t_RCCD import amplitudes_ccd_restricted


class GCCD(CoupledCluster):
    def __init__(self, basis: Basis, intermediates: bool = True):
        assert not basis.restricted, "CCD can not deal with restricted basis"

        orders = [2]
        super().__init__(basis, orders)

        o, v = self.basis.o, self.basis.v

        self.rhs = amplitudes_intermediates_ccd if intermediates else amplitudes_ccd

    def _next_t_iteration(self, t: CoupledClusterParameter) -> CoupledClusterParameter:
        basis = self.basis

        rhs2 = self.rhs(
            t2=t[2],
            u=basis.u,
            f=self._f,
            v=basis.v,
            o=basis.o,
        )

        rhs = CoupledClusterParameter(t.orders, t.N, t.M)
        rhs.initialize_dicts({
            2: rhs2
        })

        return rhs

    def _evaluate_cc_energy(self, t: CoupledClusterParameter) -> float:
        t2 = t[2]
        u, o, v = self.basis.u, self.basis.o, self.basis.v
        return 0.25 * np.einsum("ijab,abij", u[o, o, v, v], t2)


class RCCD(CoupledCluster):
    def __init__(self, basis: Basis):
        assert basis.restricted, f"Restricted CCD requires restricted basis"

        orders = [2]
        super().__init__(basis, orders)

        o, v = basis.o, basis.v
        self.f_pp_o = self._f[v, v].copy()
        self.f_hh_o = self._f[o, o].copy()

        self.rhs = amplitudes_ccd_restricted

    def _evaluate_cc_energy(self, t: CoupledClusterParameter) -> float:
        t2 = t[2]
        u, o, v = self.basis.u, self.basis.o, self.basis.v
        D = np.einsum("ijab,abij", u[o, o, v, v], t2, optimize=True)
        E = np.einsum("ijba,abij", u[o, o, v, v], t2, optimize=True)

        return 2 * D - E

    def _next_t_iteration(self, t: CoupledClusterParameter):
        basis = self.basis
        t2 = t[2]

        rhs2 = self.rhs(
            t2=t[2],
            u=basis.u,
            f=self._f,
            v=basis.v,
            o=basis.o,
        )

        rhs = CoupledClusterParameter(t.orders, t.N, t.M)
        rhs.initialize_dicts({
            2: rhs2
        })

        return rhs
