import numpy as np
from clusterfock.basis import Basis
from clusterfock.cc.coupledcluster import CoupledCluster

from clusterfock.cc.rhs.t_CCSD import amplitudes_ccsd
from clusterfock.cc.rhs.t_inter_CCSD import amplitudes_intermediates_ccsd


class GCCSD(CoupledCluster):
    def __init__(self, basis: Basis, intermediates: bool = True):
        assert not basis.restricted, "CCD can not deal with restricted basis"

        t_amplitude_orders = ["S", "D"]
        super().__init__(basis, t_amplitude_orders)
        np.fill_diagonal(self._f, 0)

        self.rhs = amplitudes_intermediates_ccsd if intermediates else amplitudes_ccsd

    def _next_t_iteration(self, amplitudes: dict) -> dict:
        basis = self.basis

        t1_next, t2_next = self.rhs(
            t1=amplitudes["S"],
            t2=amplitudes["D"],
            u=basis.u,
            f=self._f,
            v=basis.v,
            o=basis.o,
        )

        return {"S": t1_next * self._epsinv["S"], "D": t2_next * self._epsinv["D"]}

    def _evaluate_cc_energy(self, t_amplitudes: np.ndarray) -> float:
        u, o, v = self.basis.u, self.basis.o, self.basis.v
        t1, t2 = t_amplitudes["S"], t_amplitudes["D"]

        e = 0
        e += np.einsum("ia,ai->", self._f[o, v], t1, optimize=True)
        e += np.einsum("abij,ijab->", t2, u[o, o, v, v], optimize=True) / 4
        e += np.einsum("ai,bj,ijab->", t1, t1, u[o, o, v, v], optimize=True) / 2

        return 0.25 * np.einsum("ijab,abij", u[o, o, v, v], t2)
