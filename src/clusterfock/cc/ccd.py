import numpy as np
from clusterfock.basis import Basis
from clusterfock.cc.coupledcluster import CoupledCluster

from clusterfock.cc.rhs.t_CCD import amplitudes_ccd


class CCD(CoupledCluster):
    def __init__(self, basis: Basis):
        assert not basis.restricted, "CCD can not deal with restricted basis"

        t_amplitude_orders = ["D"]
        super().__init__(basis, t_amplitude_orders)

        o, v = self.basis.o, self.basis.v

        self.f_hh_o = self._f[o, o].copy()
        self.f_pp_o = self._f[v, v].copy()
        np.fill_diagonal(self.f_hh_o, 0)
        np.fill_diagonal(self.f_pp_o, 0)

    def _next_t_iteration(self, amplitudes: dict) -> dict:
        basis = self.basis

        t2_next = amplitudes_ccd(
            t2=amplitudes["D"],
            u=basis.u,
            f_hh_o=self.f_hh_o,
            f_pp_o=self.f_pp_o,
            v=basis.v,
            o=basis.o,
        )

        return {"D": t2_next * self._epsinv["D"]}

    def _evaluate_cc_energy(self, t_amplitudes: np.ndarray) -> float:
        t2 = t_amplitudes["D"]
        u, o, v = self.basis.u, self.basis.o, self.basis.v
        return 0.25 * np.einsum("ijab,abij", u[o, o, v, v], t2)
