import numpy as np
from clusterfock.basis import Basis
from clusterfock.cc.coupledcluster import CoupledCluster

from clusterfock.cc.rhs.t_CCD import amplitudes_ccd
from clusterfock.cc.rhs.t_inter_CCD import amplitudes_intermediates_ccd
from clusterfock.cc.rhs.t_RCCD import amplitudes_ccd_restricted


class GCCD(CoupledCluster):
    def __init__(self, basis: Basis, intermediates: bool = True):
        assert not basis.restricted, "CCD can not deal with restricted basis"

        t_amplitude_orders = ["D"]
        super().__init__(basis, t_amplitude_orders)

        o, v = self.basis.o, self.basis.v

        self.f_hh_o = self._f[o, o].copy()
        self.f_pp_o = self._f[v, v].copy()

        self.rhs = amplitudes_intermediates_ccd if intermediates else amplitudes_ccd

    def _next_t_iteration(self, amplitudes: dict) -> dict:
        basis = self.basis

        print(f"T2 in {np.linalg.norm(amplitudes['D'])}")
        t2_next = self.rhs(
            t2=amplitudes["D"],
            u=basis.u,
            f_hh_o=self.f_hh_o,
            f_pp_o=self.f_pp_o,
            v=basis.v,
            o=basis.o,
        )
        print(f"T2 out {np.linalg.norm(t2_next)}")
        return {"D": t2_next * self._epsinv["D"]}

    def _evaluate_cc_energy(self, t_amplitudes: np.ndarray) -> float:
        t2 = t_amplitudes["D"]
        u, o, v = self.basis.u, self.basis.o, self.basis.v
        return 0.25 * np.einsum("ijab,abij", u[o, o, v, v], t2)


class RCCD(CoupledCluster):
    def __init__(self, basis: Basis):
        assert basis.restricted, f"Restricted CCD requires restricted basis"

        t_amplitude_orders = ["D"]
        super().__init__(basis, t_amplitude_orders)

        o, v = basis.o, basis.v
        self.f_pp_o = self._f[v, v].copy()
        self.f_hh_o = self._f[o, o].copy()
        np.fill_diagonal(self.f_pp_o, 0)
        np.fill_diagonal(self.f_hh_o, 0)

        self.rhs = amplitudes_ccd_restricted

    def _evaluate_cc_energy(self, t_amplitudes: np.ndarray) -> float:
        t2 = t_amplitudes["D"]
        u, o, v = self.basis.u, self.basis.o, self.basis.v
        D = np.einsum("ijab,abij", u[o, o, v, v], t2, optimize=True)
        E = np.einsum("ijba,abij", u[o, o, v, v], t2, optimize=True)

        return 2 * D - E

    def _next_t_iteration(self, amplitudes: dict):
        basis = self.basis
        t2 = amplitudes["D"]

        t2_next = self.rhs(
            t2=amplitudes["D"],
            u=basis.u,
            f_hh_o=self.f_hh_o,
            f_pp_o=self.f_pp_o,
            v=basis.v,
            o=basis.o,
        )

        return {"D": t2_next * self._epsinv["D"]}
