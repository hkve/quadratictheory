from mypack.basis.basis import Basis
from mypack.hf.hartreefock import HartreeFock
import numpy as np


class GHF(HartreeFock):
    def __init__(self, basis: Basis) -> None:
        assert not basis.restricted, "Basis can not be restricted"
        super().__init__(basis)

    def evaluate_energy_scheme(self) -> float:
        h, u = self.basis.h, self.basis.u
        rho = self.rho
        dE = self.basis._energy_shift

        E_OB = np.trace(rho @ h)
        E_TB = 0.5 * np.einsum("ag,bd,abgd", rho, rho, u)

        return E_OB + E_TB + dE

    def evaluate_fock_matrix(self, rho: np.ndarray) -> np.ndarray:
        h, u = self.basis.h, self.basis.u
        return h + np.einsum("gd,agbd->ab", rho, u)
