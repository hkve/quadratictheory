from quadratictheory.basis.basis import Basis
from quadratictheory.hf.hartreefock import HartreeFock
import numpy as np


class GHF(HartreeFock):
    def __init__(self, basis: Basis) -> None:
        """
        Generalized Hartree-Fock (GHF). GHF is suitable for
        systems with both closed and open-shell configurations.

        Attributes:
            basis (Basis): The basis set used for calculations.

        See Also:
            HartreeFock: The base class for Hartree-Fock calculations.
            Basis: The basis set class defining the system.

        Examples:
            # Create a GHF instance with a basis set and compute the GHF energy.
            ghf = GHF(my_basis)
            ghf.run(tol=1e-8)
            print(f"GHF Energy: {ghf.energy()}")
        """
        assert not basis.restricted, "Basis can not be restricted"
        super().__init__(basis, False)

    def evaluate_energy_scheme(self) -> float:
        """
        Calculate the GHF energy, transforming the computational basis to HF basis.

        Returns:
            float: The GHF energy.
        """
        h, u = self.basis.h, self.basis.u
        rho = self.rho
        dE = self.basis._energy_shift

        E_OB = np.trace(rho @ h)
        E_TB = 0.5 * np.einsum("ag,bd,abgd", rho, rho, u)

        return E_OB + E_TB + dE

    def evaluate_fock_matrix(self, rho: np.ndarray) -> np.ndarray:
        """
        Calculate the GHF Fock matrix.

        Parameters:
            rho (np.ndarray): Single particle density matrix.

        Returns:
            np.ndarray: The GHF Fock matrix.
        """
        h, u = self.basis.h, self.basis.u
        return h + np.einsum("gd,agbd->ab", rho, u)
