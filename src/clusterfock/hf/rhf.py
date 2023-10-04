from clusterfock.basis import Basis
from clusterfock.hf.hartreefock import HartreeFock
import numpy as np


class RHF(HartreeFock):
    def __init__(self, basis: Basis) -> None:
        """
        Restricted Hartree-Fock (GHF) class for electronic structure calculations.
        Should only be used for closed-shell systems

        Attributes:
            basis (Basis): The basis set used for calculations.

        See Also:
            HartreeFock: The base class for Hartree-Fock calculations.
            Basis: The basis set class defining the system.

        Examples:
            # Create a RHF instance with a basis set and compute the GHF energy.
            rhf = RHF(my_basis)
            rhf.run(tol=1e-8)
            print(f"RHF Energy: {rhf.energy()}")
        """
        assert basis.restricted, "Basis must be restricted"
        super().__init__(basis)

    def evaluate_energy_scheme(self) -> float:
        """
        Calculate the RHF energy, transforming the computational basis to HF basis.
    
        Returns:
            float: The RHF energy.
        """        
        h, u = self.basis.h, self.basis.u
        rho = self.rho
        dE = self.basis._energy_shift

        E_OB = np.trace(rho @ h)
        E_TB = 0.5 * np.einsum("ag,bd,abgd", rho, rho, u) - 0.25 * np.einsum(
            "ag,bd,abdg", rho, rho, u
        )

        return E_OB + E_TB + dE

    def evaluate_fock_matrix(self, rho: np.ndarray) -> float:
        """
        Calculate the RHF Fock matrix.

        Parameters:
            rho (np.ndarray): Single particle density matrix. Note that in
            contrast to GHF, the identity would have elements of 2 instead of
            1, showing the states are doubly degenerat

        Returns:
            np.ndarray: The RHF Fock matrix.
        """
        h, u = self.basis.h, self.basis.u
        return h + np.einsum("gd,agbd->ab", rho, u) - 0.5 * np.einsum("gd,agdb->ab", rho, u)
