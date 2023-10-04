from __future__ import annotations
from clusterfock.basis.basis import Basis
from clusterfock.mix import Mixer, DIISMixer
from abc import ABC, abstractmethod
import numpy as np
from scipy.linalg import eigh


class HartreeFock(ABC):
    def __init__(self, basis: Basis) -> None:
        """
        Initialize a Hartree-Fock (HF) electronic structure calculation.

        Parameters:
        - basis (Basis): The basis set and molecular geometry for the HF calculation.

        Attributes:
        - basis (Basis): The basis set and molecular geometry for the calculation.
        - has_run (bool): Indicates whether the HF calculation has been executed.
        - converged (bool): Indicates whether the HF calculation has converged.
        - mixer (DIISMixer): The mixer used for DIIS (Direct Inversion in the Iterative Subspace).
        - guess (str): The initial guess for the electronic wave function coefficients.
        - _available_coefficient_guesses (dict): A dictionary of available initial coefficient guesses.

        Notes:
        - The `basis` parameter should be an instance of the `Basis` class, containing molecular geometry
          and basis set information.
        - By default, the HF calculation has not yet been executed (`has_run` is False).
        - The `mixer` is set to use DIIS with 8 vectors for coefficient mixing.
        - The `guess` attribute determines the initial guess for electronic wave function coefficients.
        - Available initial guess options can be found in `_available_coefficient_guesses`.
        """
        self.basis = basis
        self.has_run = False
        self.converged = False
        self.mixer = DIISMixer(n_vectors=8)
        self.guess = "I"

        self._available_coefficient_guesses = {
            "I": _identity_guess,
            "core": _core_guess,
        }

    def density_matrix(self, C: np.ndarray) -> np.ndarray:
        """
        Calculates the density matrix based on the sp transformation matrix

        Parameters:
        - C (ndarray): (L,L) sp transformation matrix

        Returns:
        - rho (ndarray): (L,L) density matrix
        """
        d = self.basis._degeneracy
        o, v = self.basis.o, self.basis.v
        return d * np.einsum("ai,bi->ab", C[:, o].conj(), C[:, o])

    @abstractmethod
    def evaluate_energy_scheme(self) -> float:
        # A scheme (GHF, RHF) should implement this for energy evaluation.
        pass

    @abstractmethod
    def evaluate_fock_matrix(self, rho: np.ndarray, h: np.ndarray, u: np.ndarray) -> np.ndarray:
        # A scheme (GHF, RHF) should implement this for evaluation of the fock matrix with h and u in the computational basis
        pass

    def run(self, tol: float = 1e-8, maxiters: int = 1000, vocal: bool = False) -> HartreeFock:
        """
        Perform a Hartree-Fock (HF) electronic structure calculation.

        Parameters:
        - tol (float, optional): Convergence tolerance for the change in the HF eigenvalues. Defaults to 1e-8.
        - maxiters (int, optional): Maximum number of iterations allowed. Defaults to 1000.
        - vocal (bool, optional): If True, print iteration details. Defaults to False.

        Returns:
        - HartreeFock: An updated HartreeFock instance.

        Notes:
        - Convergence is determined by monitoring the change in HF eigenvalues.
        - The HF calculation initializes from the specified guess for electronic wave function coefficients.
        - The iteration continues until convergence or reaching the maximum number of iterations.
        - If `vocal` is True, iteration information is printed.

        """
        if self.has_run:
            self.has_run = False
            self.converged = False

        basis = self.basis

        L, N = basis.L, basis.N

        C = self._available_coefficient_guesses[self.guess](basis)
        new_rho = self.density_matrix(C)
        old_rho = new_rho.copy()
        hf_sp_energies = np.diag(basis.h).copy()

        iters = 0
        diff = 1

        old_fock = np.zeros_like(C)
        while (iters < maxiters) and (diff > tol):
            new_fock = self.evaluate_fock_matrix(new_rho)

            new_fock = self.mixer(old_fock, new_fock-old_fock)
            old_fock = new_fock.copy()

            hf_sp_energies, C = eigh(new_fock, basis.s)
            
            old_rho = new_rho.copy()
            new_rho = self.density_matrix(C)

            diff = np.linalg.norm(np.abs(new_rho - old_rho))

            iters += 1

            if vocal:
                print(f"i = {iters}, mo = {hf_sp_energies}")

        self.has_run = True
        if iters < maxiters:
            self.converged = True

        self._diff = diff
        self.iters = iters
        self.rho = new_rho
        self.C = C

        return self

    def energy(self) -> float:
        """
        Calculates the hatree fock optimized energy.

        Returns:
        - energy (float): Energy of many body state

        Notes:
        - If the transformation matrix (from the computational basis) is equal to what is gotten from
          HF calculation, the Basis objects energy is evaluated. This is the case since basis has called
          .basis_change to a HF basis, and thus should not be applied again. If not, the HF energy is calculated
          using the stored coefficent matrix (which transforms from the computational basis)
        """
        self._check_state()
        if np.allclose(self.C, self.basis.C):
            return self.basis.energy()
        else:
            return self.evaluate_energy_scheme()

    def _check_state(self):
        """
        Check if HF calculation has been run and converged before performing things such as energy evaluation.
        Raises error and warning respectively.
        """
        if not self.has_run:
            raise RuntimeError("No Hartree-Fock calculation has been run. Perform .run() first.")
        if not self.converged:
            raise RuntimeWarning("Hartree-Fock calculation has not converged")


def _identity_guess(basis: Basis) -> np.ndarray:
    # Static identity guess, "the standard"
    return np.eye(basis.L, basis.L)


def _core_guess(basis: Basis) -> np.ndarray:
    # Guess that diagonolizes the sp hamiltonian, presumably closer to the HF coef matrix than identity
    _, C = np.linalg.eigh(basis.h)
    return C
