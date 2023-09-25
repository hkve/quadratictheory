from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class Basis(ABC):
    def __init__(self, L: int, N: int, restricted: bool = False, dtype=float):
        """
        Initializes an instance of a quantum many-body system with specified parameters.

        Parameters:
        - L (int): The total number of single-particle states (sp states).
        - N (int): The total number of particles.
        - restricted (bool, optional): Whether to use a restricted scheme where all sp states are doubly occupied. Defaults to False.
        - dtype (type, optional): The data type for internal arrays (e.g., float64). Defaults to float.

        Notes:
        - If 'restricted' is True, the system follows the restricted scheme, and both 'L' and 'N' must be even.
        - 'restricted' determines the degeneracy of sp states (2 for restricted, 1 for unrestricted).
        - The instance will have various attributes related to the basis set, symmetry, and matrix elements.
        - Internal arrays 'h', 'u', 'f', and 's' are initialized with zeros.
        - 'C' is initialized as an identity matrix.

        Attributes:
        - restricted (bool): Indicates whether the restricted scheme is used.
        - orthonormal (bool): Indicates if the basis states are orthonormal (True by default).
        - antisymmetric (bool): Indicates if the wave function is antisymmetric (False by default).
        - dtype (type): The data type used for internal arrays.
        - _L (int): Effective basis size considering degeneracy and 'L'.
        - _N (int): Effective particle number considering degeneracy and 'N'.
        - _M (int): Number of unoccupied states (_L - _N).
        - _o (slice): Slice for occupied states.
        - _v (slice): Slice for unoccupied states.
        - _energy_shift (float): Constant to add to energy, mostly relevant for Molecules proton repulsion.
        - _one_body_shape (tuple): Shape of the one-body operator matrix.
        - _two_body_shape (tuple): Shape of the two-body operator matrix.
        - _h (ndarray): One-body operator matrix.
        - _u (ndarray): Two-body operator matrix.
        - _f (ndarray): Fock operator matrix.
        - _s (ndarray): Overlap matrix.
        - _C (ndarray): Coefficient matrix.

        Raises:
        - AssertionError: If 'restricted' is True but 'L' or 'N' is not even.
        """

        # If restricted scheme is used, all sp states are doubly occupied
        if restricted:
            self._degeneracy = 2
            assert L % 2 == 0, "#Basis function must be even in restricted scheme"
            assert N % 2 == 0, "#Particles must be even in restricted scheme"
        else:
            self._degeneracy = 1

        # Booleans, checks sp scheme, overlap and symmetries of u and type of matrix elements
        self.restricted = restricted
        self.orthonormal = True
        self.antisymmetric = False
        self.dtype = dtype

        # Store basis set sizes
        self._L = L // self._degeneracy
        self._N = N // self._degeneracy
        self._M = self._L - self._N

        self._o = slice(0, self.N)
        self._v = slice(self.N, self.L)

        #
        self._energy_shift = 0
        self._one_body_shape = (self.L, self.L)
        self._two_body_shape = (self.L, self.L, self.L, self.L)

        self._h = np.zeros(shape=(self.L, self.L), dtype=dtype)
        self._u = np.zeros(shape=(self.L, self.L, self.L, self.L), dtype=dtype)
        self._f = np.zeros(shape=(self.L, self.L), dtype=dtype)
        self._s = np.eye(self.L, dtype=dtype)
        self._C = np.eye(self.L, dtype=dtype)

    def calculate_fock_matrix(self):
        """
        Calculates the fock matrix and stores as attribute
        """
        h, u, o, v = self.h, self.u, self.o, self.v

        self.f = h.copy()
        if self.restricted:
            D = np.einsum("piqi->pq", u[:, o, :, o], optimize=True)
            E = np.einsum("piiq->pq", u[:, o, o, :], optimize=True)
            self.f += 2 * D - E
        else:
            self.f += np.einsum("piqi->pq", u[:, o, :, o])

    def _change_basis_one_body(self, operator: np.ndarray, C: np.ndarray) -> np.ndarray:
        """
        Perform basis change on a one body operator

        Parameters:
            - operator (ndarray): (L,L) matrix to transform
            - C (ndarray): (L,L) matrix used to perform transformation

        Returns:
            - operator_transformed (ndarray): (L,L) matrix with the new operator
        """

        return np.einsum("ai,bj,ab->ij", C.conj(), C, operator, optimize=True)

    def _change_basis_two_body(self, operator: np.ndarray, C: np.ndarray) -> np.ndarray:
        """
        Perform basis change on a two body operator

        Parameters:
            - operator (ndarray): (L,L,L,L) matrix to transform
            - C (ndarray): (L,L) matrix used to perform transformation

        Returns:
            - operator_transformed (ndarray): (L,L,L,L) matrix with the new operator
        """

        return np.einsum(
            "ai,bj,gk,dl,abgd->ijkl", C.conj(), C.conj(), C, C, operator, optimize=True
        )

    def copy(self) -> Basis:
        """
        Copies self to another Basis object. Useful i.e. if a transformation of basis is
        wanted but the orginal representation should not be destroyed

        Returns:
            - new_basis (Basis): The new copy of self
        """
        L, N = self._degeneracy * self.L, self._degeneracy * self.N
        new_basis = Basis(L=L, N=N, restricted=self.restricted, dtype=self.dtype)
        new_basis.orthonormal = self.orthonormal
        new_basis.antisymmetric = self.antisymmetric

        new_basis.h = self.h.copy()
        new_basis.u = self.u.copy()
        new_basis.s = self.s.copy()
        new_basis.C = self.C.copy()
        new_basis.calculate_fock_matrix()

        return new_basis

    def change_basis(
        self, C: np.ndarray, inplace: bool = True, inverse: bool = False
    ) -> Optional[Basis]:
        """
        Perform basis change on a basis object. This method can work both in place and
        also perform inverse transformation.

        Parameters:
            - C (ndarray): The (L,L) matrix to use for basis change
            - inplace (bool): Whether to perform the basis transformation on self or return a new object
            - inverse (bool): Whether to perform the inverse transformation

        Returns:
            - obj (Basis): The basis with transformation applied. If inplace is true, this is not needed
        """
        if inverse:
            C = C.conj().T

        obj = self if inplace else self.copy()

        obj.h = obj._change_basis_one_body(obj.h, C)
        obj.u = obj._change_basis_two_body(obj.u, C)
        obj.s = obj._change_basis_one_body(obj.s, C)
        obj.C = C
        obj.calculate_fock_matrix()

        return obj

    def _antisymmetrize(self):
        """
        Antisymmetrizes two body u in place
        """
        self.antisymmetric = True
        self.u = self.u - self.u.transpose(0, 1, 3, 2)

    def _add_spin(self):
        """
        Add spins when going from restriced to scheme. One body operators go from (L,L) to (2L, 2L),
        while two body (L,L,L,L) to (2L,2L,2L,2L). Also fixes new L and N numbers and degeneracy
        """
        self.restricted = False
        self._degeneracy = 1

        I = np.eye(2)
        I2 = np.einsum("pr, qs -> pqrs", I, I)

        self.h = np.kron(self.h, I)
        self.u = np.kron(self.u, I2)
        self.s = np.kron(self.s, I)
        self.C = np.kron(self.C, I)
        if self.f is not None:
            self.f = np.kron(self.f, I)

        self._L = 2 * self._L
        self.N = 2 * self.N

    def from_restricted(self, inplace: bool = True):
        """
        Go from restricted to non-restricted scheme. Expands matricies and performs antisymmetrization.

        Parameters:
            - inplace (bool): Whether to work on this object or return new

        Returns:
            - obj (Basis): The object in non-restricted scheme. If inpalce is True, this is not needed.
        """
        obj = self if inplace else self.copy()

        obj._add_spin()
        obj._antisymmetrize()

        return obj

    def energy(self) -> float:
        """
        Calcualtes the energy expectation value of state.

        Returns:
            - Energy (float): The energy expectation value
        """
        h, u = self.h, self.u
        o, v = self.o, self.v

        E_OB = self._degeneracy * h[o, o].trace()
        E_TB = 0

        if self.restricted:
            E_TB = 2 * np.einsum("ijij", u[o, o, o, o]) - np.einsum("ijji", u[o, o, o, o])
        else:
            E_TB = 0.5 * np.einsum("ijij", u[o, o, o, o])

        return E_OB + E_TB + self._energy_shift

    """
    Getters and setters for ints (L,N) and slices (o,v). N also performs tweaking on M
    as this is easiest to understand (adding particles reduces the number of viritual states if L is const).
    """

    @property
    def L(self) -> int:
        return self._L

    @property
    def N(self) -> int:
        return self._N

    @N.setter
    def N(self, N: int) -> int:
        assert N % 2 == 0, "#Particles must be even in restricted scheme"
        N = N // self._degeneracy
        assert N <= self.L, f"#Particles = {N} must be larger than #Basis functions = {self.L}"
        self._N = N
        self._M = self.L - self.N

        self._o = slice(0, self.N)
        self._v = slice(self.N, self.L)
        self.calculate_fock_matrix()

    @property
    def M(self) -> int:
        return self._M

    @property
    def o(self) -> slice:
        return self._o

    @property
    def v(self) -> slice:
        return self._v

    # Getters and setters for matricies h, u, f and s
    @property
    def h(self) -> np.ndarray:
        return self._h

    @h.setter
    def h(self, h: np.ndarray):
        self._h = h.astype(self.dtype)

    @property
    def u(self) -> np.ndarray:
        return self._u

    @u.setter
    def u(self, u: np.ndarray):
        self._u = u.astype(self.dtype)

    @property
    def f(self) -> np.ndarray:
        return self._f

    @f.setter
    def f(self, f: np.ndarray):
        self._f = f.astype(self.dtype)

    @property
    def s(self) -> np.ndarray:
        return self._s

    @s.setter
    def s(self, s: np.ndarray):
        self._s = s.astype(self.dtype)

    @property
    def C(self) -> np.ndarray:
        return self._C

    @C.setter
    def C(self, C: np.ndarray):
        self._C = C.astype(self.dtype)
