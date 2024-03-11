from __future__ import annotations
from clusterfock.basis import Basis
from clusterfock.mix import DIISMixer
from clusterfock.cc.parameter import CoupledClusterParameter
from abc import ABC, abstractmethod
from functools import reduce
import operator
import numpy as np


class CoupledCluster(ABC):
    @abstractmethod
    def __init__(self, basis: Basis, t_orders: list, l_orders: list = None):
        """
        Abstract base class for implementing a Coupled Cluster method, where the t and lambda amplitudes
        can be solved indipendently. Energy, in addition to density matricies and expectation values
        can be calculated here.

        Args:
            basis (Basis): Basis set for the Coupled Cluster calculation.
            t_orders (list): List of orders for the excitation operator T.
            l_orders (list, optional): List of orders for the cluster operator L. Defaults to None.

        Attributes:
            basis (Basis): Basis set for the Coupled Cluster calculation.
            _f (np.ndarray): Fock matrix calculated based on the given basis.
            has_run (bool): Flag indicating whether the calculation has been executed.
            mixer (Mixer): Mixer for Newtons method. Defaults to the direct inversion in the iterative subspace method.
            _t (CoupledClusterParameter): Parameter for the excitation operator T.
            _l (CoupledClusterParameter): Parameter for the cluster operator L.
            _epsinv (CoupledClusterParameter): Inverse of the energy denominators.
            _t_info (dict): Information dictionary for the excitation operator T.
            _l_info (dict): Information dictionary for the cluster operator L.
            rho_ob (np.ndarray): Placeholder for the one-body reduced density matrix.
            rho_tb (np.ndarray): Placeholder for the two-body reduced density matrix.
        """
        self.basis = basis
        basis.calculate_fock_matrix()
        self._f = self.basis.f.copy()

        self.mixer = DIISMixer(n_vectors=8)

        self._t = CoupledClusterParameter(t_orders, basis.N, basis.M, self.basis.dtype)
        self._l = CoupledClusterParameter(l_orders, basis.N, basis.M, self.basis.dtype)
        self._epsinv = CoupledClusterParameter(t_orders, basis.N, basis.M, self.basis.dtype)

        self._t_info = {"run": None, "converged": False, "iters": 0}
        self._l_info = self._t_info.copy()

        self.rho_ob = None
        self.rho_tb = None

        self.transforms_basis = False

    def run(
        self, tol: float = 1e-8, maxiters: int = 1000, include_l: bool = False, vocal: bool = False
    ) -> CoupledCluster:
        """
        Main run method, which performs iterative solving of t and l (if derived method implments this).
        Basics such as convergence paramters can be passed.

        Args:
            tol (float): The tolerance which to holde t and l rhs functions to
            maxiters (int): Maximum number of iterations
            include_l (bool): If lambda amplitudes should be solved for as well
            vocal (bool): If iteration info should be printed to screen.

        Returns:
            self (CoupledCluster): The instance with solved amplitudes
        """
        basis = self.basis

        self._t.initialize_zero()
        self._epsinv.initialize_epsilon(epsilon=np.diag(self._f), inv=True)

        self._iterate_t(tol, maxiters, vocal)

        if include_l:
            assert (
                self._l is not None
            ), f"This scheme does not implment lambda equations, {self._l = }"

            self._l.initialize_zero()
            self.mixer.reset()
            self._iterate_l(tol, maxiters, vocal)

        return self

    def _iterate_t(self, tol: float, maxiters: int, vocal: bool):
        """
        Function to iterate t amplitudes. Solves f(t) = 0 with mixing and stores info.

        Args:
            tol (float): Tolarnce which t amplitudes must achive
            maxiters (int): maximum number of iterations
            vocal (bool): whether to print info to screen
        """
        iters, diff = 0, 1000
        corr_energy = 0

        t, epsinv = self._t, self._epsinv
        converged = False
        self._t_norms = []

        while (iters < maxiters) and not converged:
            rhs = self._next_t_iteration(t)
            rhs_norms = rhs.norm()

            if np.all(np.array(list(rhs_norms.values())) < tol):
                converged = True

            t_next_flat = self.mixer(t.to_flat(), (rhs * epsinv).to_flat())
            t.from_flat(t_next_flat)

            corr_energy = self._evaluate_cc_energy()

            iters += 1

            self._t_norms.append(rhs_norms)
            if vocal:
                print(f"i = {iters}, {corr_energy = :.6e}, rhs_norms = {rhs_norms}")

        self._t = t
        self._t_info["run"] = True
        self._t_info["iters"] = iters
        if iters < maxiters:
            self._t_info["converged"] = True

    def _iterate_l(self, tol: float, maxiters: int, vocal: bool):
        """
        Function to iterate l amplitudes. Solves f(t, l) = 0 with mixing and stores info.

        Args:
            tol (float): Tolarnce which l amplitudes must achive
            maxiters (int): maximum number of iterations
            vocal (bool): whether to print info to screen
        """
        iters, diff = 0, 1000

        t, l, epsinv = self._t, self._l, self._epsinv
        converged = False
        self._l_norms = []

        while (iters < maxiters) and not converged:
            rhs = self._next_l_iteration(t, l)

            rhs_norms = rhs.norm()

            if np.all(np.array(list(rhs_norms.values())) < tol):
                converged = True

            l_next_flat = self.mixer(l.to_flat(), (rhs * epsinv).to_flat())
            l.from_flat(l_next_flat)

            iters += 1
            self._l_norms.append(rhs_norms)

            if vocal:
                print(f"i = {iters}, rhs_norms = {rhs_norms}")

        self._l = l
        self._l_info["run"] = True
        self._l_info["iters"] = iters
        if iters < maxiters:
            self._l_info["converged"] = True

    @abstractmethod
    def _next_t_iteration(self, t: CoupledClusterParameter) -> CoupledClusterParameter:
        """
        Takes in a set of amplitudes and returns rhs of equation

        Args:
            t (CoupledClusterParameter): The amplitudes at this iteration

        Returns:
            rhs_t (CoupledClusterParameter) The rhs of equation at this iteration
        """
        pass

    def _next_l_iteration(
        self, t: CoupledClusterParameter, l: CoupledClusterParameter
    ) -> CoupledClusterParameter:
        """
        Takes in a set of amplitudes and returns rhs of equation

        Args:
            l (CoupledClusterParameter): The amplitudes at this iteration

        Returns:
            rhs_l (CoupledClusterParameter) The rhs of equation at this iteration
        """
        pass

    def _t_rhs_timedependent(
        self, t: CoupledClusterParameter, l: CoupledClusterParameter
    ) -> CoupledClusterParameter:
        """
        Adds additional calculations to '_next_t_iteration' if this is required by the
        time evelution equations for the specific CC scheme. For standard coupled cluster, this
        adds nothing

        Args:
            t (CoupledClusterParameter): The amplitude at this iteration

        Returns:
            rhs_t (CoupledClusterParameter): The rhs of the time-dependent equation
        """
        rhs_t = self._next_t_iteration(t)
        # print(f"RHS(T) sym upper: {np.linalg.norm(rhs_t[2] + rhs_t[2].transpose(1,0,2,3))}")
        # print(f"RHS(T) sym lower: {np.linalg.norm(rhs_t[2] + rhs_t[2].transpose(0,1,3,2))}")
        # print(f"RHS(T) sym both: {np.linalg.norm(rhs_t[2] - rhs_t[2].transpose(1,0,3,2))}")
        return rhs_t

    def _l_rhs_timedependent(
        self, t: CoupledClusterParameter, l: CoupledClusterParameter
    ) -> CoupledClusterParameter:
        """
        Adds additional calculations to '_next_l_iteration' if this is required by the
        time evelution equations for the specific CC scheme. For standard coupled cluster, this
        adds nothing

        Args:
            t (CoupledClusterParameter): The amplitude at this iteration
            l (CoupledClusterParameter): The Lambda-amplitude at this iteration

        Returns:
            rhs_l (CoupledClusterParameter): The rhs of the time-dependent equation
        """
        rhs_l = self._next_l_iteration(t, l)
        # print(f"RHS(L) sym upper: {np.linalg.norm(rhs_l[2] + rhs_l[2].transpose(1,0,2,3))}")
        # print(f"RHS(L) sym lower: {np.linalg.norm(rhs_l[2] + rhs_l[2].transpose(0,1,3,2))}")
        # print(f"RHS(L) sym both: {np.linalg.norm(rhs_l[2] - rhs_l[2].transpose(1,0,3,2))}")
        return rhs_l

    @abstractmethod
    def _evaluate_cc_energy() -> float:
        """
        Function to evaluate the energy. Calculates energy based on stored amplitudes

        Returns:
            energy (float): Correlation energy of the time-independent solution
        """
        pass

    def _evaluate_tdcc_energy(self) -> float:
        """
        Additional part to energy that can be added in the case where rhs_t = rhs_l = 0 might not be true.
        Defaults to no addition, but should be overwritten in the case it differs

        Returns:
            energy (float): Correlation energy additon of the time-dependent solution
        """
        return 0

    def _calculated_one_body_density(self) -> np.ndarray:
        """
        Helper function to calculate one-body density.

        Returns:
            rho_ob (np.ndarray): The one-body density, not quaranteed to be hermitian
        """
        raise NotImplementedError("This scheme does not implement one body densities")

    def _calculated_one_body_density(self) -> np.ndarray:
        """
        Helper function to calculate two-body density.

        Returns:
            rho_ob (np.ndarray): The two-body density
        """
        raise NotImplementedError("This scheme does not implement two body densities")

    def _check_valid_for_densities(self):
        """
        Helper function to check if l calculation has been performed before calculating any density matricies.
        Only warn if initialization and a run has been performed, but no convergence is achieved.
        """
        if self._l is None:
            raise RuntimeError("Expectation values without lambdas has not been implemented")
        if not self.l_info["run"]:
            raise RuntimeError("No lambda computation has been run. Perform .run() first.")
        if not self.l_info["converged"]:
            raise RuntimeWarning("Lambda computation did not converge")

    def one_body_density(self) -> np.ndarray:
        """
        Stores the one-body density as class variable and returns it

        Returns:
            rho_ob (np.ndarray): The (L,L) one-body density
        """
        self.rho_ob = self._calculate_one_body_density()

        return self.rho_ob

    def two_body_density(self) -> np.ndarray:
        """
        Stores the two-body density as class variable and returns it

        Returns:
            rho_ob (np.ndarray): The (L,L,L,L) two-body density
        """
        self.rho_tb = self._calculate_two_body_density()

        return self.rho_tb

    def densities(self):
        """
        Wrapper function to check validity of results, in addition to computing and storing one and two
        body densities.
        """
        self._check_valid_for_densities()

        self.one_body_density()
        self.two_body_density()

    def _check_valid_for_energy(self):
        """
        Helper function to check if l calculation has been performed before calculating any density matricies.
        Only warn if initialization and a run has been performed, but no convergence is achieved.
        """
        if self._t is None:
            raise RuntimeError("No t amplitudes have been initialized")
        if not self.t_info["run"]:
            raise RuntimeError("No t computation has been run. Perform .run() first.")
        if not self.t_info["converged"]:
            raise RuntimeWarning("t amplitude computation did not converge")

    def energy(self) -> float:
        """
        Wrapper function to calculate energy.

        Returns:
            energy (float): The energy calculated using t
        """
        self._check_valid_for_energy()

        return self._evaluate_cc_energy() + self.basis.energy()

    def time_dependent_energy(self) -> float:
        """
        Wrapper function to calculate energy.

        Returns:
            energy (float): The energy calculated using t
        """
        return self._evaluate_tdcc_energy() + self.energy()

    def one_body_expval(self, operator: np.ndarray) -> np.ndarray:
        """
        Calculates the one-body expectation value for an observable. The operator matrix representation must be
        passed. If the operator is not a scalar, the extra dimensions must be provided BEFORE the basis function
        indicies.

        Args:
            operator (np.ndarray): Matrix representation of the one-body operator

        Returns:
            expectation_value (np.ndarray): Expectation value based on one body density
        """
        if self.rho_ob is None:
            self.one_body_density()
        return np.einsum("...pq,pq->...", operator, self.rho_ob)

    def two_body_expval(self, operator: np.ndarray) -> np.ndarray:
        """
        Calculates the two-body expectation value for an observable. The operator matrix representation must be
        passed. If the operator is not a scalar, the extra dimensions must be provided BEFORE the basis function
        indicies.

        Args:
            operator (np.ndarray): Matrix representation of the two-body operator

        Returns:
            expectation_value (np.ndarray): Expectation value based on one two-body density
        """
        if self.rho_tb is None:
            self.two_body_density()
        asym = 1 if self.basis.restricted else 0.5
        return asym * np.einsum("...pqrs,pqrs->...", operator, self.rho_tb)

    def _overlap(self, t0, l0, t, l):
        raise NotImplementedError("This scheme does not implement overlap")

    def overlap(self, t0, l0, t, l):
        return self._overlap(t0, l0, t, l)

    def get_lowest_norm(self):
        if self.t_info["run"]:
            norms = np.array([list(norm.values()) for norm in self._t_norms])
        if self.t_info["run"]:
            l_norms = np.array([list(norm.values()) for norm in self._l_norms])
            norms = np.concatenate(norms, l_norms)

        return norms.min()
    
    # Property wrappers for convergence info dicts
    @property
    def t_info(self):
        return self._t_info

    @property
    def l_info(self):
        return self._l_info

    @property
    def info(self):
        merged = {
            **{"t_" + k: v for k, v in self.t_info.items()},
            **{"l_" + k: v for k, v in self.l_info.items()},
        }
        return merged
