from __future__ import annotations
from clusterfock.basis import Basis
from clusterfock.cc.parameter import CoupledClusterParameter, merge_to_flat
from clusterfock.cc.coupledcluster import CoupledCluster
from clusterfock.td.sampler import Sampler

import tqdm
import numpy as np
from scipy.integrate import complex_ode, ode
from rk4_integrator.rk4 import Rk4Integrator
from gauss_integrator.gauss import GaussIntegrator


class TimeDependentCoupledCluster:
    def __init__(
        self,
        cc: CoupledCluster,
        time: tuple = (0, 1.0, 0.0001),
        integrator="Rk4Integrator",
        integrator_args={},
    ):
        """
        Constructor for time propegation using a cc calculation. Supports scipy-type integrators.
        For dynamics, time dependent one and two body potentials can be set using

        instance.external_one_body = f(t, basis)
        instance.external_two_body = g(t, basis)

        The functions f and g will then be called for each intergration (sub-)step. They
        must return a (L,L) or (L,L,L,L) ndarray, which is added to the one or two body
        hamiltonian respectively.

        Similarly to save expectation values during calculations, a sampler function can also
        be set. This is set using the 'one_body_sampler' property.

        instance.one_body_sampler = sampler1(basis)
        instance.two_body_sampler = sampler2(basis)

        The functions sampler1 and sampler2 must return a dictionary with key matching the observables
        name in basis, and the corresponding observable matrix. For instance, to sample posistions

        def sampler1(basis)
            return {"r": basis.r}

        is valid, if the basis instance contains functionality that allows for calculating r.

        Note that the ENERGY observable is always calculated, in addition to the time dependent
        overlap with the ground state.
        """
        self.cc = cc
        self.basis = cc.basis

        self._t_start, self._t_end, self._dt = time
        self._integrator = integrator
        self._integrator_args = integrator_args

        if self._integrator == "Rk4Integrator" and self._integrator_args == {}:
            self._integrator_args = {"dt": self._dt}

        self._has_td_one_body = False
        self._has_td_two_body = False
        self._td_one_body = None
        self._td_two_body = None

        self.sampler = Sampler()
        self.results = {}

    def run(self, vocal: bool = False) -> dict:
        """
        Main function to run the time evolution of the CC state. If the passed cc
        instance has not been run, it will be. Also casts CC matrix elements to
        complex and sets up integration points and integrator. Lastly the main integration
        loop is run with sampling for each complete step. Resturns a dict with all sampled
        properties.

        Args:
            vocal (bool): If diagnostics should be shown for every step

        Returns:
            result (dict): Dictionary with the sampled properties for each time step
        """
        cc, basis = self.cc, self.basis

        if not (cc.t_info["run"] and cc.l_info["run"]):
            # if self._has_td_one_body:
            #     external_contribution = self.external_one_body(self._t_start, basis)
            #     cc._f += external_contribution
            # cc.run(include_l=True, vocal=vocal)
            raise AttributeError("You need to run the CC with lambda amplitudes!!!")
        if not basis.dtype == complex:
            basis.dtype = complex
            cc._t.dtype = complex
            cc._l.dtype = complex
            cc._f = cc._f.astype(complex)

            if cc.transforms_basis:
                cc._h = cc._h.astype(complex)
                cc._u = cc._u.astype(complex)

        self._setup_sample(basis)

        if cc.transforms_basis:
            cc.copy_cached_operators()
            cc.perform_t1_transform(cc._t[1])

        y_initial, self.t_slice, self.l_slice = merge_to_flat(cc._t, cc._l)
        t_start, t_end, dt = self._t_start, self._t_end, self._dt

        assert dt > 0
        assert t_end > t_start
        n_time_points = int((t_end - t_start) / dt) + 1
        time_points = np.linspace(t_start, t_end, n_time_points)

        integrator = complex_ode(self.rhs)
        integrator.set_integrator(self._integrator, **self._integrator_args)
        integrator.set_initial_value(y_initial, t_start)

        self._sample()

        loop_range = range(n_time_points - 1)
        if vocal:
            loop_range = tqdm.tqdm(range(n_time_points - 1))

        unsuccessful_index = None

        for i in loop_range:
            integrator.integrate(integrator.t + dt)

            if not integrator.successful():
                unsuccessful_index = i - 1
                break

            cc._t.from_flat(integrator.y[self.t_slice])
            cc._l.from_flat(integrator.y[self.l_slice])

            self._sample()

        if unsuccessful_index is not None:
            time_points = time_points[:unsuccessful_index]

        self.results = self._construct_results(time_points)

        return self.results

    def rhs(self, t: float, y: np.ndarray) -> np.ndarray:
        """
        Evaluates the rhs for each partial step. Since scipy integrater requires flat arrays
        the amplitudes are passed as a long flat array y, reshaped in the CC instance for calculations
        and then after flattend for the next integration step.

        Args:
            t (float): The current time
            y (np.ndarray): The amplitudes in a flat numpy array

        Returns:
            y_dot (np.ndarray): The current amplitude derivatives
        """
        basis, cc = self.basis, self.cc

        # Update t and l amplitudes in basis
        cc._t.from_flat(y[self.t_slice])
        cc._l.from_flat(y[self.l_slice])

        # Adds time dependent hamiltonian to fock matrix if present
        if self._has_td_one_body:
            external_contribution = self.external_one_body(t, basis)
            
            # If T1-transform, pass it to CC class seperatly since it needs to be transformed as well
            # Else add it to CC storage of Fock matrix
            if cc.transforms_basis:
                cc._external_contribution = self.external_one_body(t, basis)
            else:
                cc._f = basis.f + external_contribution

        t_dot = -1j * cc._t_rhs_timedependent(cc._t, cc._l)
        l_dot = 1j * cc._l_rhs_timedependent(cc._t, cc._l)

        y_dot, _, _ = merge_to_flat(t_dot, l_dot)

        return y_dot

    def _sample(self):
        """
        Internal function for sampling. Stores the results in '_one_body_results' dict.
        """
        cc, basis = self.cc, self.basis
        if self.sampler.has_one_body:
            cc.one_body_density()
            operators = self.sampler.one_body(basis)

            for key, operator in operators.items():
                sample = cc.one_body_expval(operator)
                self.results[key].append(sample)

        if self.sampler.has_two_body:
            cc.two_body_density()
            operators = self.sampler.two_body(basis)

            for key, operator in operators.items():
                sample = cc.two_body_expval(operator)
                self.results[key].append(sample)

        if self.sampler.has_misc:
            misc = self.sampler.misc(self)

            for key, value in misc.items():
                self.results[key].append(value)

    def _setup_sample(self, basis):
        if self.sampler.has_one_body:
            self.cc.one_body_density()
        if self.sampler.has_two_body:
            self.cc.two_body_density()

        if self.sampler.has_overlap:
            self._t0 = self.cc._t.copy()
            self._l0 = self.cc._l.copy()

        all_keys = self.sampler.setup_sampler(basis, self)

        for key in all_keys:
            self.results[key] = []

    def _construct_results(self, time_points: np.ndarray) -> dict:
        """
        Constructs the results dict after the integration has been completed.

        Args:
            time_points (np.ndarray): The times the different quantities are calcualted for

        Returns:
            results (dict): The combined results
        """
        for k, v in self.results.items():
            self.results[k] = np.array(v)

        self.results["t"] = time_points

        return self.results

    @property
    def external_one_body(self):
        return self._td_one_body

    @external_one_body.setter
    def external_one_body(self, func_ob):
        self._has_td_one_body = True
        self._td_one_body = func_ob

        if self.cc.transforms_basis:
            self.cc._has_td_one_body = self._has_td_one_body # Only needs to be done for transform

    @property
    def sampler(self):
        return self._sampler

    @sampler.setter
    def sampler(self, sampler):
        self._sampler = sampler
