from __future__ import annotations
from clusterfock.basis import Basis
from clusterfock.cc.parameter import CoupledClusterParameter, merge_to_flat
from clusterfock.cc.coupledcluster import CoupledCluster
import numpy as np

from scipy.integrate import complex_ode, ode
from rk4_integrator.rk4 import Rk4Integrator

class TimeDependentCoupledCluster:
    def __init__(
        self, cc: CoupledCluster, time: tuple = (0, 1.0, 0.0001), integrator="Rk4Integrator"
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

        self._has_td_one_body = False
        self._has_td_two_body = False
        self._td_one_body = None
        self._td_two_body = None

        self._has_one_body_sampler = False
        self._one_body_sampler = None
        self._one_body_shapes = None
        self._one_body_results = {}
        self._two_body_results = {}

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

        self._t0 = cc._t.copy()
        self._l0 = cc._l.copy()

        y_initial, self.t_slice, self.l_slice = merge_to_flat(cc._l, cc._l)
        t_start, t_end, dt = self._t_start, self._t_end, self._dt

        assert dt > 0
        assert t_end > t_start
        n_time_points = int(np.ceil((t_end - t_start)/dt)) + 1

        integrator = complex_ode(self.rhs)
        integrator.set_integrator(self._integrator, dt=dt)
        integrator.set_initial_value(y_initial, t_start)

        energy = np.zeros(n_time_points, dtype=basis.dtype)
        overlap = np.zeros(n_time_points, dtype=basis.dtype)
        
        self._sample()
        energy[0] = cc.energy()
        overlap[0] = cc.overlap(self._t0, self._l0, cc._t, cc._l)

        t, counter = dt, 0
        while integrator.successful() and t < t_end + dt:
            if counter > n_time_points:
                break

            integrator.integrate(t) # Integrates to y(t)
            
            cc._t.from_flat(integrator.y[self.t_slice])
            cc._l.from_flat(integrator.y[self.l_slice])
            counter += 1
            
            if vocal: print(f"Done {counter}/{n_time_points-1}, t = {t}")
            
            self._sample()
            energy[counter] = cc.energy()
            overlap[counter] = cc.overlap(self._t0, self._l0, cc._t, cc._l)
            t += dt
        

        self.results = self._construct_results(energy, overlap)

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

        if self._has_td_one_body:
            external_contribution = self.external_one_body(t, basis)
            cc._f = basis.f + external_contribution

        t_dot = -1j*cc._t_rhs_timedependent(cc._t, cc._l)
        l_dot = 1j*cc._l_rhs_timedependent(cc._t, cc._l)

        y_dot, _, _ = merge_to_flat(t_dot, l_dot)

        return y_dot

    def _sample(self):
        """
        Internal function for sampling. Stores the results in '_one_body_results' dict.
        """
        basis, cc = self.basis, self.cc
        if self._has_one_body_sampler:
            cc.one_body_density()
            operators = self.one_body_sampler(basis)

            for key, operator in operators.items():
                sample = cc.one_body_expval(operator)
                self._one_body_results[key].append(sample)

    def _construct_results(self, energy: np.ndarray, overlap: np.ndarray) -> dict:
        """
        Constructs the results dict after the integration has been completed. 

        Args:
            energy (np.ndarray): Energies for each timestep
            overlap (np.ndarray): Overlap for each timestep

        Returns:
            results (dict): The combined results
        """
        t_start, t_end, dt = self._t_start, self._t_end, self._dt
        time = np.arange(t_start, t_end+dt, dt)

        results = {"t": time, "energy": energy, "overlap": overlap}

        if self._has_one_body_sampler:
            for k, v in self._one_body_results.items():
                self._one_body_results[k] = np.array(v)
        # if self._has_two_body_sampler:
        #     for k, v in self._two_body_results.items():
        #         self._two_body_results[k] = np.array(v)

        results.update(self._one_body_results)
        results.update(self._two_body_results)

        return results

    @property
    def external_one_body(self):
        return self._td_one_body
    
    @external_one_body.setter
    def external_one_body(self, func_ob):
        self._has_td_one_body = True
        self._td_one_body = func_ob

    @property
    def one_body_sampler(self):
        return self._one_body_sampler
    
    @one_body_sampler.setter
    def one_body_sampler(self, sampler):
        basis = self.basis
        self._has_one_body_sampler = True
        self._one_body_sampler = sampler

        res = sampler(basis)
        assert type(res) == dict, f"The return type of {sampler = } must be dict not {type(res) = }."

        for key in res.keys():
            self._one_body_results[key] = []