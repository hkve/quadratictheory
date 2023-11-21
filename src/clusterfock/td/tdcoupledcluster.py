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

    def run(self, vocal=False):
        cc, basis = self.cc, self.basis

        if not (cc.t_info["run"] or cc.l_info["run"]):
            if self._has_td_one_body:
                external_contribution = self.external_one_body(self._t_start, basis)
                cc._f += external_contribution
            cc.run(include_l=True, vocal=vocal)
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
        
        energy[0] = cc.energy()
        overlap[0] = cc.overlap(self._t0, self._l0, cc._t, cc._l)

        t, counter = dt, 0
        while integrator.successful() and t < t_end:
            integrator.integrate(t) # Integrates to y(t)
            
            cc._t.from_flat(integrator.y[self.t_slice])
            cc._l.from_flat(integrator.y[self.l_slice])
            counter += 1
            
            if vocal: print(f"Done {counter}/{n_time_points}, t = {t}")
            
            if counter >= n_time_points:
                break
            
            energy[counter] = cc.energy()
            overlap[counter] = cc.overlap(self._t0, self._l0, cc._t, cc._l)
            t += dt

        return np.arange(t_start, t_end+dt, dt), energy, overlap
    

    def rhs(self, t, y):
        basis, cc = self.basis, self.cc

        # Update t and l amplitudes in basis
        cc._t.from_flat(y[self.t_slice])       
        cc._l.from_flat(y[self.l_slice])

        if self._has_td_one_body:
            external_contribution = self.external_one_body(t, basis)
            cc._f = basis.f + external_contribution

        t_dot = -1j*cc._next_t_iteration(cc._t)
        l_dot = 1j*cc._next_l_iteration(cc._t, cc._l)

        y_dot, _, _ = merge_to_flat(t_dot, l_dot)

        return y_dot

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
        self._one_body_shapes = []

        args = sampler(basis)
        if type(args) is not tuple:
            agrs = (args,)
            sampler_wrap = lambda basis: (sampler(basis), )
        else:
            sampler_wrap = sampler

        self._one_body_sampler = sampler_wrap

        for arg in args:
            print(arg.shape)
