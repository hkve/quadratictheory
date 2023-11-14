from __future__ import annotations
from clusterfock.basis import Basis
from clusterfock.cc.parameter import CoupledClusterParameter, merge_to_flat
from clusterfock.cc.coupledcluster import CoupledCluster
import numpy as np

from scipy.integrate import complex_ode


class TimeDependentCoupledCluster:
    def __init__(
        self, cc: CoupledCluster, time: tuple = (0, 1.0, 0.0001), integrator="Rk4Integrator"
    ):
        self.cc = cc
        self.basis = cc.basis

        self._t_start, self._t_end, self._dt = time
        self._integrator = integrator

        self._has_td_ob = False
        self._has_td_tb = False
        self._td_ob = None
        self._td_tb = None

    def run(self, vocal=False):
        cc, basis = self.cc, self.basis

        if not (cc.t_info["run"] and cc.l_info["run"]):
            cc.run(include_l=True, vocal=vocal)
        if not basis.dtype == complex:
            basis.dtype = complex
            cc._t.dtype = complex
            cc._l.dtype = complex
            cc._f = cc._f.astype(complex)

        self._t0 = cc._t.copy()
        self._l0 = cc._l.copy()

        y_initial, self.t_slice, self.l_slice = merge_to_flat(cc._l, cc._l)
        t_start, t_end, dt = self.t_start, self.t_end, self.dt

        integrator = complex_ode(self.rhs)
        integrator.set_integrator(self._integrator, dt=self.dt)
        integrator.set_initial_value(y_initial, t_start)

        n_time_points = int((t_end - t_start)/dt)+1
        assert t_end > t_start

        energy = np.zeros(n_time_points, dtype=basis.dtype)
        energy[0] = cc.energy()
        
        t = t_start
        counter = 0
        while t < t_end:
            t += dt
            integrator.integrate(t)
            energy[counter] = cc.energy()

            counter += 1
        
        return np.arange(t_start, t_end+dt, dt), energy
    

    def rhs(self, t, y):
        # y comes in flat and should return flat, but be evaluated in the meen while
        # it contains both t and l
        basis, cc = self.basis, self.cc
        cc._t.from_flat(y[self.t_slice])        
        cc._l.from_flat(y[self.l_slice])
        
        if self._has_td_ob:
            cc._f = basis.f + self.external_one_body(t, basis)

        # Here I should add rhs and lhs updates to
        t_dot = 1j*cc._next_t_iteration(cc._t)
        l_dot = -1j*cc._next_l_iteration(cc._t, cc._l)

        y, _, _ = merge_to_flat(t_dot, l_dot)

        return y

    @property
    def external_one_body(self):
        return self._td_ob
    
    @external_one_body.setter
    def external_one_body(self, func_ob):
        self._has_td_ob = True
        self._td_ob = func_ob

    @property
    def dt(self):
        return self._dt

    @dt.setter
    def dt(self, dt):
        self._dt = dt

    @property
    def t_start(self):
        return self._t_start

    @dt.setter
    def t_start(self, t_start):
        self._t_start = t_start

    @property
    def t_end(self):
        return self._t_end

    @t_end.setter
    def t_end(self, t_end):
        self._t_end = t_end
