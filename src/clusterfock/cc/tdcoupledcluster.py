from __future__ import annotations
from clusterfock.basis import Basis
from clusterfock.cc.parameter import CoupledClusterParameter, merge_to_flat
from clusterfock.cc.coupledcluster import CoupledCluster
import numpy as np

from scipy.integrate import complex_ode


class TimeDependentCoupledCluster:
    def __init__(
        self, cc: CoupledCluster, time: tuple = (0, 10.0, 0.0001), integrator="Rk4Integrator"
    ):
        self.cc = cc
        self.basis = cc.basis

        self._t_start, self._t_end, self.dt = time
        self._integrator = integrator

    def run(self, vocal=False):
        cc, basis = self.cc, self.basis

        if not (cc.t_info["run"] and cc.l_info["run"]):
            cc.run(include_l=True, vocal=vocal)
        if not basis.dtype == complex:
            basis.dtype = complex
            cc._t.dtype = complex
            cc._l.dtype = complex

        y_initial, self.t_slice, self.l_slice = merge_to_flat(cc._l, cc._l)
        times = self.t

        integrator = complex_ode(self.f)
        integrator.set_integrator(self._integrator, dt=self.dt)
        integrator.set_initial_value(y_initial, times[0])

        n = 100000
        for i in range(1,n):
            integrator.integrate(times[i])

    def f(self, t, y):
        # y comes in flat and should return flat, but be evaluated in the meen while
        # it contains both t and l
        basis, cc = self.basis, self.cc
        cc._t.from_flat(y[self.t_slice])        
        cc._l.from_flat(y[self.l_slice])        
        
        # Here I should add rhs and lhs updates to
        t_dot = 1j*cc._next_t_iteration(cc._t)
        l_dot = -1j*cc._next_l_iteration(cc._t, cc._l)

        y, _, _ = merge_to_flat(t_dot, l_dot)

        return y

    @property
    def dt(self):
        return self._dt

    @dt.setter
    def dt(self, dt):
        self.t = np.arange(0, self._t_end + dt, step=dt)
        self._dt = dt
