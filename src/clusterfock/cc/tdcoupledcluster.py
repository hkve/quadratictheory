from __future__ import annotations
from clusterfock.basis import Basis
from clusterfock.cc.parameter import CoupledClusterParameter
from clusterfock.cc.coupledcluster import CoupledCluster
import numpy as np

from scipy.integrate import complex_ode


class TimeDependentCoupledCluster:
    def __init__(
        self, cc: CoupledCluster, time: tuple = (0, 10.0, 0.1), integrator="Rk4Integrator"
    ):
        self.cc = cc
        self.basis = cc.basis

        self._t_start, self._t_end, self._dt = time
        self._integrator = integrator

    def run(self, vocal=False):
        cc, basis = self.cc, self.basis

        if not (cc.t_info["run"] and cc.l_info["run"]):
            cc.run(include_l=True, vocal=vocal)
        if not basis.dtype == complex:
            basis.dtype = complex
            cc._t.dtype = complex
            cc._l.dtype = complex

        integrator = complex_ode(self.f)
        integrator.set_integrator(self._integrator, dt=self.dt)

    def f(self):
        raise NotImplementedError("You must think before you act here :^^^^)")

    @property
    def dt(self):
        return self._dt

    @dt.setter
    def dt(self, dt):
        self.t = np.arange(0, self._t_end + dt, step=dt)
        self._dt = dt
