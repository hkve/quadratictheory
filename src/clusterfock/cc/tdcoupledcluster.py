from __future__ import annotations
from clusterfock.basis import Basis
from clusterfock.cc.parameter import CoupledClusterParameter
from clusterfock.cc.coupledcluster import CoupledCluster
import numpy as np

from rk4_integrator import Rk4Integrator


class TimeDependentCoupledCluster:
    def __init__(self, cc: CoupledCluster, dt: float = 1e-4):
        self.cc = cc
        self.basis = cc.basis
        self.integrator = Rk4Integrator(dt)

    def run(self, vocal=False):
        cc = self.cc
        if not (cc.t_info["run"] and cc.l_info["run"]):
            cc.run(include_l=True, vocal=vocal)

    @property
    def dt(self):
        return self.integrator.dt

    @dt.setter
    def dt(self, dt):
        self.integrator.dt = dt
