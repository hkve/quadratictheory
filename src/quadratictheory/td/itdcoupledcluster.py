from __future__ import annotations
from quadratictheory.basis import Basis
from quadratictheory.cc.parameter import CoupledClusterParameter, merge_to_flat
from quadratictheory.cc.coupledcluster import CoupledCluster
import quadratictheory.td.sampler as sampler
from quadratictheory.td.tdcoupledcluster import TimeDependentCoupledCluster

import tqdm
import numpy as np
from scipy.integrate import ode
from rk4_integrator.rk4 import Rk4Integrator
from gauss_integrator.gauss import GaussIntegrator


class ImaginaryTimeCoupledCluster(TimeDependentCoupledCluster):
    def __init__(
        self,
        cc: CoupledCluster,
        cc_gs: CoupledCluster,
        time: tuple = (0, 1.0, 0.01),
        integrator="Rk4Integrator",
        integrator_args={},
    ):
        super().__init__(cc=cc, time=time, integrator=integrator, integrator_args=integrator_args)
        self.sampler = sampler.ImagTimeSampler()
        self.cc_gs = cc_gs

    def run(self, vocal: bool = False) -> dict:
        cc, basis = self.cc, self.basis

        cc._t.initialize_zero()
        cc._l.initialize_zero()
        cc._t_info = {"run": True, "converged": True, "iters": 0}
        cc._l_info = cc._t_info.copy()

        self._setup_sample(basis)

        y_initial, self.t_slice, self.l_slice = merge_to_flat(cc._t, cc._l)
        t_start, t_end, dt = self._t_start, self._t_end, self._dt

        assert dt > 0
        assert t_end > t_start
        n_time_points = int((t_end - t_start) / dt) + 1
        time_points = np.linspace(t_start, t_end, n_time_points)

        integrator = ode(self.rhs)
        integrator.set_integrator(self._integrator, **self._integrator_args)
        integrator.set_initial_value(y_initial, t_start)

        self._sample()

        loop_range = range(n_time_points - 1)
        if vocal:
            loop_range = tqdm.tqdm(range(n_time_points - 1))

        unsuccessful_index = None
        l_dot_norm_prev = 1000
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

    def run_until_convergence(self, tol: float, vocal: bool = False):
        cc, basis = self.cc, self.basis

        cc._t.initialize_zero()
        cc._l.initialize_zero()
        cc._t_info = {"run": True, "converged": True, "iters": 0}
        cc._l_info = cc._t_info.copy()

        self._setup_sample(basis)

        y_initial, self.t_slice, self.l_slice = merge_to_flat(cc._t, cc._l)
        dt = self._dt

        assert dt > 0

        integrator = ode(self.rhs)
        integrator.set_integrator(self._integrator, **self._integrator_args)
        integrator.set_initial_value(y_initial, 0)

        self._sample()

        unsuccessful_index = None
        l_dot_norm_prev = 1000
        converged = False
        time_points = [0]

        while not converged:
            integrator.integrate(integrator.t + dt)

            cc._t.from_flat(integrator.y[self.t_slice])
            cc._l.from_flat(integrator.y[self.l_slice])

            t_norm = cc._t_rhs_timedependent(cc._t, cc._l).norm()
            l_norm = cc._l_rhs_timedependent(cc._t, cc._l).norm()

            converged = self._check_convergence(t_norm, l_norm, tol)

            if vocal:
                print(f"t = {integrator.t:.2f}, converged = {converged}, {t_norm = }, {l_norm = }")

            self._sample()
            time_points.append(integrator.t)

        time_points = np.array(time_points)
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

        t_dot = -1.0 * cc._t_rhs_timedependent(cc._t, cc._l)
        l_dot = -1.0 * cc._l_rhs_timedependent(cc._t, cc._l)

        y_dot, _, _ = merge_to_flat(t_dot, l_dot)

        return y_dot

    def _check_convergence(self, t_norms, l_norms, tol):
        t_norm_max = max([v for v in t_norms.values()])
        l_norm_max = max([v for v in l_norms.values()])

        norm_max = max([t_norm_max, l_norm_max])

        return norm_max < tol

    @property
    def external_one_body(self):
        raise NotImplementedError(f"{self.__class__.__name__} is a ground state solver!")

    @external_one_body.setter
    def external_one_body(self, func_ob):
        raise NotImplementedError(f"{self.__class__.__name__} is a ground state solver!")
