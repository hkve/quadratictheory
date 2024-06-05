from quadratictheory import PyscfBasis
from quadratictheory.td.pulse import Sin2, DeltaKick
from unittest import TestCase
import numpy as np


class TestPulse(TestCase):
    def test_sin2(self):
        basis = PyscfBasis(atom="He 0 0 0", basis="sto-3g", restricted=False)

        u = np.ones(3) / np.sqrt(3)

        omega = 1.17
        tprime = 2 * np.pi / omega

        pulse = Sin2(u=u, F_str=1e-2, omega=omega, tprime=tprime)

        inside_pulse = pulse(tprime / 2, basis)
        outside_pulse = pulse(2 * tprime, basis)

        self.assertTrue(inside_pulse.shape == basis.h.shape)
        self.assertTrue(np.all(outside_pulse == np.zeros_like(basis.h)))
        self.assertTrue(inside_pulse.dtype == basis.h.dtype)

    def test_delta_kick(self):
        basis = PyscfBasis(atom="He 0 0 0", basis="sto-3g", restricted=False)

        u = np.array([1, 0, 0]).astype(float)

        F_str, dt = 0.1, 1e-3
        pulse = DeltaKick(u, F_str=F_str, dt=dt)

        inside_pulse = pulse(dt / 2, basis)
        outside_pulse = pulse(2 * dt, basis)

        self.assertTrue(inside_pulse.shape == basis.h.shape)
        self.assertTrue(np.all(outside_pulse == np.zeros_like(basis.h)))
        self.assertTrue(inside_pulse.dtype == basis.h.dtype)
