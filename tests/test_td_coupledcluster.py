import numpy as np
from unittest import TestCase
from quadratictheory.hf import RHF
from quadratictheory.cc import GCCD, GCCSD, QCCD, QCCSD
from quadratictheory.td import TimeDependentCoupledCluster
from quadratictheory.basis import PyscfBasis

from quadratictheory.td.pulse import Sin2
from quadratictheory.td.pulse import DeltaKick
from quadratictheory.td.sampler import DipoleSampler


class TestTimeDependentCoupledCluster(TestCase):
    def sin2_pulse(self, atom, basis, CC, tol=1e-8):
        basis = PyscfBasis(atom=atom, basis=basis, restricted=True)

        hf = RHF(basis)
        hf.run(tol=tol, vocal=False)
        basis.change_basis(hf.C, inplace=True)
        basis.from_restricted(inplace=True)

        args = {"tol": tol}
        if not CC.__name__.startswith("Q"):
            args["include_l"] = True

        cc = CC(basis).run(**args)
        u = np.array([1, 0, 0])
        F_str = 1e-2
        omega = 0.2
        tprime = 2 * np.pi / omega
        dt = 0.1

        tdcc = TimeDependentCoupledCluster(cc, time=(0, 1, dt), integrator_args={"dt": dt})
        tdcc.external_one_body = Sin2(u, F_str, omega, tprime)
        tdcc.sampler = DipoleSampler()

        tdcc.run()

        results = tdcc.results
        energy, r_x = results["energy"], results["r"][:, 0]

        increasing_r_x = np.all(np.diff(r_x.real) >= 0)
        decreasing_energy = energy[0].real > energy[-1].real

        self.assertTrue(increasing_r_x, msg="<x> is not increasing under Sin^2 pulse")
        self.assertTrue(decreasing_energy, msg="<H> is not decreasing under Sin^2 pulse")

    def no_pulse(self, atom, basis, CC, tol=1e-8):
        basis = PyscfBasis(atom=atom, basis=basis, restricted=True)

        hf = RHF(basis)
        hf.run(tol=tol, vocal=False)
        basis.change_basis(hf.C, inplace=True)
        basis.from_restricted(inplace=True)

        args = {"tol": tol}
        if not CC.__name__.startswith("Q"):
            args["include_l"] = True

        cc = CC(basis).run(**args)
        u = np.array([1, 0, 0])
        dt = 0.1

        tdcc = TimeDependentCoupledCluster(cc, time=(0, 1, dt), integrator_args={"dt": dt})
        tdcc.sampler = DipoleSampler()

        tdcc.run()

        results = tdcc.results
        E, r = results["energy"], results["r"]

        isconst_E = np.all(np.diff(E.real) <= 1e-12)
        isconst_r = np.all(np.diff(r.real) <= 1e-12)

        self.assertTrue(isconst_E)
        self.assertTrue(isconst_r)

    def test_ccd_He(self):
        self.sin2_pulse(atom="He 0 0 0", basis="cc-pVDZ", CC=GCCD)
        self.no_pulse(atom="He 0 0 0", basis="cc-pVDZ", CC=GCCD)

    def test_ccsd_He(self):
        self.sin2_pulse(atom="He 0 0 0", basis="cc-pVDZ", CC=GCCSD)
        self.no_pulse(atom="He 0 0 0", basis="cc-pVDZ", CC=GCCSD)

    def test_qccd_He(self):
        self.sin2_pulse(atom="He 0 0 0", basis="cc-pVDZ", CC=QCCD)
        self.no_pulse(atom="He 0 0 0", basis="cc-pVDZ", CC=QCCD)

    def test_qccsd_He(self):
        self.sin2_pulse(atom="He 0 0 0", basis="cc-pVDZ", CC=QCCSD)
        self.no_pulse(atom="He 0 0 0", basis="cc-pVDZ", CC=QCCSD)
