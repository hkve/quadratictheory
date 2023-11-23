import numpy as np
from unittest import TestCase
from clusterfock.hf import RHF
from clusterfock.cc import GCCD, GCCSD
from clusterfock.basis import PyscfBasis


class TestCoupledClusterDensities(TestCase):
    def compare_raw_vs_intermediate(self, atom, basis, CC, tol=1e-8):
        rbasis = PyscfBasis(atom=atom, basis=basis, restricted=True)

        hf = RHF(rbasis)
        hf.run(tol=tol, vocal=False)
        rbasis.change_basis(hf.C, inplace=True)
        gbasis = rbasis.from_restricted(inplace=False)

        raw = CC(gbasis, intermediates=False).run(tol=tol, include_l=True)
        inter = CC(gbasis, intermediates=True).run(tol=tol, include_l=True)

        raw.densities()
        inter.densities()

        raw_rho = raw.rho_ob
        inter_rho = inter.rho_ob

        N = gbasis.N
        self.assertAlmostEqual(np.trace(raw_rho), N, places=6)
        self.assertAlmostEqual(np.trace(inter_rho), N, places=6)

        diff = np.abs(raw_rho - inter_rho).ravel()
        all_close = np.all(diff < 1e-6)
        self.assertTrue(
            all_close,
            msg=f"Difference between intermediate and raw results for densities max(diff) {diff.max()} > 1e-6",
        )

        raw_rho = raw.rho_tb
        inter_rho = inter.rho_tb

        self.assertAlmostEqual(np.einsum("pqpq->", raw_rho), N * (N - 1), places=6)
        self.assertAlmostEqual(np.einsum("pqpq->", inter_rho), N * (N - 1), places=6)

        diff = np.abs(raw_rho - inter_rho).ravel()
        all_close = np.all(diff < 1e-6)
        self.assertTrue(
            all_close,
            msg=f"Difference between intermediate and raw results for densities max(diff) {diff.max()} > 1e-6",
        )

    def energy_expval(self, atom, basis, CC, tol=1e-8):
        rbasis = PyscfBasis(atom=atom, basis=basis, restricted=True)

        hf = RHF(rbasis)
        hf.run(tol=tol, vocal=False)
        rbasis.change_basis(hf.C, inplace=True)
        gbasis = rbasis.from_restricted(inplace=False)

        ccd = CC(gbasis, intermediates=True).run(tol=tol, include_l=True)
        ccd.densities()

        # Energy using amplitudes
        E1 = ccd.energy() - gbasis._energy_shift

        # Energy using densities
        E2 = np.trace(ccd.rho_ob @ gbasis.h)
        E2 += 0.25 * np.einsum("pqrs,pqrs->", ccd.rho_tb, gbasis.u)

        self.assertAlmostEqual(
            E1,
            E2,
            places=6,
            msg=f"Energy from amplitudes {E1:.6f} does not match energy from density matricies {E2:.6f}",
        )

    def zero_position(self, atom, basis, CC, tol=1e-6):
        basis = PyscfBasis(atom, basis, restricted=True, center=False)
        hf = RHF(basis).run()
        basis.change_basis(hf.C)
        basis.from_restricted()

        cc = CC(basis).run(tol=tol, include_l=True)
        cc.one_body_density()

        r = cc.one_body_expval(basis.r)

        for i in range(3):
            self.assertAlmostEqual(r[i], 0)

    def test_ccd_He(self):
        self.compare_raw_vs_intermediate(atom="He 0 0 0", basis="cc-pVDZ", CC=GCCD)
        self.energy_expval(atom="He 0 0 0", basis="cc-pVDZ", CC=GCCD)

        self.zero_position(atom="He 0 0 0", basis="cc-pVDZ", CC=GCCD)
        self.zero_position(atom="He 0 0 0", basis="cc-pVDZ", CC=GCCSD)

    def test_ccd_Be(self):
        self.compare_raw_vs_intermediate(atom="Be 0 0 0", basis="cc-pVDZ", CC=GCCD)
        self.energy_expval(atom="Be 0 0 0", basis="cc-pVDZ", CC=GCCD)

    def test_ccd_LiH(self):
        self.compare_raw_vs_intermediate(atom="Li 0 0 0; H 0 0 1.2", basis="cc-pVDZ", CC=GCCD)
        self.energy_expval(atom="Li 0 0 0; H 0 0 1.2", basis="cc-pVDZ", CC=GCCD)

    def test_ccsd_He(self):
        self.compare_raw_vs_intermediate(atom="He 0 0 0", basis="cc-pVDZ", CC=GCCSD)
        self.energy_expval(atom="He 0 0 0", basis="cc-pVDZ", CC=GCCSD)

    def test_ccsd_Be(self):
        self.compare_raw_vs_intermediate(atom="Be 0 0 0", basis="cc-pVDZ", CC=GCCSD)
        self.energy_expval(atom="Be 0 0 0", basis="cc-pVDZ", CC=GCCSD)

    def test_ccsd_LiH(self):
        self.compare_raw_vs_intermediate(atom="Li 0 0 0; H 0 0 1.2", basis="cc-pVDZ", CC=GCCSD)
        self.energy_expval(atom="Li 0 0 0; H 0 0 1.2", basis="cc-pVDZ", CC=GCCSD)
