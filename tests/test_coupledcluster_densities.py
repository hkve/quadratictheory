import numpy as np
from unittest import TestCase
from clusterfock.hf import RHF
from clusterfock.cc import GCCD, GCCSD, QCCD, QCCSD, RCCD, RCCSD
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

        if CC.__name__ in ["QCCD", "QCCSD"]:
            cc = CC(gbasis, intermediates=True).run(tol=tol)
        else:
            cc = CC(gbasis, intermediates=True).run(tol=tol, include_l=True)

        cc.densities()

        # Energy using amplitudes
        E1 = cc.energy() - gbasis._energy_shift

        # Energy using densities
        E2 = cc.one_body_expval(gbasis.h)
        E2 += 0.5 * cc.two_body_expval(gbasis.u)

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

    def compare_general_with_restricted(self, atom, basis, CC, RCC, tol=1e-8):
        basis = PyscfBasis(atom, basis, restricted=True).pyscf_hartree_fock()

        rcc = RCC(basis).run(tol=tol, include_l=True)
        h_rcc = rcc.one_body_expval(basis.h)
        # r_rcc = rcc.one_body_expval(basis.r)

        basis.from_restricted()
        cc = CC(basis).run(tol=tol, include_l=True)
        h_cc = cc.one_body_expval(basis.h)
        # r_cc = cc.one_body_expval(basis.r)

        self.assertAlmostEqual(h_rcc, h_cc, places=6)
        # self.assertAlmostEqual(r_rcc, r_cc, places=6)

    def test_ccd_He(self):
        self.compare_raw_vs_intermediate(atom="He 0 0 0", basis="cc-pVDZ", CC=GCCD)
        self.energy_expval(atom="He 0 0 0", basis="cc-pVDZ", CC=GCCD)
        self.compare_general_with_restricted(atom="He 0 0 0", basis="cc-pVDZ", CC=GCCD, RCC=RCCD)

        self.zero_position(atom="He 0 0 0", basis="cc-pVDZ", CC=GCCD)

    def test_ccd_Be(self):
        self.compare_raw_vs_intermediate(atom="Be 0 0 0", basis="cc-pVDZ", CC=GCCD)
        self.energy_expval(atom="Be 0 0 0", basis="cc-pVDZ", CC=GCCD)
        self.compare_general_with_restricted(atom="Be 0 0 0", basis="cc-pVDZ", CC=GCCD, RCC=RCCD)

    def test_ccd_LiH(self):
        self.compare_raw_vs_intermediate(atom="Li 0 0 0; H 0 0 2.26", basis="cc-pVDZ", CC=GCCD)
        self.energy_expval(atom="Li 0 0 0; H 0 0 2.26", basis="cc-pVDZ", CC=GCCD)
        self.compare_general_with_restricted(
            atom="Li 0 0 0; H 0 0 2.26", basis="cc-pVDZ", CC=GCCD, RCC=RCCD
        )

    def test_ccsd_He(self):
        self.compare_raw_vs_intermediate(atom="He 0 0 0", basis="cc-pVDZ", CC=GCCSD)
        self.energy_expval(atom="He 0 0 0", basis="cc-pVDZ", CC=GCCSD)
        self.zero_position(atom="He 0 0 0", basis="cc-pVDZ", CC=GCCSD)
        self.compare_general_with_restricted(atom="He 0 0 0", basis="cc-pVDZ", CC=GCCSD, RCC=RCCSD)

    def test_ccsd_Be(self):
        self.compare_raw_vs_intermediate(atom="Be 0 0 0", basis="cc-pVDZ", CC=GCCSD)
        self.energy_expval(atom="Be 0 0 0", basis="cc-pVDZ", CC=GCCSD)
        self.compare_general_with_restricted(atom="Be 0 0 0", basis="cc-pVDZ", CC=GCCSD, RCC=RCCSD)

    def test_ccsd_LiH(self):
        self.compare_raw_vs_intermediate(atom="Li 0 0 0; H 0 0 2.26", basis="cc-pVDZ", CC=GCCSD)
        self.energy_expval(atom="Li 0 0 0; H 0 0 2.26", basis="cc-pVDZ", CC=GCCSD)
        self.compare_general_with_restricted(
            atom="Li 0 0 0; H 0 0 2.26", basis="cc-pVDZ", CC=GCCSD, RCC=RCCSD
        )

    def test_qccd_He(self):
        self.energy_expval(atom="He 0 0 0", basis="cc-pVDZ", CC=QCCD)

    def test_qccd_Be(self):
        self.energy_expval(atom="Be 0 0 0", basis="cc-pVDZ", CC=QCCD)

    def test_qccd_Li(self):
        self.energy_expval(atom="Li 0 0 0; H 0 0 2.26", basis="cc-pVDZ", CC=QCCD)

    def test_qccsd_He(self):
        self.energy_expval(atom="He 0 0 0", basis="cc-pVDZ", CC=QCCSD)

    def test_qccsd_Be(self):
        self.energy_expval(atom="Be 0 0 0", basis="cc-pVDZ", CC=QCCSD)

    def test_qccsd_Li(self):
        self.energy_expval(atom="Li 0 0 0; H 0 0 2.26", basis="cc-pVDZ", CC=QCCSD)
