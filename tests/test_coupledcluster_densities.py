import numpy as np
from unittest import TestCase
from clusterfock.hf import RHF
from clusterfock.cc import GCCD
from clusterfock.basis import PyscfBasis

class TestCoupledClusterDensities(TestCase):
    def compare_raw_vs_intermediate(self, atom, basis, tol=1e-8):
        rbasis = PyscfBasis(atom=atom, basis=basis, restricted=True)

        hf = RHF(rbasis)
        hf.run(tol=tol, vocal=False)
        rbasis.change_basis(hf.C, inplace=True)
        gbasis = rbasis.from_restricted(inplace=False)

        raw_ccd = GCCD(gbasis, intermediates=False).run(tol=tol, include_l=True)
        inter_ccd = GCCD(gbasis, intermediates=True).run(tol=tol, include_l=True)

        raw_ccd.densities()
        inter_ccd.densities()

        raw_rho = raw_ccd.rho_ob
        inter_rho = inter_ccd.rho_ob

        N = gbasis.N
        self.assertAlmostEqual(np.trace(raw_rho), N, places=6)
        self.assertAlmostEqual(np.trace(inter_rho), N, places=6)

        diff = np.abs(raw_rho - inter_rho).ravel()
        all_close = np.all(diff < 1e-6)
        self.assertTrue(all_close, msg=f"Difference between intermediate and raw results for densities max(diff) {diff.max()} > 1e-6")

        raw_rho = raw_ccd.rho_tb
        inter_rho = inter_ccd.rho_tb

        diff = np.abs(raw_rho - inter_rho).ravel()
        all_close = np.all(diff < 1e-6)
        self.assertTrue(all_close, msg=f"Difference between intermediate and raw results for densities max(diff) {diff.max()} > 1e-6")

    def test_ccd_He(self):
        self.compare_raw_vs_intermediate(atom="He 0 0 0", basis="cc-pVDZ")

    def test_ccd_Be(self):
        self.compare_raw_vs_intermediate(atom="Be 0 0 0", basis="cc-pVDZ")

    def test_ccd_LiH(self):
        self.compare_raw_vs_intermediate(atom="Li 0 0 0; H 0 0 1.2", basis="cc-pVDZ")