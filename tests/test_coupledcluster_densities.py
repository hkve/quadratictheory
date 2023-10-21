import numpy as np
from unittest import TestCase
from clusterfock.hf import RHF
from clusterfock.cc import GCCD
from clusterfock.basis import PyscfBasis

class TestCoupledClusterDensities(TestCase):
    def ccd_compare_raw_vs_intermediate(self, atom, basis, tol=1e-8):
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

        self.assertAlmostEqual(np.einsum("pqpq->", raw_rho), N*(N-1), places=6)
        self.assertAlmostEqual(np.einsum("pqpq->", inter_rho), N*(N-1), places=6)

        diff = np.abs(raw_rho - inter_rho).ravel()
        all_close = np.all(diff < 1e-6)
        self.assertTrue(all_close, msg=f"Difference between intermediate and raw results for densities max(diff) {diff.max()} > 1e-6")

    def ccd_test_energy_expval(self, atom, basis, tol=1e-8):
        rbasis = PyscfBasis(atom=atom, basis=basis, restricted=True)

        hf = RHF(rbasis)
        hf.run(tol=tol, vocal=False)
        rbasis.change_basis(hf.C, inplace=True)
        gbasis = rbasis.from_restricted(inplace=False)

        ccd = GCCD(gbasis, intermediates=True).run(tol=tol, include_l=True)
        ccd.densities()

        # Energy using amplitudes
        E1 = ccd.energy() - gbasis._energy_shift
        
        # Energy using densities
        E2 = np.trace(ccd.rho_ob @ gbasis.h)
        E2 += 0.25*np.einsum("pqrs,pqrs->", ccd.rho_tb, gbasis.u)

        self.assertAlmostEqual(E1, E2, places=6, msg=f"Energy from amplitudes {E1:.6f} does not match energy from density matricies {E2:.6f}")

    def test_ccd_He(self):
        self.ccd_compare_raw_vs_intermediate(atom="He 0 0 0", basis="cc-pVDZ")
        self.ccd_test_energy_expval(atom="He 0 0 0", basis="cc-pVDZ")

    def test_ccd_Be(self):
        self.ccd_compare_raw_vs_intermediate(atom="Be 0 0 0", basis="cc-pVDZ")
        self.ccd_test_energy_expval(atom="Be 0 0 0", basis="cc-pVDZ")

    def test_ccd_LiH(self):
        self.ccd_compare_raw_vs_intermediate(atom="Li 0 0 0; H 0 0 1.2", basis="cc-pVDZ")
        self.ccd_test_energy_expval(atom="Li 0 0 0; H 0 0 1.2", basis="cc-pVDZ")