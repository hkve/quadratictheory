import numpy as np
from unittest import TestCase
from quadratictheory.hf import RHF
from quadratictheory.cc import GCCD, RCCD, GCCSD, RCCSD
from quadratictheory.basis import PyscfBasis

import pyscf
from pyscf.cc.ccd import CCD as pyscfCCD
from pyscf.cc.ccsd import CCSD as pyscfCCSD


class TestCoupledCluster(TestCase):
    def ccd_compare_with_pyscf(self, atom, basis, tol=1e-8):
        rbasis = PyscfBasis(atom=atom, basis=basis, restricted=True)

        hf = RHF(rbasis)
        hf.run(tol=tol, vocal=False)
        rbasis.change_basis(hf.C, inplace=True)
        gbasis = rbasis.from_restricted(inplace=False)

        rccd = RCCD(rbasis).run(tol=tol, include_l=False)
        Erccd = rccd.energy()

        gccd = GCCD(gbasis).run(tol=tol, include_l=False)
        Egccd = gccd.energy()

        hf_pyscf = pyscf.scf.HF(rbasis.mol).run(verbose=0, tol=tol)
        ccd_pyscf = pyscfCCD(hf_pyscf).run(verbose=0, tol=tol)

        Eccd = ccd_pyscf.e_tot

        self.assertAlmostEqual(Erccd, Eccd, places=6)
        self.assertAlmostEqual(Egccd, Eccd, places=6)

    def ccsd_compare_with_pyscf(self, atom, basis, tol=1e-8):
        rbasis = PyscfBasis(atom=atom, basis=basis, restricted=True)

        hf = RHF(rbasis)
        hf.run(tol=tol, vocal=False)
        rbasis.change_basis(hf.C, inplace=True)

        gbasis = rbasis.from_restricted(inplace=True)

        gccsd = GCCSD(gbasis).run(tol=tol, include_l=False)
        Egccsd = gccsd.energy()

        hf_pyscf = pyscf.scf.HF(gbasis.mol).run(verbose=0, tol=tol)
        ccsd_pyscf = pyscfCCSD(hf_pyscf).run(verbose=0, tol=tol)

        Eccsd = ccsd_pyscf.e_tot

        self.assertAlmostEqual(Egccsd, Eccsd, places=6)

    def general_symmetry(self, amplitude):
        permuted_hole = -amplitude.transpose(0, 1, 3, 2)
        permuted_particle = -amplitude.transpose(1, 0, 2, 3)
        permuted_both = amplitude.transpose(1, 0, 3, 2)

        check_hole = np.linalg.norm(amplitude - permuted_hole)
        check_particle = np.linalg.norm(amplitude - permuted_particle)
        check_both = np.linalg.norm(amplitude - permuted_both)

        return check_hole, check_particle, check_both

    def restricted_symmetry(self, amplitude):
        permuted_both = amplitude.transpose(1, 0, 3, 2)

        return np.linalg.norm(amplitude - permuted_both)

    def compare_general_with_restricted(self, atom, basis, CC, RCC, tol=1e-8):
        rbasis = PyscfBasis(atom=atom, basis=basis, restricted=True).pyscf_hartree_fock()

        rcc = RCC(rbasis).run(tol=tol, include_l=True)
        e_rcc = rcc.energy()
        td_e_rcc = rcc._evaluate_tdcc_energy()
        t_sym_rcc = self.restricted_symmetry(rcc._t[2])
        l_sym_rcc = self.restricted_symmetry(rcc._l[2])

        rbasis.from_restricted()
        cc = CC(rbasis).run(tol=tol, include_l=True)
        e_cc = cc.energy()
        td_e_cc = cc._evaluate_tdcc_energy()
        t_sym_cc_hole, t_sym_cc_particle, t_sym_cc_both = self.general_symmetry(cc._t[2])
        l_sym_cc_hole, l_sym_cc_particle, l_sym_cc_both = self.general_symmetry(cc._l[2])

        # Check that energies are equal
        self.assertAlmostEqual(e_rcc, e_cc, places=8)

        # Check if <0|(1+L)e^-T H e^T|0> is 0
        self.assertAlmostEqual(td_e_cc, 0, places=8)
        self.assertAlmostEqual(td_e_rcc, 0, places=8)

        # Check rcc amplitude permutations
        self.assertAlmostEqual(t_sym_rcc, 0, places=8)
        self.assertAlmostEqual(l_sym_rcc, 0, places=8)

        # Check cc amplitude permutations
        self.assertAlmostEqual(t_sym_cc_hole, 0, places=8)
        self.assertAlmostEqual(t_sym_cc_particle, 0, places=8)
        self.assertAlmostEqual(t_sym_cc_both, 0, places=8)

        self.assertAlmostEqual(l_sym_cc_hole, 0, places=8)
        self.assertAlmostEqual(l_sym_cc_particle, 0, places=8)
        self.assertAlmostEqual(l_sym_cc_both, 0, places=8)

    def test_ccd_He(self):
        self.ccd_compare_with_pyscf(atom="He 0 0 0", basis="cc-pVDZ")
        self.compare_general_with_restricted(atom="He 0 0 0", basis="cc-pVDZ", CC=GCCD, RCC=RCCD)

    def test_ccd_Be(self):
        self.ccd_compare_with_pyscf(atom="Be 0 0 0", basis="cc-pVDZ")
        self.compare_general_with_restricted(atom="Be 0 0 0", basis="cc-pVDZ", CC=GCCD, RCC=RCCD)

    def test_ccsd_He(self):
        self.ccsd_compare_with_pyscf(atom="He 0 0 0", basis="cc-pVDZ")
        self.compare_general_with_restricted(atom="He 0 0 0", basis="cc-pVDZ", CC=GCCSD, RCC=RCCSD)

    def test_ccd_Be(self):
        self.ccsd_compare_with_pyscf(atom="Be 0 0 0", basis="cc-pVDZ")
        self.compare_general_with_restricted(atom="Be 0 0 0", basis="cc-pVDZ", CC=GCCSD, RCC=RCCSD)

    # def ccd_compare_with_cccbdb(self, atom, basis, cccbdb_energy):
    #     rbasis = PyscfBasis(atom=atom, basis=basis, restricted=True)

    #     hf = RHF(rbasis)
    #     hf.run(tol=1e-8, vocal=False)
    #     rbasis.change_basis(hf.C, inplace=True)
    #     gbasis = rbasis.from_restricted(inplace=False)

    #     rccd = RCCD(rbasis).run(tol=1e-8)
    #     Erccd = rccd.energy()

    #     gccd = GCCD(gbasis).run(tol=1e-8)
    #     Egccd = gccd.energy()

    #     self.assertAlmostEqual(Erccd, cccbdb_energy, places=6)
    #     self.assertAlmostEqual(Egccd, cccbdb_energy, places=6)

    # def ccsd_compare_with_cccbdb(self, atom, basis, cccbdb_energy):
    #     rbasis = PyscfBasis(atom=atom, basis=basis, restricted=True)
    #     hf = RHF(rbasis).run(tol=1e-8, vocal=False)
    #     rbasis.change_basis(hf.C, inplace=True)
    #     gbasis = rbasis.from_restricted(inplace=False)

    #     gbasis.calculate_fock_matrix()
    #     gccsd = GCCSD(gbasis, intermediates=True).run(tol=1e-8)
    #     Egccsd = gccsd.energy()

    #     self.assertAlmostEqual(Egccsd, cccbdb_energy, places=6)

    # def test_ccd_He(self):
    #     self.ccd_compare_with_cccbdb(atom="He 0 0 0", basis="cc-pVDZ", cccbdb_energy=-2.887592)

    # def test_ccsd_He(self):
    #     self.ccsd_compare_with_cccbdb(atom="He 0 0 0", basis="cc-pVDZ", cccbdb_energy=-2.887595)

    # def test_ccd_Be(self):
    #     self.ccd_compare_with_cccbdb(atom="Be 0 0 0", basis="cc-pVDZ", cccbdb_energy=-14.616422)

    # def test_ccsd_Be(self):
    #     self.ccsd_compare_with_cccbdb(atom="Be 0 0 0", basis="cc-pVDZ", cccbdb_energy=-14.616843)

    # def test_ccd_LiH(self):
    #     self.ccd_compare_with_cccbdb(
    #         atom="Li 0 0 0; H 0 0 1.6167", basis="cc-pVDZ", cccbdb_energy=-8.014079
    #     )

    # def test_ccsd_LiH(self):
    #     self.ccsd_compare_with_cccbdb(
    #         atom="Li 0 0 0; H 0 0 1.6191", basis="cc-pVDZ", cccbdb_energy=-8.014421
    #     )
