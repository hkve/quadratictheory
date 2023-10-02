import numpy as np
from unittest import TestCase
from clusterfock.hf import RHF
from clusterfock.cc import GCCD, RCCD, GCCSD
from clusterfock.basis import PyscfBasis

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

        rccd = RCCD(rbasis).run(tol=tol)
        Erccd = rccd.energy()

        gccd = GCCD(gbasis).run(tol=tol)
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

        gccsd = GCCSD(gbasis).run(tol=tol)
        Egccsd = gccsd.energy()

        hf_pyscf = pyscf.scf.HF(gbasis.mol).run(verbose=0, tol=tol)
        ccsd_pyscf = pyscfCCSD(hf_pyscf).run(verbose=0, tol=tol)
        
        Eccsd = ccsd_pyscf.e_tot

        self.assertAlmostEqual(Egccsd, Eccsd, places=6)



    def test_ccd_He(self):
        self.ccd_compare_with_pyscf(atom="He 0 0 0", basis="cc-pVDZ")

    def test_ccd_Be(self):
        self.ccd_compare_with_pyscf(atom="Be 0 0 0", basis="cc-pVDZ")

    def test_ccsd_He(self):
        self.ccsd_compare_with_pyscf(atom="He 0 0 0", basis="cc-pVDZ")

    def test_ccd_Be(self):
        self.ccsd_compare_with_pyscf(atom="Be 0 0 0", basis="cc-pVDZ")

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
