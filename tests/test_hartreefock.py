import numpy as np
from unittest import TestCase
from mypack.hf import HF
from mypack.basis import PyscfBasis
import pyscf


class TestHartreeFock(TestCase):
    def compare_with_pyscf(self, atom, basis):
        basis = PyscfBasis(atom=atom, basis=basis, restricted=True)

        tol = 1e-8
        hf1 = HF(basis).run(tol=tol)
        E_rhf = hf1.energy()

        basis.from_restricted()
        hf2 = HF(basis).run(tol=tol)
        E_hf = hf2.energy()

        hf3 = pyscf.scf.HF(basis.mol).run(verbose=0, tol=tol)
        E_pyscf = hf3.e_tot

        self.assertAlmostEqual(E_rhf, E_pyscf)
        self.assertAlmostEqual(E_hf, E_pyscf)

    # Atoms
    def test_He(self):
        self.compare_with_pyscf(atom="He 0 0 0", basis="cc-pVDZ")

    def test_Be(self):
        self.compare_with_pyscf(atom="Be 0 0 0", basis="cc-pVDZ")

    def test_Ne(self):
        self.compare_with_pyscf(atom="Ne 0 0 0", basis="cc-pVDZ")

    # Molecules
    def test_H2(self):
        self.compare_with_pyscf(atom="H 0 0 0; H 0 0 1.28", basis="sto-3g")

    def test_LiH(self):
        self.compare_with_pyscf(atom="Li 0 0 0; H 0 0 1.619", basis="cc-pVDZ")
        self.compare_with_pyscf(atom="Li 0 0 0; H 0 0 1.511", basis="sto-3g")
