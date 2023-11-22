import numpy as np
from unittest import TestCase
from clusterfock.hf import HF
from clusterfock.basis import PyscfBasis
import pyscf


class TestHartreeFock(TestCase):
    def is_diagonal(self, f):
        all_elms = np.sum(f**2)
        diag_elms = np.sum(np.diag(f**2).sum())

        return abs(diag_elms - all_elms) < 1e-7

    def test_diagonal_fock_matrix(self):
        rbasis = PyscfBasis(atom="He 0 0 0", basis="cc-pVDZ")
        rbasis.calculate_fock_matrix()

        tol = 1e-8
        rhf = HF(rbasis).run(tol=tol)

        gbasis = rbasis.from_restricted(inplace=False)
        ghf = HF(gbasis).run(tol=tol)

        rbasis.change_basis(rhf.C)
        gbasis.change_basis(ghf.C)

        self.assertTrue(self.is_diagonal(rbasis.f))
        self.assertTrue(self.is_diagonal(gbasis.f))

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

    def zero_posistion(self, atom, basis, restricted):
        basis = PyscfBasis(atom=atom, basis=basis, restricted=restricted)

        hf = HF(basis).run()

        r = hf.one_body_expval(basis.r)

        for i in range(3):
            self.assertAlmostEqual(r[i], 0)

    def translate_posistion(self, atom, atom_translated, translation, basis, restricted):
        first = PyscfBasis(atom=atom, basis=basis, restricted=restricted)
        hf = HF(first).run()
        r = hf.one_body_expval(first.r)

        second = PyscfBasis(atom=atom_translated, basis=basis, restricted=restricted)
        hf = HF(second).run()
        r_translated = hf.one_body_expval(second.r)

        translation_bohr = [1.8897259886*t for t in translation]
        for i in range(3):
            self.assertAlmostEqual(r_translated[i], r[i]+translation_bohr[i])

    # Atoms
    def test_He(self):
        self.compare_with_pyscf(atom="He 0 0 0", basis="cc-pVDZ")
        self.zero_posistion(atom="He 0 0 0", basis="cc-pVDZ", restricted=True)
        self.zero_posistion(atom="He 0 0 0", basis="cc-pVDZ", restricted=False)
        # self.translate_posistion(atom="He 0 0 0", atom_translated="He 0 0 1", translation=[0,0,1], basis="cc-pVDZ", restricted=False)

    def test_Be(self):
        self.compare_with_pyscf(atom="Be 0 0 0", basis="cc-pVDZ")
        self.zero_posistion(atom="Be 0 0 0", basis="cc-pVDZ", restricted=True)
        self.zero_posistion(atom="Be 0 0 0", basis="cc-pVDZ", restricted=False)


    def test_Ne(self):
        self.compare_with_pyscf(atom="Ne 0 0 0", basis="cc-pVDZ")
        self.zero_posistion(atom="Ne 0 0 0", basis="cc-pVDZ", restricted=True)
        self.zero_posistion(atom="Ne 0 0 0", basis="cc-pVDZ", restricted=False)

    # Molecules
    def test_H2(self):
        self.compare_with_pyscf(atom="H 0 0 0; H 0 0 1.28", basis="sto-3g")

    def test_LiH(self):
        self.compare_with_pyscf(atom="Li 0 0 0; H 0 0 1.619", basis="cc-pVDZ")
        self.compare_with_pyscf(atom="Li 0 0 0; H 0 0 1.511", basis="sto-3g")
