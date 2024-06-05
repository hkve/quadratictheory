import numpy as np
from unittest import TestCase
from quadratictheory.hf import RHF
from quadratictheory.cc import QCCSD, QCCSD_T1
from quadratictheory.basis import PyscfBasis


class TestCoupledCluster(TestCase):
    def compare_with_general(self, atom, basis, CC, CC_T1, restricted=False, tol=1e-8):
        b = PyscfBasis(atom=atom, basis=basis, restricted=False).pyscf_hartree_fock()

        cc = CC(b).run(tol=tol)
        energy_cc = cc.energy()

        t1cc = CC_T1(b, copy=True).run(tol=tol, maxiters=100)
        energy_t1cc = t1cc.energy()

        # # Check that energies are equal.
        self.assertAlmostEqual(energy_cc, energy_t1cc, places=6)

    def test_He(self):
        self.compare_with_general(atom="He 0 0 0", basis="cc-pVDZ", CC=QCCSD, CC_T1=QCCSD_T1, restricted=False)

    def testl_Be(self):
        self.compare_with_general(atom="Be 0 0 0", basis="6-31G", CC=QCCSD, CC_T1=QCCSD_T1, restricted=False)

    def testl_LiH(self):
        self.compare_with_general(atom="Li 0 0 0; H 0 0 2.0", basis="DZ", CC=QCCSD, CC_T1=QCCSD_T1, restricted=False)
