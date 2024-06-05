import numpy as np
from unittest import TestCase
from quadratictheory.hf import RHF
from quadratictheory.cc import GCCSD, GCCSD_T1
from quadratictheory.cc import QCCSD, QCCSD_T1
from quadratictheory.basis import PyscfBasis


class TestCoupledCluster(TestCase):
    def compare_with_general(self, atom, basis, CC, CC_T1, restricted=False, tol=1e-8):
        b = PyscfBasis(atom=atom, basis=basis, restricted=False).pyscf_hartree_fock()

        cc = CC(b).run(tol=tol, include_l=True)
        energy_cc = cc.energy()
        r_cc = cc.one_body_expval(b.r)

        t1cc = CC_T1(b, copy=True).run(tol=tol, include_l=True)
        b_t1 = t1cc.basis
        energy_t1cc = t1cc.energy()
        energy_from_density_t1cc = t1cc.one_body_expval(b_t1.h) + 0.5*t1cc.two_body_expval(b_t1.u) + b_t1._energy_shift
        energy_td_t1cc = t1cc._evaluate_tdcc_energy()
        r_t1cc = t1cc.one_body_expval(b_t1.r)

        # # Check that energies are equal.
        self.assertAlmostEqual(energy_cc, energy_t1cc, places=8)
        self.assertAlmostEqual(energy_t1cc, energy_from_density_t1cc, places=8)

        # # Check that td part is close to zero
        self.assertAlmostEqual(energy_td_t1cc, 0, places=8)

        for i in range(3):
            self.assertAlmostEqual(r_cc[i], r_t1cc[i], places=8)

    def test_He(self):
        self.compare_with_general(atom="He 0 0 0", basis="cc-pVDZ", CC=GCCSD, CC_T1=GCCSD_T1, restricted=False)

    def testl_Be(self):
        self.compare_with_general(atom="Be 0 0 0", basis="cc-pVDZ", CC=GCCSD, CC_T1=GCCSD_T1, restricted=False)

    def testl_LiH(self):
        self.compare_with_general(atom="Li 0 0 0; H 0 0 2.0", basis="cc-pVDZ", CC=GCCSD, CC_T1=GCCSD_T1, restricted=False)
