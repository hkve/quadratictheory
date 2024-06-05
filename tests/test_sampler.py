from quadratictheory import PyscfBasis
from quadratictheory.td.sampler import Sampler, DipoleSampler, OverlapSampler
from quadratictheory import CCD, TimeDependentCoupledCluster
from unittest import TestCase


class TestMixer(TestCase):
    def test_empty_sampler(self):
        basis = PyscfBasis(atom="He 0 0 0", basis="sto-3g", restricted=False).pyscf_hartree_fock()
        cc = CCD(basis).run(tol=1e-8, include_l=True)
        tdcc = TimeDependentCoupledCluster(cc)
        sampler = Sampler()

        self.assertTrue(sampler.setup_sampler(basis, tdcc) == [])
        self.assertTrue(sampler.one_body(basis) == {})
        self.assertTrue(sampler.two_body(basis) == {})
        self.assertTrue(sampler.misc(tdcc) == {})

    def test_dipole_sampler(self):
        basis = PyscfBasis(atom="He 0 0 0", basis="sto-3g", restricted=False).pyscf_hartree_fock()
        cc = CCD(basis).run(tol=1e-8, include_l=True)
        tdcc = TimeDependentCoupledCluster(cc)

        sampler = DipoleSampler()

        tdcc.sampler = sampler
        tdcc._setup_sample(basis)

        tdcc._sample()
        tdcc._sample()
        tdcc._sample()

        results = tdcc.results

        self.assertTrue(len(results["r"]) == 3)
        self.assertTrue(len(results["delta_rho1"]) == 3)
        self.assertTrue(len(results["energy"]) == 3)

        self.assertFalse("t" in list(results.keys()))
        self.assertFalse("_t0" in list(tdcc.__dict__.keys()))
        self.assertFalse("_l0" in list(tdcc.__dict__.keys()))

    def test_overlap_sampler(self):
        basis = PyscfBasis(atom="He 0 0 0", basis="sto-3g", restricted=False).pyscf_hartree_fock()
        cc = CCD(basis).run(tol=1e-8, include_l=True)
        tdcc = TimeDependentCoupledCluster(cc)

        sampler = OverlapSampler()

        tdcc.sampler = sampler
        tdcc._setup_sample(basis)

        self.assertTrue("_t0" in list(tdcc.__dict__.keys()))
        self.assertTrue("_l0" in list(tdcc.__dict__.keys()))

        tdcc._sample()

        results = tdcc.results
        overlap = results["overlap"]

        self.assertTrue(sampler.has_overlap)
        self.assertTrue(len(overlap) == 1)
        self.assertAlmostEqual(overlap[0], 1.0)
