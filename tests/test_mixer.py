from clusterfock.basis import PyscfBasis
from clusterfock.mix import RelaxedMixer, DIISMixer
from clusterfock.hf import HF
from unittest import TestCase


class TestMixer(TestCase):
    def test_alpha_with_hf(self):
        b = PyscfBasis(atom="He 0 0 0", basis="cc-pVDZ")
        hf = HF(b)
        hf.mixer = RelaxedMixer(alpha=0)
        hf.run(tol=1e-8)

        E_nomix, iter_nomix = hf.energy(), hf.iters

        hf = HF(b)
        hf.mixer = RelaxedMixer(alpha=0.5)
        hf.run(tol=1e-8)

        self.assertAlmostEqual(E_nomix, hf.energy())
        self.assertTrue(iter_nomix < hf.iters)

    def test_DIIS_with_hf(self):
        b = PyscfBasis(atom="Ne 0 0 0", basis="cc-pVDZ")
        hf = HF(b)
        hf.mixer = RelaxedMixer(alpha=0)
        hf.run(tol=1e-8)
        E_nomix, iter_nomix = hf.energy(), hf.iters

        hf = HF(b)
        hf.mixer = DIISMixer(n_vectors=4)
        hf.run(tol=1e-8)

        self.assertAlmostEqual(E_nomix, hf.energy())
        self.assertTrue(iter_nomix > hf.iters)

    def test_alpha_mixer(self):
        mixer = RelaxedMixer(alpha=0.1)

        old, new = 1, 10
        expected = 0.1 + 9
        calculated = mixer(old, new)

        self.assertEqual(expected, calculated)
