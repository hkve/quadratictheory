import numpy as np
from unittest import TestCase

from quadratictheory.basis import TestBasis


class TestBasisProperties(TestCase):
    def setUp(self) -> None:
        self.gbasis = TestBasis(L=6, N=2, restricted=False).setup()
        self.rbasis = TestBasis(L=6, N=2, restricted=True).setup()

    def test_transform_copy(self):
        basis = self.gbasis.copy()
        L = basis.L

        # Make orthogonal transformation
        H = np.random.rand(L, L)
        u, s, vh = np.linalg.svd(H, full_matrices=False)
        C = u @ vh

        h_old, s_old = basis.h.copy(), basis.s.copy()

        new_basis = basis.change_basis(C, inplace=False)

        self.assertTrue(np.array_equal(h_old, basis.h))
        self.assertTrue(np.array_equal(s_old, basis.s))

        new_basis.change_basis(C, inplace=True, inverse=True)

        self.assertTrue(np.allclose(new_basis.h, basis.h))
        self.assertTrue(np.allclose(new_basis.s, basis.s))

    def test_ints(self):
        rbasis = self.rbasis.copy()

        self.assertEqual(rbasis.L, self.rbasis.L)
        self.assertEqual(rbasis.N, self.rbasis.N)
        self.assertEqual(rbasis.M, self.rbasis.M)

        rbasis.N = 2 * rbasis.N + 2

        self.assertEqual(rbasis.L, self.rbasis.L)
        self.assertEqual(rbasis.N, 1 + 1)
        self.assertEqual(rbasis.M, 2 - 1)

    def test_add_spin(self):
        basis = self.rbasis.copy()

        old_diag = np.diag(basis.h)

        basis._add_spin()
        new_diag = np.diag(basis.h)

        self.assertAlmostEqual(2 * old_diag.sum(), new_diag.sum())

    def test_from_restricted(self):
        basis = self.rbasis

        self.assertTrue(np.allclose(basis.u, basis.u.transpose(1, 0, 3, 2)))

        gbasis = basis.from_restricted(inplace=False)

        self.assertTrue(np.allclose(gbasis.u, -gbasis.u.transpose(0, 1, 3, 2)))
        self.assertTrue(np.allclose(gbasis.u, gbasis.u.transpose(1, 0, 3, 2)))
        self.assertEqual(2 * basis.L, gbasis.L)
        self.assertEqual(2 * basis.N, gbasis.N)

    def test_basis_change(self):
        basis_core = [self.rbasis, self.gbasis]

        for b in basis_core:
            basis = b.copy()
            C = np.random.uniform(low=-1, high=1, size=(basis.L, basis.L))
            h, u, s = basis.h.copy(), basis.u.copy(), basis.s.copy()

            self.assertTrue(basis._computational_basis)

            basis.change_basis(C)

            self.assertFalse(basis._computational_basis)

            basis.change_basis(C, inverse=True)

            self.assertTrue(basis._computational_basis)
            self.assertTrue(np.allclose(h, basis.h, atol=1e-6))
            self.assertTrue(np.allclose(u, basis.u, atol=1e-6))
            self.assertTrue(np.allclose(s, basis.s, atol=1e-6))
