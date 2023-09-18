import numpy as np
from unittest import TestCase

from mypack.basis import TestBasis


class TestBasisProperties(TestCase):
    def setUp(self) -> None:
        self.gbasis = TestBasis(L=10, N=4, restricted=False).setup()
        self.rbasis = TestBasis(L=10, N=4, restricted=True).setup()

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

    def test_add_spin(self):
        basis = self.rbasis.copy()

        old_diag = np.diag(basis.h)

        basis._add_spin()
        new_diag = np.diag(basis.h)

        self.assertEqual(2 * old_diag.sum(), new_diag.sum())

    def test_from_restricted(self):
        basis = self.rbasis

        gbasis = basis.from_restricted(inplace=False)

        self.assertTrue(np.allclose(gbasis.u, -gbasis.u.transpose(1, 0, 3, 2)))
        self.assertEqual(2 * basis.L, gbasis.L)
        self.assertEqual(2 * basis.N, gbasis.N)
