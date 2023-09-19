from mypack.basis import Basis
import numpy as np


class Lipkin(Basis):
    def __init__(
        self, N: int, eps: float = 1.0, V: float = 1.0, W: float = 1.0, dtype: type = float
    ):
        super().__init__(L=2 * N, N=N, restricted=False, dtype=dtype)
        self.eps = eps
        self.V = V
        self.W = W

        self.setup()

    def setup(self):
        self.u = np.zeros(self._two_body_shape)

        self.h = self._get_H0(self.L, self.N)
        if self.V is not None:
            self.u += self._get_HV(self.L, self.N)
        if self.W is not None:
            self.u += self._get_HW(self.L, self.N)

    def _get_H0(self, L: int, N: int) -> np.ndarray:
        H0_lower = -self.eps / 2 * np.ones(N)
        H0_upper = self.eps / 2 * np.ones(N)

        return np.diag(np.r_[H0_lower, H0_upper])

    def _get_HV(self, L: int, N: int) -> np.ndarray:
        spin_equal = lambda i, j: i // N == j // N
        pos_equal = lambda i, j: i - N == j or i == j - N

        u = np.zeros_like(self.u)

        for g in range(L):
            for d in range(L):
                for a in range(L):
                    for b in range(L):
                        dspin_bra = spin_equal(g, d)
                        dspin_ket = spin_equal(a, b)
                        dspin_exchange = not spin_equal(g, a)

                        dspin = dspin_bra * dspin_ket * dspin_exchange

                        pos_direct = pos_equal(g, a) * pos_equal(d, b)
                        pos_exchange = pos_equal(g, b) * pos_equal(d, a)

                        u[g, d, a, b] = self.V * dspin * (pos_direct - pos_exchange)

        return u

    def _get_HW(self, L: int, N: int) -> np.ndarray:
        spin_oposite = lambda i, j: i // N != j // N
        pos_equal = lambda i, j: i - N == j or i == j - N

        u = np.zeros_like(self.u)

        for a in range(L):
            for ap in range(L):
                for b in range(L):
                    for bp in range(L):
                        dspin_bra = spin_oposite(b, bp)
                        dspin_ket = spin_oposite(a, ap)

                        dspin = dspin_bra * dspin_ket

                        pos_direct = pos_equal(a, b) * pos_equal(ap, bp)
                        pos_exchange = pos_equal(b, ap) * pos_equal(a, bp)

                        u[b, bp, a, ap] = self.W * dspin * (pos_direct - pos_exchange)

        return u
