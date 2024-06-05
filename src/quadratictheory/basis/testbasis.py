from quadratictheory.basis.basis import Basis

import numpy as np
from functools import cached_property


class TestBasis(Basis):
    __test__ = False

    def __init__(
        self, L: int, N: int, restricted: bool = False, dtype: type = float, seed: int = 42
    ):
        super().__init__(L, N, restricted, dtype=dtype)

        np.random.seed(seed)
        self.setup()

    def setup(self):
        ob_shape = self._one_body_shape
        tb_shape = self._two_body_shape

        self.h = np.random.uniform(low=-1, high=1, size=ob_shape)
        self.h = self.h.T @ self.h

        self.u = np.random.uniform(low=-1, high=1, size=tb_shape)
        self.u = self.u + self.u.transpose(1, 0, 3, 2)

        self.s = np.random.uniform(low=-1, high=1, size=ob_shape)
        self.s = self.s.T @ self.s
        np.fill_diagonal(self.s, 1)

        return self

    def _fetch_r(self):
        from time import sleep

        sleep(0.01)

        r = np.random.uniform(low=-1, high=1, size=(3, *self._one_body_shape))

        return self._new_one_body_operator(r, add_spin=False)

    @cached_property
    def r(self):
        return self._fetch_r()

    @property
    def mu(self):
        return 2 * self.r
