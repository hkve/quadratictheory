from clusterfock.basis.basis import Basis

import numpy as np


class TestBasis(Basis):
    __test__ = False

    def __init__(
        self, L: int, N: int, restricted: bool = False, dtype: type = float, seed: int = 42
    ):
        super().__init__(L, N, restricted, dtype=dtype)
        np.random.seed(seed)

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
