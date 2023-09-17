from mypack.basis.basis import Basis

import numpy as np


class TestBasis(Basis):
    def __init__(self, L, N, restricted=False, dtype=float):
        super().__init__(L, N, restricted, dtype=dtype)

    def setup(self):
        self.h = np.random.uniform(low=-1, high=1, size=self.h.shape)
        self.h = self.h.T @ self.h

        self.u = np.random.uniform(low=-1, high=1, size=self.u.shape)
        self.u = self.u.transpose(1, 0, 3, 2)

        self.s = np.random.uniform(low=-1, high=1, size=self.s.shape)
        self.s = self.s.T @ self.s
        np.fill_diagonal(self.s, 1)
