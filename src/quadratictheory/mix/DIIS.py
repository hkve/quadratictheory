from quadratictheory.mix import Mixer
from collections import deque
import numpy as np
from scipy.linalg import solve

# import warnings

# warnings.filterwarnings("ignore")


class DIISMixer(Mixer):
    def __init__(self, n_vectors: int = 4):
        self.n_vectors = n_vectors
        self.n_stored = 0
        self.iter = 0

        self.errors = deque([None] * n_vectors)
        self.vectors = deque([None] * n_vectors)

    def reset(self):
        self.n_stored = 0
        self.iter = 0

        self.errors = deque([None] * self.n_vectors)
        self.vectors = deque([None] * self.n_vectors)

    def __call__(self, p: np.ndarray, dp: np.ndarray) -> np.ndarray:
        self.n_stored += 1

        if self.n_stored > self.n_vectors:
            self.n_stored = self.n_vectors
            self.errors.rotate(-1)
            self.vectors.rotate(-1)

        pos = self.n_stored - 1

        new = p + dp
        old = p

        shape = new.shape
        new = new.ravel()
        old = old.ravel()

        self.errors[pos] = new - old
        self.vectors[pos] = new

        B = np.zeros(shape=(self.n_stored + 1, self.n_stored + 1), dtype=new.dtype)

        for i in range(self.n_stored):
            B[i, i] = np.dot(self.errors[i], self.errors[i])
            for j in range(i + 1, self.n_stored):
                B[i, j] = np.dot(self.errors[i], self.errors[j])
                B[j, i] = B[i, j]

        B[:-1, -1] = 1
        B[-1, :-1] = 1

        rhs = np.zeros(shape=self.n_stored + 1)
        rhs[-1] = 1

        c = np.linalg.solve(B, rhs)
        c = c[:-1]

        self.iter += 1

        mixed_vector = np.zeros_like(new)

        for i in range(self.n_stored):
            mixed_vector += c[i] * self.vectors[i]

        return mixed_vector.reshape(shape)
