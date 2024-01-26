from abc import ABC, abstractmethod
import numpy as np

class Pulse:
    def __init__(self, u, *args):
        self._u = u

    def r_dot_u(self, r):
        return np.einsum("xij,x->ij", r, self._u, optimize=True)
    
    def __call__(self, t, basis):
        return -self.r_dot_u(basis.r) * self.E(t)

    @abstractmethod
    def E(t):
        pass

class Sin2(Pulse):
    def __init__(self, u, F_str, omega, tprime):
        super().__init__(u)

        self._F_str = F_str
        self._omega = omega
        self._tprime = tprime

    def E(self, t):
        return (
            (np.sin(np.pi * t / self._tprime) ** 2)
            * np.heaviside(t, 1.0)
            * np.heaviside(self._tprime - t, 1.0)
            * np.sin(self._omega * t)
            * self._F_str
        )
    

class DeltaKick(Pulse):
    def __init__(self, u, F_str, dt):
        super().__init__(u)

        self._F_str = F_str
        self._dt = dt

    def E(self, t):
        if t < self._dt:
            return self._F_str/self._dt
        else:
            return 0