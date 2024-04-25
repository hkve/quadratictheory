from abc import ABC, abstractmethod
import numpy as np


class Pulse(ABC):
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
    def __init__(self, u, F_str, omega, tprime=None):
        super().__init__(u)

        self._F_str = F_str
        self._omega = omega

        if tprime is None:
            self._tprime = 2 * np.pi / omega
        else:
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
            return self._F_str / self._dt
        else:
            return 0


class LieEtAl(Pulse):
    def __init__(self, u, E_max, omega):
        super().__init__(u)

        self._E_max = E_max
        self._omega = omega

    def E(self, t):
        ot = self._omega * t
        E_max = self._E_max

        amplitude = 0

        if 0 < ot <= 2*np.pi:
            amplitude =  ot/(2*np.pi) * E_max
        elif 2*np.pi <= ot <= 4*np.pi:
            amplitude = E_max
        elif 4*np.pi <= ot <= 6*np.pi:
            amplitude = (3 - ot/(2*np.pi)) * E_max
        
        return amplitude*np.sin(ot)
    
class Luzanov(Pulse):
    def __init__(self, u, E_max, T, omega):
        super().__init__(u)
        
        self._omega = omega
        self._E_max = E_max
        self._T = T

    def E(self, t):
        T = self._T
        E_max = self._E_max
        omega = self._omega

        amplitude = 0

        if 0 < t <= T/3:
            amplitude = (3*t/T) * E_max
        elif T/3 <= t <= 2*T/3:
            amplitude = E_max
        elif 2*T/3 <= t <= T:
            amplitude = 3*E_max*(1- t/T)

        return amplitude * np.sin(omega*t)