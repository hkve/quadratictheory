from clusterfock.basis import FiniteDifferenceBasis, FiniteDifferenceBasisFunctions
import numpy as np
from scipy import special
import numba

class FunctionsODHO(FiniteDifferenceBasisFunctions):
    def __init__(self, omega: float):
        super().__init__(eigenfunction=True, orthonormal=False)
        self._omega = omega

    def _raw(self, n, x):
        o = self._omega
        return np.exp(-o / 2 * x**2) * special.hermite(n)(np.sqrt(o) * x)
    
    def _normalization(self, n):
        o = self._omega
        return (o / np.pi) ** (0.25) / np.sqrt(2**n * special.factorial(n))


class HarmonicOscillatorOneDimension(FiniteDifferenceBasis):
    def __init__(self, L: int, N: int, restricted: bool = True, omega=1.0, a=1.0, **kwargs):
        defaults = {"x": (-5, 5, 5000)}
        defaults.update(kwargs)

        phi = FunctionsODHO(omega=omega)
        super().__init__(L=L, N=N, phi=phi, restricted=restricted, x=defaults["x"])
        self._omega = omega
        self._a = a
        self.setup()

    def _potential(self, x):
        o = self._omega
        return 0.5 * o**2 * x**2

    def _interaction(self, x, y):
        a = self._a
        return 1 / np.sqrt((x - y) ** 2 + a**2)
