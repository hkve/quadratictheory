from clusterfock.basis import FiniteDifferenceBasis, FiniteDifferenceBasisFunctions
import numpy as np
from scipy import special


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

    def _potential(self, x):
        o = self._omega
        return 0.5 * o**2 * x**2


class HarmonicOscillatorOneDimension(FiniteDifferenceBasis):
    def __init__(self, L: int, N: int, restricted: bool = True, omega=1.0):
        phi = FunctionsODHO(omega=omega)
        super().__init__(L=L, N=N, phi=phi, restricted=True)
        self.x = np.linspace(-5, 5, 10000)
        self.omega = omega
        self.setup()
