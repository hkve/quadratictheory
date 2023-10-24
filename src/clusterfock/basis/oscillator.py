from clusterfock.basis import Basis
import numpy as np
import pyscf

class HarmonicOscillatorOneDimension(Basis):
    def __init__(self, L: int, N: int, restricted: bool = True):
        super().__init__(L=L, N=N, restricted=True)

        self.setup()

    def setup(self):
        pass