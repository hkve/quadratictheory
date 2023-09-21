import numpy as np
from clusterfock.basis import Basis
from clusterfock.cc.coupledcluster import CoupledCluster


class CCD(CoupledCluster):
    def __init__(self, basis: Basis):
        t_amplitude_orders = [1, 2]
        l_amplitude_orders = [1, 2]

        super().__init__(basis, t_amplitude_orders, l_amplitude_orders)

    def _next_iteration(self, t: np.ndarray) -> np.ndarray:
        pass

    def _evaluate_cc_energy(self, t: np.ndarray) -> float:
        pass
