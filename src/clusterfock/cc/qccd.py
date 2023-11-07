import numpy as np
from clusterfock.basis import Basis
from clusterfock.cc.quadcoupledcluster import QuadraticCoupledCluster
from clusterfock.cc.parameter import CoupledClusterParameter



class QCCD(QuadraticCoupledCluster):
    def __init__(self, basis: Basis, intermediates: bool = False):
        assert not basis.restricted, "QCCD can not deal with restricted basis"

        t_orders = [2]
        l_orders = [2]
        super().__init__(basis, t_orders, l_orders)

        self.t_rhs = amplitudes_intermediates_ccd if intermediates else amplitudes_ccd
        self.l_rhs = lambda_amplitudes_intermediates_ccd if intermediates else lambda_amplitudes_ccd
