import numpy as np
from clusterfock.basis import Basis
from clusterfock.cc.coupledcluster_t1 import QuadCoupledCluster_T1
from clusterfock.cc.parameter import CoupledClusterParameter

from clusterfock.cc.rhs.t1transform_t_inter_QCCSD import (
    t1_transform_t1_intermediates_qccsd,
    t1_transform_t2_intermediates_qccsd
)

from clusterfock.cc.rhs.t1transform_l_inter_QCCSD import t1_transform_l_intermediates_qccsd

from clusterfock.cc.energies.e_inter_qccsd_t1transformed import t1transformed_qccsd_energy  
# from clusterfock.cc.rhs.t_CCSD import amplitudes_ccsd
# from clusterfock.cc.rhs.l_CCSD import lambda_amplitudes_ccsd
# from clusterfock.cc.rhs.t_inter_CCD import amplitudes_intermediates_ccd

# from clusterfock.cc.rhs.t1transform_CCSD import (
#     t1_transform_intermediates_ccsd,
#     t1_transform_lambda_intermediates_ccsd
# )
# from clusterfock.cc.rhs.l_inter_CCSD import lambda_amplitudes_intermediates_ccsd

# from clusterfock.cc.densities.l_CCSD import one_body_density, two_body_density
# from clusterfock.cc.energies.e_inter_ccsd import td_energy_addition

# from clusterfock.cc.densities.l_CCSD_t1transformed import one_body_density, two_body_density
# from clusterfock.cc.energies.e_inter_ccsd_t1transformed import td_energy_addition

class GQCCSD_T1(QuadCoupledCluster_T1):
    def __init__(self, basis: Basis, intermediates: bool = True, copy=False):
        assert not basis.restricted, "T1-transformed CCSD can not deal with restricted basis"

        t_orders = [1,2]
        l_orders = [1,2]
        super().__init__(basis, t_orders, l_orders, copy=copy)

        self.t1_rhs = t1_transform_t1_intermediates_qccsd
        self.t2_rhs = t1_transform_t2_intermediates_qccsd
        self.l_rhs = t1_transform_l_intermediates_qccsd

        self.energy_expression = t1transformed_qccsd_energy

    def _next_t_iteration(self, t: CoupledClusterParameter, l: CoupledClusterParameter) -> dict:
        basis = self.basis

        self.perform_t1_transform(t1=t[1])

        rhs1 = self.t1_rhs(
            t2=t[2],
            l1=l[1],
            l2=l[2],
            u=basis.u,
            f=basis.f,
            v=basis.v,
            o=basis.o,
        )

        rhs2 = self.t2_rhs(
            t2=t[2],
            l1=l[1],
            l2=l[2],
            u=basis.u,
            f=basis.f,
            v=basis.v,
            o=basis.o,
        )
        
        rhs = CoupledClusterParameter(t.orders, t.N, t.M, dtype=t.dtype)
        rhs.initialize_dicts({1: rhs1, 2: rhs2})

        return rhs

    def _next_l_iteration(self, t: CoupledClusterParameter, l: CoupledClusterParameter) -> CoupledClusterParameter:
        basis = self.basis
        M, N = basis.M, basis.N

        rhs1, rhs2 = self.l_rhs(
            t2=t[2],
            l1=l[1],
            l2=l[2],
            u=basis.u,
            f=basis.f,
            v=basis.v,
            o=basis.o,
        )

        rhs = CoupledClusterParameter(l.orders, l.N, l.M, dtype=l.dtype)
        rhs.initialize_dicts({1: rhs1, 2: rhs2})

        return rhs


    def _evaluate_cc_energy(self) -> float:
        basis = self.basis
        o, v = basis.o, basis.v
        u, f = basis.u, basis.f
        t2 = self._t[2]
        l1, l2 = self._l[1], self._l[2]

        return self.energy_expression(t2, l1, l2, u, f, o, v)
    
    # def _calculate_one_body_density(self) -> np.ndarray:
    #     basis = self.basis
    #     rho = np.zeros((basis.L, basis.L), dtype=basis.dtype)

    #     l1 = self._l[1]
    #     l2, t2 = self._l[2], self._t[2]
    #     o, v = basis.o, basis.v

    #     M, N = basis.M, basis.N
    #     rho = one_body_density(rho, t2, l1, l2, o, v)

    #     return rho
    
    # def _calculate_two_body_density(self) -> np.ndarray:
    #     basis = self.basis
    #     rho = np.zeros((basis.L, basis.L, basis.L, basis.L), dtype=basis.dtype)

    #     l1 = self._l[1]
    #     l2, t2 = self._l[2], self._t[2]
    #     o, v = basis.o, basis.v

    #     M, N = basis.M, basis.N
    #     rho = two_body_density(rho, t2, l1, l2, o, v)

    #     return rho
