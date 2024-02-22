import numpy as np
from abc import ABC, abstractmethod
from clusterfock.basis import Basis
from clusterfock.cc.coupledcluster import CoupledCluster
from clusterfock.cc.parameter import CoupledClusterParameter

class CoupledCluster_T1(CoupledCluster):
    @abstractmethod
    def __init__(self, basis: Basis, t_orders: list, l_orders: list = None):
        super().__init__(basis, t_orders, l_orders)
        self._u = self.basis.u.copy()

    def perform_t1_transform(self, t1: np.ndarray, h: np.ndarray, u: np.ndarray):
        basis = self.basis
        N, M, L = basis.N, basis.M, basis.L
        o, v = basis.o, basis.v

        X = np.eye(L, dtype=basis.dtype)
        Y = np.eye(L, dtype=basis.dtype)

        X[v, o] = X[v, o] - t1
        Y[o ,v] = Y[o, v] + t1.T


        h_tilde = np.einsum("pr,qs,rs", X, Y, h, optimize=True)
        u_tilde = np.einsum(
            "pt,qu,rm,sn,tumn->pqrs", X, X, Y, Y, u, optimize=True
        )
        f_tilde = h_tilde + np.einsum("piqi->pq", u_tilde[:, o, :, o])

        return f_tilde, u_tilde