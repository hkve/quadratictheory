import numpy as np
from abc import ABC, abstractmethod
from clusterfock.basis import Basis
from clusterfock.cc.coupledcluster import CoupledCluster
from clusterfock.cc.parameter import CoupledClusterParameter

import warnings

class CoupledCluster_T1(CoupledCluster):
    @abstractmethod
    def __init__(self, basis: Basis, t_orders: list, l_orders: list = None, copy: bool = False):
        if copy:
            basis_copy = basis.copy()
        else:
            basis_copy = basis
            warnings.warn(f"{basis = } will be T1 transformed during calculations and should not be used for other calculations after .run() has been called")
        super().__init__(basis_copy, t_orders, l_orders)
        self._u = self.basis.u.copy()
        self._h = self.basis.h.copy()

        self.transforms_basis = True
        self._has_td_one_body = False

    def perform_t1_transform(self, t1: np.ndarray):
        basis = self.basis
        N, M, L = basis.N, basis.M, basis.L
        o, v = basis.o, basis.v

        X = np.eye(L, dtype=basis.dtype)
        Y = np.eye(L, dtype=basis.dtype)

        X[v, o] = X[v, o] - t1
        Y[o ,v] = Y[o, v] + t1.T

        basis.h = self.t1_transform_one_body(self._h, X, Y)
        basis.u = self.t1_transform_two_body(self._u, X, Y)
        basis.calculate_fock_matrix()

        # For time dependent calculations
        if self._has_td_one_body:
            basis.f = basis.f + self.t1_transform_one_body(self._external_contribution, X, Y)

    def t1_transform_one_body(self, operator, X, Y):
        return np.einsum("pr,qs,rs", X, Y, operator, optimize=True)
    
    def t1_transform_two_body(self, operator, X, Y):
        return np.einsum(
            "pt,qu,rm,sn,tumn->pqrs", X, X, Y, Y, operator, optimize=True
        )