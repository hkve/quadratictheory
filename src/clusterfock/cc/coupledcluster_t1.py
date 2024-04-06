import numpy as np
from abc import ABC, abstractmethod
from clusterfock.basis import Basis
from clusterfock.cc.coupledcluster import CoupledCluster
from clusterfock.cc.quadcoupledcluster import QuadraticCoupledCluster
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
        self.transform_cached = False
        self._has_td_one_body = False

    def run(
        self, tol: float = 1e-8, maxiters: int = 1000, include_l: bool = False, vocal: bool = False
    ) -> CoupledCluster:
        super().run(tol, maxiters, include_l, vocal)

        self.copy_cached_operators()
        self.perform_t1_transform(self._t[1])
        
        return self
    
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
        if self.transform_cached:
            cached_operators = basis._check_cached_operators()
            
            for operator in cached_operators:
                basis.__dict__[operator] = self.t1_transform_one_body(self.__dict__[operator], X, Y)


    def copy_cached_operators(self):
        self.transform_cached = True
        basis = self.basis
        
        cached_operators = basis._check_cached_operators()

        for operator in cached_operators:
            self.__dict__[operator] = basis.__dict__[operator].copy()

    def t1_transform_one_body(self, operator, X, Y):
        return np.einsum("pr,qs,...rs->...pq", X, Y, operator, optimize=True)
    
    def t1_transform_two_body(self, operator, X, Y):
        return np.einsum(
            "pt,qu,rm,sn,tumn->pqrs", X, X, Y, Y, operator, optimize=True
        )

class QuadCoupledCluster_T1(QuadraticCoupledCluster):
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
        self.transform_cached = False
        self._has_td_one_body = False

    def run(
        self, tol: float = 1e-8, maxiters: int = 1000, vocal: bool = False
    ) -> CoupledCluster:
        super().run(tol, maxiters, vocal)

        self.copy_cached_operators()
        self.perform_t1_transform(self._t[1])
        
        return self
    
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
        if self.transform_cached:
            cached_operators = basis._check_cached_operators()
            
            for operator in cached_operators:
                basis.__dict__[operator] = self.t1_transform_one_body(self.__dict__[operator], X, Y)


    def copy_cached_operators(self):
        self.transform_cached = True
        basis = self.basis
        
        cached_operators = basis._check_cached_operators()

        for operator in cached_operators:
            self.__dict__[operator] = basis.__dict__[operator].copy()

    def t1_transform_one_body(self, operator, X, Y):
        return np.einsum("pr,qs,...rs->...pq", X, Y, operator, optimize=True)
    
    def t1_transform_two_body(self, operator, X, Y):
        return np.einsum(
            "pt,qu,rm,sn,tumn->pqrs", X, X, Y, Y, operator, optimize=True
        )