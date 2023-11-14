from __future__ import annotations
import numpy as np
from functools import reduce
import operator


def merge_to_flat(t: CoupledClusterParameter, l: CoupledClusterParameter) -> np.ndarray:
    t_flat, l_flat = t.to_flat(), l.to_flat()
    t_len, l_len = len(t_flat), len(l_flat)

    t_slice = slice(0, t_len)
    l_slice = slice(t_len, t_len + l_len)

    flat = np.r_[t.to_flat(), l.to_flat()]

    return flat, t_slice, l_slice


class CoupledClusterParameter:
    def __init__(self, orders: list, N: int, M: int, dtype=float):
        if orders is None:
            return None
        self.orders = orders

        self.N, self.M = N, M

        self._dtype = dtype
        self._data = {}
        self._data_shape = {}

        self._get_info()

    def _get_info(self):
        # Iterate over orders and get the shapes
        for order in self.orders:
            shape = tuple([self.M] * order + [self.N] * order)
            self._data_shape[order] = shape

        # Now get sizes and flat slices
        prod = lambda shape: reduce(operator.mul, shape, 1)
        sizes = [0] + [prod(shape) for shape in self._data_shape.values()]
        offset = 0
        self._flat_slices = {}

        for i, order in enumerate(self.orders):
            order_slice = slice(sizes[i], offset + sizes[i + 1])
            self._flat_slices[order] = order_slice
            offset += sizes[i + 1]

    def initialize_zero(self) -> CoupledClusterParameter:
        for order in self.orders:
            shape = self._data_shape[order]
            self._data[order] = np.zeros(shape, dtype=self.dtype)

        return self

    def initialize_epsilon(self, epsilon, inv=True) -> CoupledClusterParameter:
        eps_v = epsilon[self.N :]
        eps_o = epsilon[: self.N]

        if 1 in self.orders:
            self._data[1] = -eps_v[:, None] + eps_o[None, :]
        if 2 in self.orders:
            self._data[2] = (
                -eps_v[:, None, None, None]
                - eps_v[None, :, None, None]
                + eps_o[None, None, :, None]
                + eps_o[None, None, None, :]
            )

        if inv:
            for order in self.orders:
                self._data[order] = 1 / self._data[order]

        return self

    def initialize_dicts(self, data) -> CoupledClusterParameter:
        assert self.orders == list(data.keys())
        self._data = data

        return self

    def norm(self) -> list:
        return {o: np.linalg.norm(t) for o, t in self._data.items()}

    def to_flat(self) -> np.ndarray:
        return np.concatenate(tuple(d.ravel() for d in self._data.values()))

    def from_flat(self, flat: np.ndarray):
        prod = lambda shape: reduce(operator.mul, shape, 1)
        sizes = [0] + [prod(shape) for shape in self._data_shape.values()]

        amplitudes = {}
        offset = 0
        for order in self.orders:
            order_slice = self._flat_slices[order]
            order_shape = self._data_shape[order]
            self._data[order] = flat[order_slice].reshape(order_shape)

    def __getitem__(self, order) -> np.ndarray:
        return self._data[order]

    def __mul__(self, other: CoupledClusterParameter) -> CoupledClusterParameter:
        product = None
        if type(other) == type(self):
            assert other.orders == self.orders
            product = {o: self[o] * other[o] for o in self.orders}
        else:
            assert type(other) == self.dtype
            product = {o: self[o] * other for o in self.orders}



        return CoupledClusterParameter(self.orders, self.N, self.M).initialize_dicts(product)

    def __rmul__(self, other: CoupledClusterParameter) -> CoupledClusterParameter:
        return self.__mul__(other)

    def copy(self):
        new = CoupledClusterParameter(self.orders, self.N, self.M, self.dtype)
        return new.initialize_dicts({o: self[o].copy() for o in self.orders})

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, dtype):
        self._dtype = dtype
        for o, t in self._data.items():
            self._data[o] = t.astype(dtype)
