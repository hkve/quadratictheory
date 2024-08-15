from quadratictheory.basis import Basis
from functools import cached_property
import numpy as np


class CustomBasis(Basis):
    def __init__(self, L: int, N: int, restricted: bool = True, dtype=float, **kwargs):
        super().__init__(L=L, N=N, restricted=restricted, dtype=dtype)
        self.custom_cached_operators = {}

    def set_custom_cached_operator(self, operator_name, operator):
        self.custom_cached_operators[operator_name] = operator

    def _get_custom_cached_operator(self, operator_name):
        custom_cached = list(self.custom_cached_operators.keys())
        
        if not operator_name in custom_cached:
            raise ValueError
        
        else:
            return self.custom_cached_operators.pop(operator_name)

    @cached_property
    def r(self) -> np.array:
        r = self._get_custom_cached_operator("r")

        return self._new_one_body_operator(r)
     