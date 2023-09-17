from mypack.basis import Basis
from mypack.hf.hartreefock import HartreeFock


class RHF(HartreeFock):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
