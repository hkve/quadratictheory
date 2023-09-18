from mypack.basis import Basis
from abc import ABC, abstractmethod
import numpy as np


class CoupledCluster(ABC):
    def __init__(self, basis: Basis) -> None:
        self.basis = basis
        self.has_run = False
        self.converged = False

    def run(self):
        pass

    @abstractmethod
    def next_iteration(self, t):
        pass

    @abstractmethod
    def evalute_energy_iteration(self, t):
        pass
