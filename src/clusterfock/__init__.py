from clusterfock.basis.basis import Basis
from clusterfock.basis.testbasis import TestBasis
from clusterfock.basis.pyscfbasis import PyscfBasis
from clusterfock.basis.lipkin import Lipkin

from clusterfock.hf.ghf import GHF
from clusterfock.hf.rhf import RHF
from clusterfock.basis import Basis


def HF(basis: Basis):
    return RHF(basis) if basis.restricted else GHF(basis)
