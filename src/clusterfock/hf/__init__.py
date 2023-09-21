from clusterfock.hf.ghf import GHF
from clusterfock.hf.rhf import RHF
from clusterfock.basis import Basis


def HF(basis: Basis):
    return RHF(basis) if basis.restricted else GHF(basis)
