from mypack.hf.ghf import GHF
from mypack.hf.rhf import RHF
from mypack.basis import Basis


def HF(basis: Basis):
    return RHF(basis) if basis.restricted else GHF(basis)
