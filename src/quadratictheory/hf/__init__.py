from quadratictheory.hf.ghf import GHF
from quadratictheory.hf.rhf import RHF
from quadratictheory.basis import Basis


def HF(basis: Basis):
    return RHF(basis) if basis.restricted else GHF(basis)
