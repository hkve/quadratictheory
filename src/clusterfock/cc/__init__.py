from clusterfock.basis import Basis
from clusterfock.cc.ccd import GCCD, RCCD


def CCD(basis: Basis, intermediates=True):
    return RCCD(basis) if basis.restricted else GCCD(basis, intermediates)
