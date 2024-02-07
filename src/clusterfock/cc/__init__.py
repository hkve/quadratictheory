from clusterfock.basis import Basis
from clusterfock.cc.ccd import GCCD, RCCD
from clusterfock.cc.ccsd import GCCSD, RCCSD
from clusterfock.cc.uccd import UCCD2
from clusterfock.cc.qccd import QCCD
from clusterfock.cc.qccsd import QCCSD


def CCD(basis: Basis, intermediates=True):
    return RCCD(basis) if basis.restricted else GCCD(basis, intermediates)

def CCSD(basis: Basis, intermediates=True):
    return RCCSD(basis) if basis.restricted else GCCSD(basis, intermediates=intermediates)
