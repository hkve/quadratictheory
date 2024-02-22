from clusterfock.basis import Basis
from clusterfock.cc.ccd import GCCD, RCCD
from clusterfock.cc.ccsd import GCCSD, RCCSD
from clusterfock.cc.uccd import UCCD2
from clusterfock.cc.qccd import QCCD, RQCCD
from clusterfock.cc.qccsd import QCCSD

from clusterfock.cc.ccsd_t1 import GCCSD_T1

def CCD(basis: Basis, intermediates=True):
    return RCCD(basis) if basis.restricted else GCCD(basis, intermediates)


def CCSD(basis: Basis, intermediates=True):
    return (
        RCCSD(basis, intermediates=True)
        if basis.restricted
        else GCCSD(basis, intermediates=intermediates)
    )
