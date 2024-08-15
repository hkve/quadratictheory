from quadratictheory.basis import Basis
from quadratictheory.cc.ccd import GCCD, RCCD
from quadratictheory.cc.ccsd import GCCSD, RCCSD
from quadratictheory.cc.uccd import UCCD2
from quadratictheory.cc.qccd import QCCD, RQCCD
from quadratictheory.cc.qccsd import QCCSD

from quadratictheory.cc.ccsd_t1 import GCCSD_T1
from quadratictheory.cc.qccsd_t1 import QCCSD_T1


def CCD(basis: Basis, intermediates=True):
    return RCCD(basis) if basis.restricted else GCCD(basis, intermediates)


def CCSD(basis: Basis, intermediates=True):
    return (
        RCCSD(basis, intermediates=True)
        if basis.restricted
        else GCCSD(basis, intermediates=intermediates)
    )
