from clusterfock.basis import Basis
from clusterfock.cc.ccd import GCCD, RCCD
from clusterfock.cc.ccsd import GCCSD
from clusterfock.cc.uccd import UCCD2
from clusterfock.cc.qccd import QCCD
from clusterfock.cc.tdcoupledcluster import TimeDependentCoupledCluster

def CCD(basis: Basis, intermediates=True):
    return RCCD(basis) if basis.restricted else GCCD(basis, intermediates)


def CCSD(basis: Basis, intermediates=True):
    if basis.restricted:
        raise RuntimeError("Restricted CCSD is not implemented")

    return GCCSD(basis, intermediates=intermediates)
