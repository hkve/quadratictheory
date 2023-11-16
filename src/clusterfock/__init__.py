from clusterfock.basis.basis import Basis
from clusterfock.basis.testbasis import TestBasis
from clusterfock.basis.pyscfbasis import PyscfBasis
from clusterfock.basis.lipkin import Lipkin

from clusterfock.hf.ghf import GHF
from clusterfock.hf.rhf import RHF
from clusterfock.basis import Basis

from clusterfock.cc.ccd import GCCD, RCCD
from clusterfock.cc.ccsd import GCCSD

from clusterfock.cc.uccd import UCCD2
from clusterfock.cc.qccd import QCCD

from clusterfock.td.tdcoupledcluster import TimeDependentCoupledCluster

def HF(basis: Basis):
    return RHF(basis) if basis.restricted else GHF(basis)


def CCD(basis: Basis, intermediates=True):
    return RCCD(basis) if basis.restricted else GCCD(basis, intermediates)


def CCSD(basis: Basis, intermediates=True):
    if basis.restricted:
        raise RuntimeError("Restricted CCSD is not implemented")

    return GCCSD(basis, intermediates=intermediates)
