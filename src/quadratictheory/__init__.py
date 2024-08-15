from quadratictheory.basis.basis import Basis
from quadratictheory.basis.testbasis import TestBasis
from quadratictheory.basis.pyscfbasis import PyscfBasis
from quadratictheory.basis.custom import CustomBasis
from quadratictheory.basis.lipkin import Lipkin

from quadratictheory.hf.ghf import GHF
from quadratictheory.hf.rhf import RHF
from quadratictheory.basis import Basis

from quadratictheory.cc.ccd import GCCD, RCCD
from quadratictheory.cc.ccsd import GCCSD, RCCSD

from quadratictheory.cc.uccd import UCCD2
from quadratictheory.cc.qccd import QCCD, RQCCD
from quadratictheory.cc.qccsd import QCCSD

from quadratictheory.cc.ccsd_t1 import GCCSD_T1
from quadratictheory.cc.qccsd_t1 import QCCSD_T1

from quadratictheory.td.tdcoupledcluster import TimeDependentCoupledCluster
from quadratictheory.td import ImaginaryTimeCoupledCluster

from quadratictheory.td import pulse
from quadratictheory.td import sampler


def HF(basis: Basis):
    return RHF(basis) if basis.restricted else GHF(basis)


def CCD(basis: Basis, intermediates=True):
    return RCCD(basis) if basis.restricted else GCCD(basis, intermediates)


def CCSD(basis: Basis, intermediates=True):
    return (
        RCCSD(basis, intermediates=intermediates)
        if basis.restricted
        else GCCSD(basis, intermediates=intermediates)
    )
