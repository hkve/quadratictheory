from quadratictheory.basis import PyscfBasis
from quadratictheory.hf import HF
from quadratictheory.cc import CCD, CCSD

from coupled_cluster import CCSD as HyCCSD
from coupled_cluster import CCD as HyCCD

from cccbdb_comparison import angstrom_to_bohr
import numpy as np


def run_qt(atom, basis):
    b = PyscfBasis(atom=atom, basis=basis, restricted=False)

    hf = HF(b).run()
    print(hf.energy())
    b.change_basis(hf.C)
    ccd = CCD(b, intermediates=False).run(include_l=True, tol=1e-8)
    ccd.one_body_density()

    return ccd

def ccd_expvals(atom, basis):
    b = PyscfBasis(atom, basis, restricted=False)
    hf = HF(b).run()
    b.change_basis(hf.C)

    rho_qt = run_qt(atom, basis).rho_ob

    r_qt = np.einsum("pq,iqp->i", rho_qt, b.r)

    print(f"For {atom}")
    print(f"{r_qt = }\n")


def main():
    # ccd("He 0 0 0", basis="cc-pVDZ")
    # ccd("Be 0 0 0", basis="cc-pVDZ")
    # ccd("Li 0 0 0; H 0 0 1.2", "cc-pVDZ")

    # ccd_expvals("He 0 0 0", basis="cc-pVDZ")
    # ccd_expvals("Be 0 0 0", basis="cc-pVDZ")
    ccd_expvals("Li 0 0 0; H 0 0 1.2", "cc-pVDZ")
    # ccd_expvals("N 0 0 0; N 0 0 1.2", "cc-pVDZ")


if __name__ == "__main__":
    main()
