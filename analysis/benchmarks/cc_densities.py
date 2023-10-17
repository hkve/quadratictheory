from clusterfock.basis import PyscfBasis
from clusterfock.hf import HF
from clusterfock.cc import CCD, CCSD

from quantum_systems import construct_pyscf_system_rhf
from coupled_cluster import CCSD as HyCCSD
from coupled_cluster import CCD as HyCCD

from cccbdb_comparison import angstrom_to_bohr
import numpy as np


def ccd(atom, basis, tol=1e-4):
    b = PyscfBasis(atom=atom, basis=basis, restricted=False)

    hf = HF(b).run()
    b.change_basis(hf.C)
    ccd = CCD(b, intermediates=False).run(include_l=True, tol=1e-8)
    ccd.one_body_density()
    rho_cf = ccd.rho_ob

    system = construct_pyscf_system_rhf(molecule=angstrom_to_bohr(atom), basis=basis)
    ccd_hyqd = HyCCD(system)
    ccd_hyqd.compute_ground_state(t_kwargs={"tol": 1e-8}, l_kwargs={"tol": 1e-8})

    rho_hyqd = ccd_hyqd.compute_one_body_density_matrix().real

    diff = np.abs(rho_cf - rho_hyqd)

    indicies = np.argwhere(diff > tol)

    for index in indicies:
        i, j = index
        # if i >= j:
        print(f"CF: {rho_cf[i,j]:.4e}, HYQD: {rho_hyqd[i,j]:.4e}, (i,j) = ({i},{j})")

    print(f"Missmatched {len(indicies)}/{rho_cf.size}")


def ccsd(atom, basis):
    pass


def main():
    ccd("Li 0 0 0; H 0 0 1.2", "cc-pVDZ")
    # ccd("He 0 0 0", basis="cc-pVDZ")
    # ccd("Be 0 0 0", basis="cc-pVDZ")


if __name__ == "__main__":
    main()
