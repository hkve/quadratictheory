from clusterfock.basis import PyscfBasis
from clusterfock.hf import HF
from clusterfock.cc import CCD, CCSD

from quantum_systems import construct_pyscf_system_rhf
from coupled_cluster import CCSD as HyCCSD
from coupled_cluster import CCD as HyCCD

from cccbdb_comparison import angstrom_to_bohr
import numpy as np


def run_cf(atom, basis):
    b = PyscfBasis(atom=atom, basis=basis, restricted=False)

    hf = HF(b).run()
    b.change_basis(hf.C)
    ccd = CCD(b, intermediates=False).run(include_l=True, tol=1e-8)
    ccd.one_body_density()

    return ccd

def run_hyqd(atom, basis):
    system = construct_pyscf_system_rhf(molecule=angstrom_to_bohr(atom), basis=basis)
    ccd_hyqd = HyCCD(system)
    ccd_hyqd.compute_ground_state(t_kwargs={"tol": 1e-8}, l_kwargs={"tol": 1e-8})
    
    return ccd_hyqd.compute_one_body_density_matrix().real

def ccd(atom, basis, tol=1e-4):
    ccd = run_cf(atom, basis)
    rho_cf = ccd.rho_ob

    rho_hyqd = run_hyqd(atom, basis)

    diff = np.abs(rho_cf - rho_hyqd)

    indicies = np.argwhere(diff > tol)

    print(f"For {atom}")
    print(f"rho_cf_trace = {np.trace(rho_cf)}")
    print(f"rho_hyqd_trace = {np.trace(rho_hyqd)}")
    print(f"Missmatched {len(indicies)}/{rho_cf.size}")
    for index in indicies:
        i, j = index
        if i >= j:
            print(f"CF: {rho_cf[i,j]:.4e}, HYQD: {rho_hyqd[i,j]:.4e}, (i,j) = ({i},{j})")

def ccd_expvals(atom, basis):
    b = PyscfBasis(atom, basis, restricted=False)
    hf = HF(b).run()
    b.change_basis(hf.C)

    rho_cf = run_cf(atom, basis).rho_ob
    rho_hyqd = run_hyqd(atom, basis)

    r_cf = np.einsum("pq,iqp->i", rho_cf, b.r)
    r_hyqd = np.einsum("pq,iqp->i", rho_hyqd, b.r)

    print(f"For {atom}")
    print(f"{r_cf = }\n{r_hyqd = }")

def ccsd(atom, basis):
    pass


def main():
    ccd("He 0 0 0", basis="cc-pVDZ")
    ccd("Be 0 0 0", basis="cc-pVDZ")
    ccd("Li 0 0 0; H 0 0 1.2", "cc-pVDZ")

    # ccd_expvals("He 0 0 0", basis="cc-pVDZ")
    # ccd_expvals("Be 0 0 0", basis="cc-pVDZ")
    # ccd_expvals("Li 0 0 0; H 0 0 1.2", "cc-pVDZ")
    # ccd_expvals("N 0 0 0; N 0 0 1.2", "cc-pVDZ")


if __name__ == "__main__":
    main()
