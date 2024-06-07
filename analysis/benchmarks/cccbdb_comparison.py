import numpy as np

import quadratictheory as qt
import pyscf
from pyscf.cc.ccd import CCD as pyscfCCD
from pyscf.cc import GCCSD as pyscfCCSD


def angstrom_to_bohr(input_string):
    c = 1.8897259886

    atoms = input_string.split(";")
    for i in range(len(atoms)):
        atom = atoms[i].split(" ")
        for j in range(len(atom)):
            try:
                atom[j] = f"{float(atom[j])*c:.6f}"
            except:
                pass

        atoms[i] = " ".join(atom)

    return ";".join(atoms)


def run_hf(atom, basis, db_results, restricted=False, tol=1e-8):
    basis = qt.PyscfBasis(atom=atom["hf"], basis=basis, restricted=restricted)

    hf = qt.HF(basis).run(tol=tol)
    hf_pyscf = pyscf.scf.HF(basis.mol).run(verbose=0, tol=tol)

    print(
        f"""
        Hartree_fock for {basis._atom_string} with {basis._basis_string}
        clufo  = {hf.energy():>20.6f}
        pyscf  = {hf_pyscf.e_tot:>20.6f}
        cccbdb = {db_results['hf']:>20.6f}          
        """
    )


def run_cc(atom, basis, db_results, tol=1e-8):
    b = qt.PyscfBasis(atom=atom["ccd"], basis=basis, restricted=True)
    hf = qt.HF(b).run(tol=tol)
    b.change_basis(hf.C)
    b.from_restricted()

    ccd = qt.CCD(b, intermediates=True).run(tol=tol)
    ccsd = qt.CCSD(b, intermediates=True).run(tol=tol)

    hf_pyscf = pyscf.scf.HF(b.mol).run(verbose=0, tol=tol)
    ccd_pyscf = pyscfCCD(hf_pyscf).run(verbose=0, tol=tol)

    print(
        f"""
        CCD for {b._atom_string} with {b._basis_string}
        clufo  = {ccd.energy():>20.6f}
        pyscf  = {ccd_pyscf.e_tot:>20.6f}
        cccbdb = {db_results['ccd']:>20.6f}
        """
    )

    b = qt.PyscfBasis(atom=atom["ccsd"], basis=basis, restricted=True)
    hf = qt.HF(b).run(tol=tol)
    b.change_basis(hf.C)
    b.from_restricted()

    hf_pyscf = pyscf.scf.HF(b.mol).run(verbose=0, tol=tol)
    ccsd_pyscf = pyscfCCSD(hf_pyscf).run(verbose=0, tol=tol)
    
    print(
        f"""
        CCSD for {b._atom_string} with {b._basis_string}
        clufo  = {ccsd.energy():>20.6f}
        pyscf  = {ccsd_pyscf.e_tot:>20.6f}
        cccbdb = {db_results['ccsd']:>20.6f}
        """
    )


def run(atom, basis, db_results, restricted=False, mode=0, tol=1e-8):
    _run_hf, _run_cc = False, False
    if mode in [0, 1]:
        _run_hf = True
    if mode in [0, 2]:
        _run_cc = True

    if _run_hf:
        run_hf(atom, basis, db_results, restricted=True, tol=tol)
    if _run_cc:
        run_cc(atom, basis, db_results, tol=tol)


if __name__ == "__main__":
    mode = 2

    He = {"hf": -2.855160, "ccd": -2.887592, "ccsd": -2.887595}
    He_atom = {"hf": "He 0 0 0", "ccd": "He 0 0 0", "ccsd": "He 0 0 0"}
    run(atom=He_atom, basis="cc-pVDZ", db_results=He, mode=mode)

    LiH = {"hf": -7.983686, "ccd": -8.014079, "ccsd": -8.014421}
    LiH_atom = {
        "hf": "Li 0 0 0; H 0 0 1.6190",
        "ccd": "Li 0 0 0; H 0 0 1.6167",
        "ccsd": "Li 0 0 0; H 0 0 1.6191",
    }
    run(atom=LiH_atom, basis="cc-pVDZ", db_results=LiH, mode=mode)

    H2 = {"hf": -1.128746, "ccd": -1.163530, "ccsd": -1.163673}
    H2_atom = {
        "hf": "H 0 0 0; H 0 0 0.7480",
        "ccd": "H 0 0 0; H 0 0 0.7602",
        "ccsd": "H 0 0 0; H 0 0 0.7609",
    }
    run(atom=H2_atom, basis="cc-pVDZ", db_results=H2, mode=mode)

    N2 = {"hf": -108.955559, "ccd": -109.260854, "ccsd": -109.263578}
    N2_atom = {
        "hf": "N 0 0 0; N 0 0 1.0773",
        "ccd": "N 0 0 0; N 0 0 1.1108",
        "ccsd": "N 0 0 0; N 0 0 1.1128",
    }
    run(atom=N2_atom, basis="cc-pVDZ", db_results=N2, mode=mode)
