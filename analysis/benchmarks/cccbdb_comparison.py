import clusterfock as cf
import pyscf
from pyscf.cc.ccd import CCD as pyscfCCD
from pyscf.cc.ccsd import CCSD as pyscfCCSD

from pyscf.cc import GCCSD

def run_hf(atom, basis, db_results, restricted=False, tol=1e-8):
    basis = cf.PyscfBasis(atom=atom["hf"], basis=basis, restricted=restricted)

    hf = cf.HF(basis).run(tol=tol)
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
    b = cf.PyscfBasis(atom=atom["ccd"], basis=basis, restricted=True)
    hf = cf.HF(b).run(tol=tol)
    b.change_basis(hf.C)
    b.from_restricted()

    ccd = cf.CCD(b, intermediates=True).run(tol=tol)
    ccsd = cf.CCSD(b, intermediates=True).run(tol=tol)


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

    b = cf.PyscfBasis(atom=atom["ccsd"], basis=basis, restricted=True)
    hf = cf.HF(b).run(tol=tol)
    b.change_basis(hf.C)
    b.from_restricted()

    hf_pyscf = pyscf.scf.HF(b.mol).run(verbose=0, tol=tol)
    ccsd_pyscf = GCCSD(hf_pyscf).run(verbose=0, tol=tol)

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
    He = {"hf": -2.855160, "ccd": -2.887592, "ccsd": -2.887595}
    He_atom = {"hf": "He 0 0 0", "ccd": "He 0 0 0", "ccsd": "He 0 0 0"}
    run(atom=He_atom, basis="cc-pVDZ", db_results=He)

    LiH = {"hf": -7.983686, "ccd": -8.014079, "ccsd": -8.014421}
    LiH_atom = {
        "hf": "Li 0 0 0; H 0 0 1.6190",
        "ccd": "Li 0 0 0; H 0 0 1.6167",
        "ccsd": "Li 0 0 0; H 0 0 1.6191",
    }
    run(atom=LiH_atom, basis="cc-pVDZ", db_results=LiH)

    H2 = {"hf": -1.128746, "ccd": -1.163530, "ccsd": -1.163673}
    H2_atom = {
        "hf": "H 0 0 0; H 0 0 0.7480",
        "ccd": "H 0 0 0; H 0 0 0.7602",
        "ccsd": "H 0 0 0; H 0 0 0.7609",
    }
    run(atom=H2_atom, basis="cc-pVDZ", db_results=H2)

    N2 = {"hf": -108.955559, "ccd": -109.260854, "ccsd": -109.263578}
    N2_atom = {
            "hf": "N 0 0 0; N 0 0 1.0773", 
            "ccd": "N 0 0 0; N 0 0 1.1108", 
            "ccsd": "N 0 0 0; N 0 0 1.1128",
               }
    run(atom=N2_atom, basis="cc-pVDZ", db_results=N2)