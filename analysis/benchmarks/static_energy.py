import numpy as np
import pandas as pd
import pathlib as pl
import json

import clusterfock as cf
from pyscf import gto, scf, fci, ao2mo
from pyscf.cc.ccd import CCD
from pyscf.cc.ccsd import CCSD

def run_cf_CC(geometry, basis, CC, restricted, **kwargs):
    mixer = kwargs.get("mixer", None)
    
    b = cf.PyscfBasis(geometry, basis, restricted=restricted).pyscf_hartree_fock()
    cc = CC(b)
    

    if mixer: cc.mixer = mixer
    cc.run(tol=1e-8, vocal=True, maxiters=100)
    if mixer: mixer.reset()

    if not cc.info["t_converged"]:
        return 0

    return cc.energy()

def run_FCI(geometry, basis):
    mol = gto.M(unit="bohr")
    mol.verbose = 3
    mol.build(atom=geometry, basis=basis)

    s = mol.intor_symmetric("int1e_ovlp")

    hf = scf.RHF(mol)
    hf.conv_tol_grad = 1e-5
    hf.max_cycle = 1000
    hf.kernel()

    n_fci_states = 3

    myfci = fci.direct_spin0.FCI(mol)
    myfci.conv_tol = 1e-10
    h1 = hf.mo_coeff.T.dot(hf.get_hcore()).dot(hf.mo_coeff)
    eri = ao2mo.kernel(mol, hf.mo_coeff)
    e_fci, c_fci = myfci.kernel(
        h1, eri, h1.shape[1], mol.nelec, ecore=mol.energy_nuc(), nroots=5
    )

    return e_fci[0]

def run_pyscf_cc(geometry, basis, CC):
    mol = gto.M(
        atom = geometry,
        basis = basis,
        verbose = 0,
        unit="bohr"
    )

    mf = scf.RHF(mol)
    mf.kernel()

    mycc = CC(mf)
    mycc.conv_tol  = 1e-8
    mycc.kernel()

    return mycc.e_tot

def run_pyscf_ccsd(geometry, basis):
    pass

def save(results, name, basis, folder="static_energy_tests"):
    path = f"{folder}/{name}_{basis}.csv"
    
    with open(path, 'w') as f:
        json.dump(results, f, indent=4)

def load(name, basis, folder="static_energy_tests"):
    path = f"{folder}/{name}_{basis}.csv"

    with open(path) as f:
        data = json.load(f)    

    return data

def two_particles(run=False):
    geometries = {
        "He": "He 0 0 0",
        "H2": "H  0 0 0; H 0 0 1.4040664095",
    }

    basis = [
        "6-31g",
        "cc-pVDZ",
        "cc-pVTZ"
    ]

    cc_methods = [cf.CCD, cf.QCCD, cf.CCSD, cf.QCCSD]

    mixer = cf.mix.DIISMixer(n_vectors=10)

    if run:
        for name, geometry in geometries.items():
            for b in basis:
                results = {}

                for CC in cc_methods:
                    cc_energy = run_cf_CC(geometry, b, CC, restricted=False, mixer=mixer)
                    results[CC.__name__] = cc_energy

                e_fci = run_FCI(geometry, b)

                results["FCI"] = e_fci

                save(results, name, b)

    name = "H2"

    data = {}
    for b in basis:
        data[b] = load(name, b)

    df = pd.DataFrame(data).T

    print(df.to_latex(column_format="lccccc"))

def diatoms(run=False):
    geometries = {
        "Be": "Be 0 0 0",
        "O": "O 0 0 0",
        "Ne": "Ne 0 0 0",
        "Ar": "Ar 0 0 0",
    }

    basis = [
        "6-31g",
        "cc-pVDZ",
        "cc-pVTZ"
    ]

    spin_restrictions = [False, True]
    ccd_methods = [cf.GCCD, cf.RCCD]
    ccsd_methods = [cf.GCCSD, cf.RCCSD]

    mixer = cf.mix.DIISMixer(n_vectors=10)
    
    if run:
        for name, geometry in geometries.items():
            for b in basis:
                results = {}

                for CC, restricted in zip(ccd_methods, spin_restrictions):
                    cc_energy = run_cf_CC(geometry, b, CC, restricted=restricted, mixer=mixer)
                    results[CC.__name__] = cc_energy

                for CC, restricted in zip(ccsd_methods, spin_restrictions):
                    cc_energy = run_cf_CC(geometry, b, CC, restricted=restricted, mixer=mixer)
                    results[CC.__name__] = cc_energy

                results["p_CCD"] = run_pyscf_cc(geometry, b, CCD)
                results["p_CCSD"] = run_pyscf_cc(geometry, b, CCSD)
                
                save(results, name, b)

def other_atoms(run=False):
    geometries = {
        "LiH": "Li 0 0 0; H 0 0",
        "CHp": "C 0 0 0; H 0 0",
        "HF": "H 0 0 0; F 0 0",
    }

    equilibrium = {
        "LiH": 3.0519074716,
        "CHp": 2.1372800931,
        "HF": 1.7290992795,
    }

    basis = "sto-3g"

    cc_methods = [cf.CCD, cf.CCSD]
    if run:
        for name, geometry in geometries.items():
            R = equilibrium[name]
            
            for d in [1,2]:
                geometry += f" {d*R}"
                name += f"_{d}Re"

                results = {}

                for CC in cc_methods:
                    cc_energy = run_cf_CC(geometry, basis, CC, restricted=True)
                    results[CC.__name__] = cc_energy

                results["p_CCD"] = run_pyscf_cc(geometry, basis, CCD)
                results["p_CCSD"] = run_pyscf_cc(geometry, basis, CCSD)
                
                save(results, name, basis)

if __name__ == '__main__':
    # two_particles(run=False)
    # diatoms(run=True)
    other_atoms(run=True)