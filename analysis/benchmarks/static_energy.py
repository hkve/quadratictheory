import numpy as np
import pandas as pd
import pathlib as pl
import json

import quadratictheory as qt
from pyscf import gto, scf, fci, ao2mo
from pyscf.cc.ccd import CCD
from pyscf.cc.ccsd import CCSD

def run_cf_CC(geometry, basis, CC, restricted, **kwargs):
    mixer = kwargs.get("mixer", None)
    tol = kwargs.get("tol", 1e-8)
    charge = kwargs.get("charge", 0)
    maxiters = kwargs.get("maxiters", 100)
    b = qt.PyscfBasis(geometry, basis, restricted=restricted, charge=charge).pyscf_hartree_fock()
    cc = CC(b)
    

    if mixer: cc.mixer = mixer
    cc.run(tol=tol, vocal=True, maxiters=maxiters)
    if mixer: mixer.reset()

    if not cc.info["t_converged"]:
        return 0

    return cc.energy()

def run_FCI(geometry, basis, **kwargs):
    charge = kwargs.get("charge", 0)
    
    mol = gto.M(unit="bohr")
    mol.verbose = 3
    mol.charge = charge
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

def run_pyscf_cc(geometry, basis, CC, **kwargs):
    charge = kwargs.get("charge", 0)

    mol = gto.M(
        atom = geometry,
        basis = basis,
        verbose = 0,
        charge = charge,
        unit="bohr"
    )

    mf = scf.RHF(mol)
    mf.kernel()

    mycc = CC(mf)
    mycc.conv_tol  = 1e-8
    mycc.kernel()

    return mycc.e_tot


def save(results, name, basis, folder="static_energy_tests"):
    path = f"{folder}/{name}_{basis}.json"
    
    with open(path, 'w') as f:
        json.dump(results, f, indent=4)

def load(name, basis, folder="static_energy_tests"):
    path = f"{folder}/{name}_{basis}.json"

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

    cc_methods = [qt.CCD, qt.QCCD, qt.CCSD, qt.QCCSD]

    mixer = qt.mix.DIISMixer(n_vectors=10)

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

def atoms(run=False):
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
    ccd_methods = [qt.GCCD, qt.RCCD]
    ccsd_methods = [qt.GCCSD, qt.RCCSD]

    mixer = qt.mix.DIISMixer(n_vectors=10)
    
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


    name = "Ar"
    data = {}
    for b in basis:
        data[b] = load(name, b)
        
    df = pd.DataFrame(data).T
    df = df.reindex(["GCCD", "RCCD", "p_CCD", "GCCSD", "RCCSD", "p_CCSD"], axis=1)
    print(df.to_latex(column_format="lcccccc"))

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

    charge = {
        "LiH": 0,
        "CHp": 1,
        "HF": 0,
    }

    basis = "cc-pVTZ"

    cc_methods = [qt.CCD, qt.CCSD]
    if run:
        for name, geometry in geometries.items():
            R = equilibrium[name]
            q = charge[name]

            for d in [1,2]:
                geometry_dist = geometry + f" {d*R}"
                name_dist = name + f"_{d}Re"
                
                results = {}

                for CC in cc_methods:
                    cc_energy = run_cf_CC(geometry_dist, basis, CC, restricted=True, charge=q)
                    results[CC.__name__] = cc_energy

                results["p_CCD"] = run_pyscf_cc(geometry_dist, basis, CCD, charge=q)
                results["p_CCSD"] = run_pyscf_cc(geometry_dist, basis, CCSD, charge=q)
                
                save(results, name_dist, basis)

    name = "HF"
    data = {}
    for d in [1,2]:
        key = r"$R_e$" if d == 1 else rf"${d}R_e$" 
        data[key] = load(name + f"_{d}Re", basis)
        
    df = pd.DataFrame(data).T
    df = df.reindex(["CCD", "p_CCD", "CCSD", "p_CCSD"], axis=1)
    print(df.to_latex(column_format="llcccc"))


def QCCSD_benchmark_N2(run=False):
    distances = np.arange(1.5,8.0+0.1,0.5)
    basis = "sto-3g"
    folder = "static_energy_tests"
    filename_qcc = f"{folder}/N2_QCCSD.npz"
    filename_fci = f"{folder}/N2_FCI.npz"

    E_fci_fanetal = np.array([-106.720117, -107.623240, -107.651880, -107.546614, -107.473442, -107.447822, -107.441504, -107.439549, -107.438665, -107.438265, -107.438054, -107.438029])
    E_qcc_fanetal = np.array([0.886,1.988,3.443,3.909,5.294,15.815,27.792,35.335,39.983,42.609,44.839,45.508])
    distances_fanetal = np.array([1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0,7.0,8.0])

    if run:
        E_qcc = np.zeros_like(distances)
        for i, d in enumerate(distances):
            geometry = f"N 0 0 0; N 0 0 {d}"
            mixer = qt.mix.SoftStartDIISMixer(alpha=0.75, start_DIIS_after=20, n_vectors=20)
            E_qcc[i] = run_cf_CC(geometry, basis, qt.QCCSD, False, tol=1e-4, maxiters=200, mixer=mixer)
        np.savez(filename_qcc, E_qcc)

        E_fci = np.zeros_like(distances)
        for i, d in enumerate(distances):
            geometry = f"N 0 0 0; N 0 0 {d}"
            E_fci[i] = run_FCI(geometry, basis)
        np.savez(filename_fci, E_fci)

    E_qcc = np.load(filename_qcc)["arr_0"]
    E_fci = np.load(filename_fci)["arr_0"]

    E_qcc_cut, E_fci_cut = [], []
    for i, d in enumerate(distances):
        if d in distances_fanetal:
            E_qcc_cut.append(E_qcc[i]) 
            E_fci_cut.append(E_fci[i]) 

    E_qcc_cut = np.array(E_qcc_cut)
    E_fci_cut = np.array(E_fci_cut)
    
    df = {
        "R": distances_fanetal,
        "FCI_f": E_fci_fanetal,
        "QCCSD_f": E_qcc_fanetal,
        "FCI": E_fci_cut,
        "QCCSD": (E_qcc_cut - E_fci_cut)*1000,
    }

    formatters = {
        "R": lambda x: f"{x:.1f}",
        "FCI_f": lambda x: f"{x:.5f}",
        "QCCSD_f": lambda x: f"{x:.3f}",
        "FCI": lambda x: f"{x:.5f}",
        "QCCSD": lambda x: f"{x:.3f}",
    }
    df = pd.DataFrame(df)

    print(
        df.to_latex(index=False, formatters=formatters)
    )

def QCCSD_benhmark_Cooper_and_Knowls(run=False):
    folder = "static_energy_tests"
    filename_HF = f"{folder}/HF_QCCSD.npz"

    basis = "cc-pVDZ"

    # Distances from angstrom to bohr
    r = np.array([1.0,1.5,2.0,2.5,3.0])
    r *= 1.8897259886

    E0_fci = -100.230466
    fci_article = np.array([-0.000852,-0.004682,-0.023859,-0.085758,-0.176039]) + E0_fci
    qccsd_article = np.array([9.333,7.859,5.536,3.154,1.595])

    qccsd = np.zeros_like(r)

    if run:
        for i, d in enumerate(r):
            geometry = f"H 0 0 0; F 0 0 {d};"
            qccsd[i] = run_cf_CC(geometry, basis, qt.QCCSD, restricted=False, tol=1e-4)

        np.savez(filename_HF, qccsd)
    
    qccsd = np.load(filename_HF)["arr_0"] - fci_article

    from IPython import embed
    embed()
if __name__ == '__main__':
    # two_particles(run=False)
    # atoms(run=False)
    # other_atoms(run=False)
    
    # QCCSD_benchmark_N2(run=False)
    QCCSD_benhmark_Cooper_and_Knowls()