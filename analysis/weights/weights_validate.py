import clusterfock as cf
import csv
import numpy as np
import pandas as pd
from fci_pyscf import fci_pyscf
import os
import matplotlib.pyplot as plt
import plot_utils as pl

PATH = "dat/validate/"

def run_hf(geometry, basis, restricted=False, **kwargs):
    default = {
        "charge": 0,
        "tol": 1e-10,
    }

    default.update(kwargs)

    charge = default["charge"]
    tol = default["tol"]

    b = cf.PyscfBasis(geometry, basis, charge=charge)
    b.pyscf_hartree_fock(tol=tol)

    if not restricted:
        b.from_restricted()

    return b

def run_cc(basis, CC, **kwargs):
    is_quadratic = False
    is_SD = False
    run_kwargs = {"tol": 1e-4, "vocal": False, "maxiters": 300}
    if not CC.__name__.startswith("Q"):
        run_kwargs = {"include_l": True}
    else:
        is_quadratic = True
    
    if "SD" in CC.__name__:
        is_SD = True

    cc = CC(basis)
    cc.mixer = cf.mix.SoftStartDIISMixer(alpha=0.75, start_DIIS_after=10, n_vectors=10)
    cc.run(**run_kwargs)

    W = {"0": 0.0,"S": 0.0,"D": 0.0,"T": 0.0,"Q": 0.0}
    if cc.t_info["converged"] and cc.l_info["converged"]:
        W["0"], W["D"] = cc.reference_weights(), 0.25*cc.doubles_weights().sum()

        if is_SD:
            W["S"] = cc.singles_weights().sum()
        
        if is_quadratic:
            W["Q"] = cc.quadruple_weight()
            if is_SD: 
                W["T"] = cc.triples_weight()
    else:
        print("Did not converge")
    
    return cc.energy(), W

def append_to_file(filename, W):
    fields = [key for key in W.keys()]
    row = [value for value in W.values()]

    filename = f"{PATH}/{filename}"
    is_emtpy = not os.path.exists(filename)

    with open(filename,'a') as file:
        writer = csv.DictWriter(file, fieldnames=fields)

        if is_emtpy:
            writer.writeheader()

        writer.writerow(W)

def float_format(x):
    if np.abs(x) > 0.01:
        return f"{x:.4f}"
    else:
        exponent = np.floor(np.log10(x))
        mantissa = x/10**exponent
        mantissa_format = str(mantissa)[0:4]
        return r"${0} \cdot 10^{{{1}}}$"\
            .format(mantissa_format, str(int(exponent)))


def atom_tests(run=False):
    atoms = ["He", "Be", "Ne"]

    geometries = [f"{atom} 0 0 0" for atom in atoms]
    basis = "cc-pVDZ"

    if run:
        for atom, geom in zip(atoms, geometries):
            _, W_fci = fci_pyscf(geom, basis)
            W_fci["atom"] = atom

            append_to_file("FCI_atoms.csv", W_fci)
            print(f"FCI {geom}")

            b = run_hf(geom, basis)
            _, W_ccsd = run_cc(b, cf.CCSD)
            W_ccsd["atom"] = atom
            
            append_to_file("CCSD_atoms.csv", W_ccsd)
            print(f"CCSD {geom}")

            _, W_qccsd = run_cc(b, cf.QCCSD)
            W_qccsd["atom"] = atom
            
            append_to_file("QCCSD_atoms.csv", W_qccsd)
            print(f"QCCSD {geom}")

    atom, W = "He", ["0", "S", "D"]
    methods = ["CCSD", "QCCSD", "FCI"]
    df_combined = pd.DataFrame(columns=[
        f"{method}{w}" for w in W for method in methods 
    ])

    for method in methods:
        df = pd.read_csv(f"dat/validate/{method}_atoms.csv", header=0)
        if "T" in df.columns or "Q" in df.columns:
            df.drop(columns=["T", "Q"], inplace=True)
    
        elm = df[df["atom"] == atom]
        
        for w in W:
            df_combined[method + w] = elm[w]
    
    formatters = [float_format for _ in range(len(df_combined.columns))]
    print(df_combined.to_latex(index=False, formatters=formatters))

def fix_stupied_write_error(string):
    string = string[1:-1]
    string = string.split()
    floats = [float(s) for s in string]

    return min(floats)

def dissociation_lih(run=False):
    distances = np.arange(2.5,10+0.1,0.25)
    geometries = [f"Li 0 0 0; H 0 0 {r}" for r in distances]
    basis = "cc-pVDZ"

    if run:
        for r, geom in zip(distances, geometries):
            E, W_fci = fci_pyscf(geom, basis)
            W_fci["r"] = r
            W_fci["E"] = E

            append_to_file("FCI_lih.csv", W_fci)
            print(f"FCI {geom}")

            b = run_hf(geom, basis)
            E, W_ccsd = run_cc(b, cf.CCSD)
            W_ccsd["r"] = r
            W_ccsd["E"] = E
            
            append_to_file("CCSD_lih.csv", W_ccsd)
            print(f"CCSD {geom}")

            E, W_qccsd = run_cc(b, cf.QCCSD)
            W_qccsd["r"] = r
            W_qccsd["E"] = E

            append_to_file("QCCSD_lih.csv", W_qccsd)
            print(f"QCCSD {geom}")

    fig, ax = plt.subplots()


    methods = ["FCI", "CCSD"]
    for method in methods:
        filename = f"dat/validate/{method}_lih.csv"

        df = pd.read_csv(filename, header=0)

        if method == "FCI":
            df["E"] = df["E"].map(fix_stupied_write_error)
            ax.scatter(df["r"], df["0"], s=15, marker="o", color="k", zorder=1, label="$W_0^{FCI}$")
            ax.scatter(df["r"], df["S"], s=15, marker="s", color="k", zorder=1, label="$W_S^{FCI}$")
            ax.scatter(df["r"], df["D"], s=15, marker="x", color="k", zorder=1, label="$W_D^{FCI}$")
        else:
            ax.plot(df["r"], df["0"], zorder=0, label="$W_0^{CCSD}$")
            ax.plot(df["r"], df["S"], zorder=0, label="$W_S^{CCSD}$")
            ax.plot(df["r"], df["D"], zorder=0, label="$W_D^{CCSD}$")

    ax.set(xlabel="R [$a_0$]", ylabel="$W_\mu$")
    ax.legend(ncols=2)
    pl.save("LIH_cc-pVDZ_dissociated_weigths")
    plt.show()

    fig, axes = plt.subplots(nrows=2, ncols=1, height_ratios=[6,3], sharex=True)
    methods = ["FCI", "QCCSD", "CCSD"]
    labels = ["FCI", "CCSD", "QCCSD"]
    ls = {"CCSD": "-", "QCCSD": (0, (2.5,2.5))}
    for method,label in zip(methods, labels):
        filename = f"dat/validate/{method}_lih.csv"

        df = pd.read_csv(filename, header=0)

        if method == "FCI":
            df["E"] = df["E"].map(fix_stupied_write_error)
            axes[0].scatter(df["r"], df["E"], s=15, marker="o", color="k", zorder=1, label="FCI")
            E_fci = df["E"].to_numpy()
        else:
            E_cc = df["E"].to_numpy()
            axes[0].plot(df["r"], E_cc, zorder=0, label=label, ls=ls[label])

            diff = np.abs(E_cc - E_fci)

            axes[1].plot(df["r"], diff, label=label)

    axes[0].set(ylabel=r"E")
    axes[0].legend()
    axes[1].set(xlabel="R [$a_0$]", ylabel=r"$|E - E_{FCI}|$")
    axes[1].set_yscale("log")
    axes[1].legend()
    axes[1].set_ylim(1e-5,1e-4)
    pl.save("LIH_cc-pVDZ_dissociated_energy")
    plt.show()


def dissociation_hf(run=False):
    distances = np.arange(1.0,5.00+0.1,0.125)
    geometries = [f"H 0 0 0; F 0 0 {r}" for r in distances]
    basis = "dzp"

    if run:
        for r, geom in zip(distances, geometries):
            E, W_fci = fci_pyscf(geom, basis)
            W_fci["r"] = r
            W_fci["E"] = E

            append_to_file("FCI_dzp.csv", W_fci)
            print(f"FCI {geom}")

            b = run_hf(geom, basis)
            E, W_ccsd = run_cc(b, cf.CCSD)
            W_ccsd["r"] = r
            W_ccsd["E"] = E
            
            append_to_file("CCSD_dzp.csv", W_ccsd)
            print(f"CCSD {geom}")

            E, W_qccsd = run_cc(b, cf.QCCSD)
            W_qccsd["r"] = r
            W_qccsd["E"] = E

            append_to_file("QCCSD_dzp.csv", W_qccsd)
            print(f"QCCSD {geom}")

def size_extensivity():
    geom1 = "H 0 0 0; H 0 0 1.4378925047;"
    geom2 = geom1 + "H 1000 0 0; H 1000 0 1.4378925047;"
    
    print(geom1)
    print(geom2)
    basis = "sto-3g"

    E1_fci, W1_fci = fci_pyscf(geom1, basis, nroots=2)
    E2_fci, W2_fci = fci_pyscf(geom2, basis, nroots=2)

    b1 = run_hf(geom1, basis)
    E1_cc, W1_cc = run_cc(b1, cf.CCSD)

    b2 = run_hf(geom2, basis)
    E2_cc, W2_cc = run_cc(b2, cf.CCSD)

    from IPython import embed
    embed()

if __name__ == "__main__":
    # atom_tests(run=False)

    # dissociation_lih(run=False)
    # dissociation_hf(run=False)

    size_extensivity()