import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import clusterfock as cf

import pyscf
from pyscf.cc.ccd import CCD as pyscfCCD

def get_dissociation(df):
    for col in ["CCD", "QCCD", "VCCD"]:
        df[col] = df[col] + df.FCI

    E0 = df.FCI.iloc[-1]
    df_diss = df.copy()
    
    for col in ["CCD", "QCCD", "VCCD", "FCI"]:
        df_diss[col] = df_diss[col] - E0

    return df_diss, E0

def run_CCD_pyscf(atom, basis, tol=1e-8):
    basis = cf.PyscfBasis(atom, basis, restricted=True)
    hf_pyscf = pyscf.scf.HF(basis.mol).run(verbose=0, tol=tol)
    ccd_pyscf = pyscfCCD(hf_pyscf).run(verbose=0, tol=tol)

    print(f"{atom} hf? {hf_pyscf.converged} ccd? {ccd_pyscf.converged}")
    return ccd_pyscf.e_tot

def run_CCD(atom, basis, tol=1e-8):
    basis = cf.PyscfBasis(atom, basis, restricted=True)
    hf = cf.HF(basis).run()

    basis.change_basis(hf.C)
    basis.from_restricted()

    ccd = cf.CCD(basis).run(vocal=False, tol=tol, maxiters=100)

    energy = 0
    if not ccd.converged:
        energy = np.nan
    else:
        energy = ccd.energy()

    print(f"{atom}, hf? = {hf.converged} after {hf.iters}, ccd? {ccd.converged} after {ccd.iters}")
    return energy


def plot_N2():
    df = pd.read_csv("vanvoorhis_headgordon/n2.txt", sep=" ", skiprows=5, header=0, index_col=False)
    
    df_diss, E0 = get_dissociation(df)

    fig, ax = plt.subplots()
    for col in ["CCD", "FCI"]:
        ax.scatter(df_diss.R, df_diss[col], label=col)

    cf_CCD = np.zeros_like(df.R.to_numpy())*np.nan
    pyscf_CCD = np.zeros_like(df.R.to_numpy())*np.nan
    for i, r in enumerate(df_diss.R[:-2]):
        cf_CCD[i] = run_CCD(atom=f"N 0 0 0; N 0 0 {r}", basis="sto-3g") - E0
        pyscf_CCD[i] = run_CCD_pyscf(atom=f"N 0 0 0; N 0 0 {r}", basis="sto-3g") - E0

    ax.scatter(df_diss.R, cf_CCD, label="cf.CCD")
    ax.scatter(df_diss.R, pyscf_CCD, label="pyscf.CCD")
    ax.legend()
    ax.set(xlabel=r"$\Delta R$", ylabel=r"$E(N_2) - 2E(N)$")
    plt.show()


def plot_HF():
    df = pd.read_csv("vanvoorhis_headgordon/hf.txt", sep=" ", skiprows=5, header=0, index_col=False)

    df_diss = df    
    df_diss, E0 = get_dissociation(df)

    fig, ax = plt.subplots()
    for col in ["CCD", "FCI"]:
        ax.scatter(df_diss.R, df_diss[col], label=col)

    cf_CCD = np.zeros_like(df.R.to_numpy())*np.nan
    pyscf_CCD = np.zeros_like(df.R.to_numpy())*np.nan
    for i, r in enumerate(df_diss.R):
        cf_CCD[i] = run_CCD(atom=f"F 0 0 0; H 0 0 {r}", basis="DZ") - E0
        pyscf_CCD[i] = run_CCD_pyscf(atom=f"F 0 0 0; H 0 0 {r}", basis="DZ") - E0

    ax.scatter(df_diss.R, cf_CCD, label="cf.CCD")
    ax.scatter(df_diss.R, pyscf_CCD, label="pyscf.CCD")
    ax.legend()
    ax.set(xlabel=r"$\Delta R$", ylabel=r"$E(HF) - E(H) - E(F)$")
    plt.show()

if __name__ == '__main__':
    plot_N2()
    # plot_HF()