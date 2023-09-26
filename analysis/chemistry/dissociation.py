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

def run_CCD_pyscf(r):
    atom, basis = f"N 0 0 0; N 0 0 {r}", "sto-3g"

    basis = cf.PyscfBasis(atom, basis, restricted=True)
    hf_pyscf = pyscf.scf.HF(basis.mol).run(verbose=0, tol=1e-8)
    ccd_pyscf = pyscfCCD(hf_pyscf).run(verbose=0, tol=1e-8)

    print(f"r = {r} hf? {hf_pyscf.converged} after {hf_pyscf.iter} ccd? {ccd_pyscf.converged}")
    return ccd_pyscf.e_tot

def run_CCD(r):
    atom, basis = f"N 0 0 0; N 0 0 {r}", "sto-3g"

    basis = cf.PyscfBasis(atom, basis, restricted=True)
    hf = cf.HF(basis).run()

    basis.change_basis(hf.C)
    basis.from_restricted()

    ccd = cf.CCD(basis).run(vocal=False, tol=1e-8, maxiters=100)

    energy = 0
    if not ccd.converged:
        energy = np.nan
    else:
        energy = ccd.energy()

    print(f"{r = }, hf? = {hf.converged} after {hf._iters}, ccd? {ccd.converged} after {ccd.iters}")
    return energy


def plot_N2():
    df = pd.read_csv("vanvoorhis_headgordon/n2.txt", sep=" ", skiprows=5, header=0, index_col=False)
    
    df_diss, E0 = get_dissociation(df)

    fig, ax = plt.subplots()
    for col in ["CCD", "QCCD", "VCCD", "FCI"]:
        ax.scatter(df_diss.R, df_diss[col], label=col)

    my_CCD = np.zeros_like(df.R.to_numpy())
    for i, r in enumerate(df_diss.R):
        my_CCD[i] = run_CCD_pyscf(r) - E0

    ax.scatter(df_diss.R, my_CCD, label="fidosfos")
    ax.legend()
    plt.show()

if __name__ == '__main__':
    plot_N2()