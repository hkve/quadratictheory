import numpy as np
import clusterfock as cf
from utils.runs import run_fci_single, disassociate_2dof, disassociate_h2o
from plotting.plot_dissociation import plot
import plotting.plot_utils as pl
import matplotlib.pyplot as plt

import pathlib as pl
import pandas as pd

def run_fci(atoms, basis, vocal=False):
    E = np.zeros(len(atoms))
    for i, atom in enumerate(atoms):
        E[i] = run_fci_single(atom, basis)
        if vocal: print(f"FCI\tE = {E[i]:.4f}\t{atom = }\t{basis = }")
    
    return E

def merge_df(df1, df2):
    df_merged = pd.merge(df1, df2, on="r", how="outer", suffixes=(None, "_new"))
    cols_new = [col for col in df_merged.columns if col.endswith("_new")]
    cols_to_update = [col.rstrip("_new") for col in cols_new]

    for col, col_new in zip(cols_to_update, cols_new):
        if df_merged[col_new].isna().any():
            to_keep = np.argwhere(np.isnan(df_merged[col_new]))[:,0]
            df_merged[col_new].iloc[to_keep] = df_merged[col].iloc[to_keep]

    df_merged[cols_to_update] = df_merged[cols_new]
    df_merged.drop(columns=cols_new, inplace=True)

    return df_merged

def save(filename, df):
    filename = pl.Path(filename)
    df_new = None
    if not filename.exists():
        df_new = df
    else:
        df_stored = pd.read_csv(filename, sep=",", header=0, index_col=0)

        try:
            df_new = merge_df(df_stored, df)
        except:
            print(f"The format of {filename} does not match the passed dataframe format")
            print(f"Passed {df}")
            print(f"Stored {df_stored}")

            filename_new = filename.root + filename.stem + str(np.random.randint(1000, 9999, size=1)) + filename.suffix
            save(filename_new, df)

    df_new.to_csv(filename, sep=",", index=True)

def run_cc(atoms, basis, method, vocal=False, **kwargs):
    opts = {
        "tol": 1e-6,
        "maxiters": 200,
        "mixer": cf.mix.DIISMixer(n_vectors=20), 
    }

    opts.update(kwargs)

    tol_hf = 1e-8
    tol_cc = opts["tol"]

    E = np.zeros(len(atoms))
    for i, atom in enumerate(atoms):
        b = cf.PyscfBasis(atom, basis, restricted=True).pyscf_hartree_fock()

        # hf = cf.HF(b).run(tol=tol_hf)
        # if not hf.converged:
        #     print(f"Hartree-Fock did not converge at {tol_hf = }!")
        #     while not hf.converged:
        #         tol_hf *= 5
        #         hf = cf.HF(b).run(tol=tol_hf)
        #         if tol_hf > 0.1:
        #             print("Stopping iterations")
        #             break

        # b.change_basis(hf.C)
        b.from_restricted()

        cc = method(b)
        cc.run(tol=tol_cc, maxiters=opts["maxiters"])
        cc.mixer = opts["mixer"]

        if not cc.info["t_converged"]:
            print(f"{method.__name__} did not converge at {tol_cc = }!")
            while not cc.info["t_converged"]:
                tol_cc *= 5
                cc = method(b)
                cc.run(tol=tol_cc, maxiters=opts["maxiters"])
                if tol_cc > 1 and cc.info["t_converged"]:
                    print("Stopping iterations")
                    break
                if cc.info["t_converged"]:
                    print(f"Converged at {tol_cc = }")
                else:
                    print(f"Did not converge at {tol_cc = }\t{atom = }\t{basis = }")

        E[i] = cc.energy()
        if vocal: print(f"{method.__name__}\tE = {E[i]:.4f}\t{atom = }\t{basis = }")

    return E

def calculate_N2():
    distances1 = np.array([1.2, 1.4, 1.6, 1.7, 1.8, 1.9])
    distances2 = np.arange(2.0, 3.8+0.1, 0.1)

    distances = np.r_[distances1, distances2]

    distances = np.array([3.4])

    basis = "sto-3g"
    atoms = disassociate_2dof("N", "N", distances)

    E_fci = run_fci(atoms, basis, vocal=True)
    df_fci = pd.DataFrame({"r": distances, "FCI": E_fci})
    save("csv/N2.csv", df_fci)

    E_ccd = run_cc(atoms, basis, method=cf.CCD, vocal=True)
    df_ccd = pd.DataFrame({"r": distances, "CCD": E_ccd})
    save("csv/N2.csv", df_ccd)

    E_qccd = run_cc(atoms, basis, method=cf.QCCD, vocal=True)
    df_qccd = pd.DataFrame({"r": distances, "QCCD": E_qccd})
    save("csv/N2.csv", df_qccd)

def calculate_LiH():
    # distances = np.array([2.5, 3.0, 3.5, 4.0])
    # distances = np.array([1.0, 1.5, 2.0, 4.5, 5.0, 6.0])
    distances  = np.array([6.5, 7.0, 7.5, 8.0, 0.5, 9.0])

    basis = "sto-3g"
    atoms = disassociate_2dof("Li", "H", distances)

    E_fci = run_fci(atoms, basis, vocal=True)
    df_fci = pd.DataFrame({"r": distances, "FCI": E_fci})
    save("csv/LiH.csv", df_fci)

    E_ccd = run_cc(atoms, basis, method=cf.CCD, vocal=True)
    df_ccd = pd.DataFrame({"r": distances, "CCD": E_ccd})
    save("csv/LiH.csv", df_ccd)

    E_qccd = run_cc(atoms, basis, method=cf.QCCD, vocal=True)
    df_qccd = pd.DataFrame({"r": distances, "QCCD": E_qccd})
    save("csv/LiH.csv", df_qccd)

def calculate_HF():
    distances = np.arange(1.0, 5.5+0.125, 0.125)

    basis = "DZ"
    atoms = disassociate_2dof("H", "F", distances)

    E_fci = run_fci(atoms, basis, vocal=True)
    df_fci = pd.DataFrame({"r": distances, "FCI": E_fci})
    save("csv/HF.csv", df_fci)

    E_ccd = run_cc(atoms, basis, method=cf.CCD, vocal=True)
    df_ccd = pd.DataFrame({"r": distances, "CCD": E_ccd})
    save("csv/HF.csv", df_ccd)

    E_qccd = run_cc(atoms, basis, method=cf.QCCD, vocal=True)
    df_qccd = pd.DataFrame({"r": distances, "QCCD": E_qccd})
    save("csv/HF.csv", df_qccd)

def calculate_H2O():
    # distances = np.array([1.0, 1.3, 1.5, 1.7, 1.9, 2.1, 2.3, 2.5, 2.7, 2.8, 3.1, 3.3, 3.5, 3.7, 3.9, 4.1, 4.3, 4.5])
    # distances = np.array([1.0, 1.5, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 3.75, 4.00, 4.25, 4.50])
    # distances = np.array([1.0, 1.75, 2.50, 2.75, 3.30, 3.60, 4.00, 4.25, 4.50])
    distances = np.array([2.25, 3.0])
    basis = "sto-3g"
    atoms = disassociate_h2o(distances)

    E_fci = run_fci(atoms, basis, vocal=True)
    df_fci = pd.DataFrame({"r": distances, "FCI": E_fci})
    save("csv/H2O.csv", df_fci)

    E_ccd = run_cc(atoms, basis, method=cf.CCD, vocal=True)
    df_ccd = pd.DataFrame({"r": distances, "CCD": E_ccd})
    save("csv/H2O.csv", df_ccd)

    E_qccd = run_cc(atoms, basis, method=cf.QCCD, vocal=True)
    df_qccd = pd.DataFrame({"r": distances, "QCCD": E_qccd})
    save("csv/H2O.csv", df_qccd)    

def plot_N2():
    E_free = -107.43802235
    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(6,8), height_ratios=[6,3])
    plot("csv/N2.csv", axes, E0=E_free, splines=True, ylabel=True, x_min=1.61, y_max=0.2, save=True)
    plt.show()

def plot_LiH():
    E_free = -7.78249491
    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(6,8), height_ratios=[6,3])
    plot("csv/LiH.csv", axes, E0=E_free, splines=True, ylabel=True, x_min=1.4, y_min=-0.12, y_max=0.05, dash_quad=True)
    plt.show()

def plot_HF():
    E_free = -99.98329223
    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(6,8), height_ratios=[6,3])
    plot("csv/HF.csv", axes, E0=E_free, splines=True, ylabel=True, x_min=1.20, y_max=0.07, dash_quad=True)
    plt.show()

def plot_H2O():
    E_free = -74.73731393275897
    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(6,8), height_ratios=[6,3])
    plot("csv/H2O.csv", axes, E0=E_free, splines=True, ylabel=True, x_min=1.20, y_max=0.07, dash_quad=False, save=True)
    plt.show()


def main():
    # calculate_N2()
    plot_N2()

    # calculate_LiH()
    # plot_LiH()

    # calculate_HF()
    # plot_HF()

    # calculate_H2O()
    plot_H2O()

if __name__ == "__main__":
    main()