import numpy as np
import quadratictheory as qt
from utils.runs import run_fci_single, disassociate_2dof, disassociate_h2o
from plotting.plot_dissociation import plot, plot_no_error, plot_correlation
import plotting.plot_utils as pu
import matplotlib.pyplot as plt
import pyscf

import pathlib as pl
import pandas as pd

def run_hf(atoms, basis, vocal=False):
    E = np.zeros(len(atoms))
    for i, atom in enumerate(atoms):
        mol = pyscf.gto.Mole()
        mol.unit = "bohr"
        mol.build(atom=atom, basis=basis)
        mol.verbose = 0
        mf = mol.HF().run()

        E[i] = mf.e_tot
        
        if vocal: print(f"HF\tE = {E[i]:.4f}\t{atom = }\t{basis = }")
    
    return E

def run_cisd(atoms, basis, vocal=False):
    E = np.zeros(len(atoms))
    for i, atom in enumerate(atoms):
        mol = pyscf.gto.Mole()
        mol.unit = "bohr"
        mol.build(atom=atom, basis=basis)
        mol.verbose = 0
        mf = mol.HF().run()
        cisd = mf.CISD()
        cisd.verbose = 0
        cisd  = cisd.run()

        E[i] = cisd.e_tot
        
        if vocal: print(f"CISD\tE = {E[i]:.4f}\t{atom = }\t{basis = }")
    
    return E

def run_fci(atoms, basis, vocal=False):
    E = np.zeros(len(atoms))
    for i, atom in enumerate(atoms):
        E[i] = run_fci_single(atom, basis)
        if vocal: print(f"FCI\tE = {E[i]:.4f}\t{atom = }\t{basis = }")
    
    return E

def merge_df(df1, df2):
    df1 = df1.round({"r": 3})
    df2 = df2.round({"r": 3})
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
        "tol": 1e-4,
        "maxiters": 400,
        "mixer": None,
        "hf_args": {"tol": 1e-4, "max_cycle": 750}
    }

    opts.update(kwargs)

    hf_args = opts["hf_args"]
    tol_cc = opts["tol"]
    mixer = opts["mixer"]

    E = np.zeros(len(atoms))
    for i, atom in enumerate(atoms):
        b = qt.PyscfBasis(atom, basis, restricted=False).pyscf_hartree_fock(**hf_args)

        cc = method(b)

        if mixer is not None: cc.mix = mixer

        cc.run(tol=tol_cc, maxiters=opts["maxiters"], vocal=True)

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
        if vocal: print(f"{method.__name__}\tE = {E[i]}\t{atom = }\t{basis = }")

    return E

"""
DOUBLES TRUNCATION --------------------------------------------------------
"""

def calculate_N2_ccd():

    distances = np.arange(1.25, 7.00+0.1, 0.25)

    basis = "sto-3g"
    atoms = disassociate_2dof("N", "N", distances)

    E_fci = run_fci(atoms, basis, vocal=True)
    df_fci = pd.DataFrame({"r": distances, "FCI": E_fci})
    save("csv/N2_ccd_test.csv", df_fci)

    ccd_mixer = qt.mix.SoftStartDIISMixer(alpha=0.90, start_DIIS_after=40, n_vectors=5)
    E_ccd = run_cc(atoms, basis, method=qt.CCD, vocal=True, mixer=ccd_mixer)
    df_ccd = pd.DataFrame({"r": distances, "CCD": E_ccd})
    save("csv/N2_ccd_test.csv", df_ccd)

    qccd_mixer = qt.mix.SoftStartDIISMixer(alpha=0.75, start_DIIS_after=7, n_vectors=5)
    E_qccd = run_cc(atoms, basis, method=qt.QCCD, vocal=True, mixer=qccd_mixer)
    df_qccd = pd.DataFrame({"r": distances, "QCCD": E_qccd})
    save("csv/N2_ccd_test.csv", df_qccd)
    

def calculate_LiH_ccd():
    # distances = np.array([2.5, 3.0, 3.5, 4.0])
    # distances = np.array([1.0, 1.5, 2.0, 4.5, 5.0, 6.0])
    distances  = np.array([6.5, 7.0, 7.5, 8.0, 0.5, 9.0])

    basis = "sto-3g"
    atoms = disassociate_2dof("Li", "H", distances)

    E_fci = run_fci(atoms, basis, vocal=True)
    df_fci = pd.DataFrame({"r": distances, "FCI": E_fci})
    save("csv/LiH_ccd.csv", df_fci)

    E_ccd = run_cc(atoms, basis, method=qt.CCD, vocal=True)
    df_ccd = pd.DataFrame({"r": distances, "CCD": E_ccd})
    save("csv/LiH_ccd.csv", df_ccd)

    E_qccd = run_cc(atoms, basis, method=qt.QCCD, vocal=True)
    df_qccd = pd.DataFrame({"r": distances, "QCCD": E_qccd})
    save("csv/LiH_ccd.csv", df_qccd)

def calculate_HF_ccd():
    distances = np.arange(1.0, 5.5+0.125, 0.125)

    basis = "DZ"
    atoms = disassociate_2dof("H", "F", distances)

    E_fci = run_fci(atoms, basis, vocal=True)
    df_fci = pd.DataFrame({"r": distances, "FCI": E_fci})
    save("csv/HF_ccd.csv", df_fci)

    E_ccd = run_cc(atoms, basis, method=qt.CCD, vocal=True)
    df_ccd = pd.DataFrame({"r": distances, "CCD": E_ccd})
    save("csv/HF_ccd.csv", df_ccd)

    E_qccd = run_cc(atoms, basis, method=qt.QCCD, vocal=True)
    df_qccd = pd.DataFrame({"r": distances, "QCCD": E_qccd})
    save("csv/HF_ccd.csv", df_qccd)

def calculate_H2O_ccd():
    # distances = np.arange(1.25, 7.00+0.1, 0.25)
    distances = np.r_[np.arange(1.25, 4.75+0.1, 0.25), np.arange(6.00, 7.00+0.1, 0.25)]
    basis = "sto-3g"
    atoms = disassociate_h2o(distances)

    # HF DOES NOT CONVERGE FOR R = 5.00 to 5.50

    # E_fci = run_fci(atoms, basis, vocal=True)
    # df_fci = pd.DataFrame({"r": distances, "FCI": E_fci})
    # save("csv/H2O_ccd.csv", df_fci)

    ccd_mixer = qt.mix.SoftStartDIISMixer(alpha=0.90, start_DIIS_after=40, n_vectors=5)
    E_ccd = run_cc(atoms, basis, method=qt.CCD, vocal=True, mixer=ccd_mixer)
    df_ccd = pd.DataFrame({"r": distances, "CCD": E_ccd})
    save("csv/H2O_ccd.csv", df_ccd)

    qccd_mixer = qt.mix.SoftStartDIISMixer(alpha=0.75, start_DIIS_after=10, n_vectors=10)
    E_qccd = run_cc(atoms, basis, method=qt.QCCD, vocal=True, mixer=qccd_mixer)
    df_qccd = pd.DataFrame({"r": distances, "QCCD": E_qccd})
    save("csv/H2O_ccd.csv", df_qccd)       

def plot_N2_ccd():
    E_free = -107.43802235
    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(6,8), height_ratios=[6,3])
    plot("csv/N2_ccd_test.csv", axes, E0=E_free, splines=True, ylabel=True, x_min=1.61, y_max=0.2, save_name="N2_diss_ccd")
    plt.show()

def plot_LiH_ccd():
    E_free = -7.78249491
    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(6,8), height_ratios=[6,3])
    plot("csv/LiH_ccd.csv", axes, E0=E_free, splines=True, ylabel=True, x_min=1.4, y_min=-0.12, y_max=0.05, dash_quad=True)
    plt.show()

def plot_HF_ccd():
    E_free = -99.98329223
    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(6,8), height_ratios=[6,3])
    plot("csv/HF_ccd.csv", axes, E0=E_free, splines=True, ylabel=True, x_min=1.20, y_max=0.07, dash_quad=True)
    plt.show()

def plot_H2O_ccd():
    E_free = -74.73731393275897
    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(6,8), height_ratios=[6,3])
    plot("csv/H2O_ccd.csv", axes, E0=E_free, splines=True, ylabel=True, x_min=1.20, y_max=0.07, dash_quad=False, save_name="H2O_diss_ccd")
    plt.show()

"""
SINGLES AND DOUBLES TRUNCATION --------------------------------------------------------
"""

def calculate_N2_ccsd():
    distances = np.array([4.75, 5.00, 5.25, 5.50, 5.75, 6.00])
    # distances = np.array([1.25, 1.5, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 3.75, 4.00, 4.25, 4.50])

    basis = "cc-pVTZ"
    atoms = disassociate_2dof("N", "N", distances)

    # E_hf = run_hf(atoms, basis, vocal=True)
    # df_hf = pd.DataFrame({"r": distances, "HF": E_hf})
    # save("csv/N2_ccsd_TZ.csv", df_hf)

    # E_cisd = run_cisd(atoms, basis, vocal=True)
    # df_cisd = pd.DataFrame({"r": distances, "CISD": E_cisd})
    # save("csv/N2_ccsd_TZ.csv", df_cisd)

    # E_ccsd = run_cc(atoms, basis, method=qt.CCSD, vocal=True)
    # df_ccsd = pd.DataFrame({"r": distances, "CCSD": E_ccsd})
    # save("csv/N2_ccsd_TZ.csv", df_ccsd)

    mixer = qt.mix.SoftStartDIISMixer(alpha=0.75, start_DIIS_after=8, n_vectors=8)
    E_qccsd = run_cc(atoms, basis, method=qt.QCCSD, vocal=True, mixer=mixer)
    df_qccsd = pd.DataFrame({"r": distances, "QCCSD": E_qccsd})
    save("csv/N2_ccsd_TZ.csv", df_qccsd)       

def calculate_H2O_ccsd():
    # distances = np.array([1.0, 1.3, 1.5, 1.7, 1.9, 2.1, 2.3, 2.5, 2.7, 2.8, 3.1, 3.3, 3.5, 3.7, 3.9, 4.1, 4.3, 4.5])
    distances = np.array([1.0, 1.5, 1.9, 2.25, 2.50, 3.00, 3.50, 4.00, 4.50, 5.00])
    basis = "sto-3g"
    atoms = disassociate_h2o(distances)

    E_hf = run_hf(atoms, basis, vocal=True)
    df_hf = pd.DataFrame({"r": distances, "HF": E_hf})
    save("csv/H2O_ccsd_TZ.csv", df_hf)

    E_cisd = run_cisd(atoms, basis, vocal=True)
    df_cisd = pd.DataFrame({"r": distances, "CISD": E_cisd})
    save("csv/H2O_ccsd_TZ.csv", df_cisd)

    E_ccsd = run_cc(atoms, basis, method=qt.CCSD, vocal=True)
    df_ccsd = pd.DataFrame({"r": distances, "CCSD": E_ccsd})
    save("csv/H2O_ccsd_TZ.csv", df_ccsd)

    E_qccsd = run_cc(atoms, basis, method=qt.QCCSD, vocal=True)
    df_qccsd = pd.DataFrame({"r": distances, "QCCSD": E_qccsd})
    save("csv/H2O_ccsd_TZ.csv", df_qccsd)       

def plot_N2_ccsd():
    plot_no_error("csv/N2_ccsd_TZ.csv", splines=True, y_max=-108.3, x_min=1.25)
    pu.save("N2_cc-pVTZ_dissociation_energy")
    plt.show()
    plot_correlation("csv/N2_ccsd_TZ.csv", splines=True)
    pu.save("N2_cc-pVTZ_dissociation_correlation_energy")
    plt.show()

def plot_H2O_ccsd():
    plot_no_error("csv/H2O_ccsd_TZ.csv", splines=False)
    plt.show()

def main():
    # calculate_N2_ccd()
    plot_N2_ccd()

    # calculate_LiH_ccd()
    # plot_LiH_ccd()

    # calculate_HF_ccd()
    # plot_HF_ccd()

    # calculate_H2O_ccd()
    # plot_H2O_ccd()

    # calculate_H2O_ccsd()
    # plot_H2O_ccsd()

    # calculate_N2_ccsd()
    # plot_N2_ccsd()

def test():
    geom, = disassociate_h2o(4.5)
    b = qt.PyscfBasis(geom, "cc-pVTZ").pyscf_hartree_fock()

    cc = qt.CCSD(b).run(tol=1e-6, include_l=True, vocal=True)

    print(
        cc.energy()
    )

    from pyscf import gto, scf, cc
    mol = gto.M(
        atom = geom,  # in Angstrom
        basis = 'ccpvtz',
        unit="bohr"
    )

    mf = scf.HF(mol).run()
    mycc = cc.CCSD(mf).run()
    print(mycc.e_tot)

if __name__ == "__main__":
    main()
    # test()