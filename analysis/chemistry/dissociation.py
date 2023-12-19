import numpy as np
import clusterfock as cf
from fci_utils import run_fci_single

import pathlib as pl
import pandas as pd

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

def make_N2(distances):
    atoms = [f"N 0 0 0; N 0 0 {r}" for r in distances]
    return atoms

def run_fci(atoms, basis, vocal=False):
    E = np.zeros(len(atoms))
    for i, atom in enumerate(atoms):
        E[i] = run_fci_single(atom, basis)
        if vocal: print(f"FCI\tE = {E[i]:.4f}\t{atom = }\t{basis = }")
    return E

def run_cc(atoms, basis, method, vocal=False, **kwargs):
    opts = {
        "tol": 1e-8,
        "maxiters": 200,
        "mixer": cf.mix.DIISMixer(n_vectors=20), 
    }

    opts.update(kwargs)

    tol_hf = 1e-8
    tol_cc = opts["tol"]

    E = np.zeros(len(atoms))
    for i, atom in enumerate(atoms):
        b = cf.PyscfBasis(atom, basis, restricted=True)

        hf = cf.HF(b).run(tol=tol_hf)
        if not hf.converged:
            print(f"Hartree-Fock did not converge at {tol_hf = }!")
            while not hf.converged:
                tol_hf *= 5
                hf = cf.HF(b).run(tol=tol_hf)
                if tol_hf > 0.1:
                    print("Stopping iterations")
                    break

        b.change_basis(hf.C)
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
    # distances = np.array([1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4])
    distances = np.array([0.6, 0.8])

    basis = "sto-3g"
    atoms = make_N2(distances)

    E_fci = run_fci(atoms, basis, vocal=True)
    df_fci = pd.DataFrame({"r": distances, "FCI": E_fci})
    save("N2.csv", df_fci)

    E_ccd = run_cc(atoms, basis, method=cf.CCD, vocal=True)
    df_ccd = pd.DataFrame({"r": distances, "CCD": E_ccd})
    save("N2.csv", df_ccd)

    E_qccd = run_cc(atoms, basis, method=cf.QCCD, vocal=True)
    df_qccd = pd.DataFrame({"r": distances, "QCCD": E_qccd})
    save("N2.csv", df_qccd)


def main():
    calculate_N2()

if __name__ == "__main__":
    main()