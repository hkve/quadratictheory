import clusterfock as cf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from utils.runs import run_fci_density_matrix

geometries = {
    "HF": "H 0 0 0; F 0 0 1.7325007863",
    "LiH": "Li 0 0 0; H 0 0 3.0141129518",
}


charges = {
    "HF": 0,
    "LiH": 0,
}
# geoms = {
#     "BH": "",
#     "CH": "",
#     "CN": "",
#     "OH": "",
#     "HF": "",
#     "LiH": "",
#     "NaH": "",
# }

# charge = {
#     "BH": 0,
#     "CH": 0,
#     "CN": 0,
#     "OH": 0,
#     "HF": 0,
#     "LiH": 0,
#     "NaH": 0,
# }

# In debye
experimental_dipole = {
    "HF": [1.827],
    "LiH": [5.880],
}

df_experimental_dipole = pd.DataFrame.from_dict(experimental_dipole, orient="index", columns=["dipole"])
df_experimental_dipole.reset_index(inplace=True)
df_experimental_dipole.rename(columns={"index":"molecule"}, inplace=True)

au_to_angstrom = lambda x: x/1.8897259886
angstrom_to_au = lambda x: x*1.8897259886
au_to_debye = lambda x: x*2.541746473
debye_to_au = lambda x: x/2.541746473
au_to_debye_angstrom = lambda x: au_to_angstrom(au_to_debye(x))
debye_angstrom_to_au = lambda x: angstrom_to_au(debye_to_au(x))


def get_run_kwargs(CC, **kwargs):
    run_kwargs = {"tol": 1e-6}
    
    if not "Q" in CC.__name__:
        run_kwargs["include_l"] = True

    return run_kwargs

def run_expvals(name, basis, CC):
    geom = geometries[name]
    charge = charges[name]

    b = cf.PyscfBasis(geom, basis, charge=charge, restricted=False).pyscf_hartree_fock()

    run_kwargs = get_run_kwargs(CC)
    cc = CC(b).run(**run_kwargs)

    cc.densities()

    delta_rho_ob = np.linalg.norm(cc.rho_ob  - cc.rho_ob.T)
    delta_rho_tb = np.linalg.norm(cc.rho_tb  - cc.rho_tb.transpose(2,3,0,1))
    dipole = cc.one_body_expval(b.r)
    quadropole = 0 #cc.two_body_expval(b.Q)

    print(f"Done {name}, {basis}, {CC.__name__}")

    return delta_rho_ob, delta_rho_tb, dipole, quadropole

def run_expvals_fci(name, basis):
    geom = geometries[name]
    charge = charges[name]

    b = cf.PyscfBasis(geom, basis, charge=charge, restricted=True).pyscf_hartree_fock()

    rho = run_fci_density_matrix(geom, basis)
    r = b.r

    dipole = np.einsum("...ij,ji->...", r, rho)    

    return dipole

def make_CC_table(names, basis, CC, convert_units=True):
    df = pd.DataFrame(columns=["molecule", "delta_rho_ob", "delta_rho_tb", "dipole"])
    for i, name in enumerate(names):
        delta_rho_ob, delta_rho_tb, dipole, quadropole = run_expvals(name, basis, CC)

        if convert_units:
            dipole = au_to_debye(dipole)

        df_new = pd.DataFrame({
            "molecule": [name],
            "delta_rho_ob": [delta_rho_ob],
            "delta_rho_tb": [delta_rho_tb],
            "dipole": [dipole[-1]],
        })
        
        df = pd.concat([df, df_new])

    df = df.reset_index().drop(columns="index")

    return df

def residual(df, add_experimental=True):
    df["dipole_residual"] = df["dipole"] - df_experimental_dipole["dipole"]

    if add_experimental:
        df_new = df_experimental_dipole.rename(columns={"dipole": "experimental_dipole"})
        df_new.drop(columns="molecule", inplace=True)
        df = pd.concat([df, df_new], axis=1)

    return df

def make_table():
    names = ["HF", "LiH"]
    basis = "cc-pVDZ"
    
    df_cc = make_CC_table(names, basis, cf.CCSD)
    df_qcc = make_CC_table(names, basis, cf.QCCSD)

    df_cc = residual(df_cc)
    df_qcc = residual(df_qcc)
    from IPython import embed
    embed()

if __name__ == "__main__":
    name = "HF"
    basis = "cc-pVDZ"
    
    print(
        run_expvals_fci(name, basis)
    )

    print(
        run_expvals(name, basis, cf.CCSD)
    )

    print(
        run_expvals(name, basis, cf.QCCSD)
    )
