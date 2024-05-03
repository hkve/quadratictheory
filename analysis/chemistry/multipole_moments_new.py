import clusterfock as cf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json

from utils.runs import run_fci_density_matrix

geometries = {
    "HF": "H 0 0 0; F 0 0 1.7325007863",
    "LiH": "Li 0 0 0; H 0 0 3.0141129518",
}


charges = {
    "HF": 0,
    "LiH": 0,
}

def get_run_kwargs(CC, **kwargs):
    run_kwargs = {"tol": 1e-8}
    
    if not "Q" in CC.__name__:
        run_kwargs["include_l"] = True

    return run_kwargs

def save_json(data, filename):
    data_json = json.dumps(data, indent=4)
    with open(f"json/{filename}.json", 'w') as file:
        file.write(data_json)

def load_json(filename):
    with open(f"json/{filename}.json", "r") as file:
        data_json = json.load(file)

    data = data_json.copy()

    for k, v in data_json.items():
        if type(v) == list:
            data[k] = np.array(v)

    return data

def format_tex(float_number):
    exponent = np.floor(np.log10(float_number))
    mantissa = float_number/10**exponent
    mantissa_format = str(mantissa)[0:5]
    return r"${0} \cdot 10^{{{1}}}$"\
           .format(mantissa_format, str(int(exponent)))

def run_expvals(name, basis, CC):
    geom = geometries[name]
    charge = charges[name]

    b = cf.PyscfBasis(geom, basis, charge=charge, restricted=False).pyscf_hartree_fock()

    run_kwargs = get_run_kwargs(CC)
    cc = CC(b).run(**run_kwargs)
    print(f"Done {name}, {basis}, {CC.__name__}")

    cc.densities()

    delta_rho_ob = np.linalg.norm(cc.rho_ob  - cc.rho_ob.T)
    delta_rho_tb = np.linalg.norm(cc.rho_tb  - cc.rho_tb.transpose(2,3,0,1))
    dipole = cc.one_body_expval(b.r)
    quadropole_e = cc.one_body_expval(b.Q)
    quadropole_nuc = b.Q_nuc()


    data = {
        "delta_rho_ob": delta_rho_ob,
        "delta_rho_tb": delta_rho_tb,
        "dipole": dipole.tolist(),
        "quadropole_e": quadropole_e.tolist(),
        "quadropole_nuc": quadropole_nuc.tolist(),
    }

    return data

def run_expvals_fci(name, basis):
    geom = geometries[name]
    charge = charges[name]

    b = cf.PyscfBasis(geom, basis, restricted=True, center=True).pyscf_hartree_fock()
    rho_ob = run_fci_density_matrix(geom, basis)
    print(f"Done {name}, {basis}, FCI")

    dipole = np.einsum("pq,...qp->...", rho_ob, b.r)
    rr = b.rr

    r2 = np.einsum("iipq->pq", rr)
    delta_ij_r2 = np.einsum("ij,pq->ijpq", np.eye(3), r2)
    Q_e = 0.5*(3*rr - delta_ij_r2)
    quadropole_e = np.einsum("pq,...qp->...", rho_ob, Q_e)
    quadropole_nuc = b.Q_nuc()

    data = {
        "dipole": dipole.tolist(),
        "quadropole_e": quadropole_e.tolist(),
        "quadropole_nuc": quadropole_nuc.tolist(),
    }

    return data

def compare_with_fci(run=False):
    names = ["LiH", "HF"]
    basis_sets = ["6-31g", "6-31g*"]

    if run:
        for basis in basis_sets:
            for name in names:
                data_ccsd = run_expvals(name, basis, cf.CCSD)
                save_json(data_ccsd, f"{name}_{basis}_ccsd")

                data_qccsd = run_expvals(name, basis, cf.QCCSD)
                save_json(data_qccsd, f"{name}_{basis}_qccsd")

                data_fci = run_expvals_fci(name, basis)
                save_json(data_fci, f"{name}_{basis}_fci")
    
    # value = "delta_rho_tb"
    # for name in names:
    #     methods = ["CCSD", "QCCSD"]
    #     table = {"met": methods + ["ratio"]}
    #     # "6-31g": [], "6-31g*": []

    #     for basis in basis_sets:
    #         delta_rho = []
    #         for method in methods:
    #             data = load_json(f"{name}_{basis}_{method.lower()}")
    #             delta_rho.append(data[value])

    #         ratio = delta_rho[0] / delta_rho[1]
    #         delta_rho.append(ratio)
    #         table[basis] = delta_rho
                    
    #     table_df = pd.DataFrame(table)

    #     for basis in basis_sets:
    #         table_df[basis].iloc[:-1] = table_df[basis].iloc[:-1].map(lambda x: format_tex(x))        
    #         table_df[basis].iloc[-1] = round(table_df[basis].iloc[-1], ndigits=3)        

    #     print(table_df.to_latex(index=False))


    for name in names:
        methods = ["CCSD", "QCCSD", "FCI"]
        table = {"met": methods}


        for basis in basis_sets:
            mu_z = []
            for method in methods:
                data = load_json(f"{name}_{basis}_{method.lower()}")
                mu_z.append(data["dipole"][2])

            mu_z = np.array(mu_z)
            mu_z[0:2] = mu_z[0:2] - mu_z[-1]
            mu_z[0:2] = 100 * mu_z[0:2]/mu_z[-1]
            table[basis] = mu_z
                    
        table_df = pd.DataFrame(table)
        
        print(table_df)

if __name__ == "__main__":
    compare_with_fci(run=False)
    # test()