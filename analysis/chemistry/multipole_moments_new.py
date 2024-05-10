import clusterfock as cf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json

from utils.runs import run_fci_density_matrix

geometries = {
    "HF": "H 0 0 0; F 0 0 1.7325007863",
    "LiH": "Li 0 0 0; H 0 0 3.0141129518",
    "CH+": "C 0.0 0.0 0.0; H 0.0 0.0 2.13713",
    "BeO": "Be 0 0 0; O 0 0 2.5269415919",
    "N2": "N 0 0 0; N 0 0 2.1013752993",
    "H2O": "O 0 0 0.2286568446; H 0 1.4191842174 -0.9146273785; H 0 -1.4191842174 -0.9146273785;",
    "BeH2": "Be 0 0 0; H 0 0 -2.5265636467; H 0 0 2.5265636467",
}


charges = {
    "HF": 0,
    "LiH": 0,
    "CH+": 1,
    "BeO": 0,
    "N2": 0,
    "H2O": 0,
    "BeH2": 0,
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

    b = cf.PyscfBasis(geom, basis, restricted=False, center=True).pyscf_hartree_fock()
    rho_ob = run_fci_density_matrix(geom, basis)
    print(f"Done {name}, {basis}, FCI")

    dipole = np.einsum("pq,...qp->...", rho_ob, b.r)
    
    quadropole_e = np.einsum("pq,...qp->...", rho_ob, b.Q)
    quadropole_nuc = b.Q_nuc()

    data = {
        "dipole": dipole.tolist(),
        "quadropole_e": quadropole_e.tolist(),
        "quadropole_nuc": quadropole_nuc.tolist(),
    }

    return data

def density_matrix_asymmetry(run=False):
    names = ["HF", "LiH", "CH+", "BeO", "N2", "H2O", "BeH2"]
    basis = "cc-pVDZ"

    if run:
        for name in names:
            data_ccsd = run_expvals(name, basis, cf.CCSD)
            save_json(data_ccsd, f"{name}_{basis}_ccsd")
            print(f"DONE CCSD {name}")

            data_qccsd = run_expvals(name, basis, cf.QCCSD)
            save_json(data_qccsd, f"{name}_{basis}_qccsd")
            print(f"DONE QCCSD {name}")

    table = {"CCSD1": [], "QCCSD1": [], "CCSD2": [], "QCCSD2": []}
    for name in names:
        methods = ["CCSD", "QCCSD"]

        for method in methods:
            data = load_json(f"{name}_{basis}_{method.lower()}")
            table[f"{method}1"].append(data["delta_rho_ob"])
            table[f"{method}2"].append(data["delta_rho_tb"])

    table_df = pd.DataFrame(table)
    table_df.insert(loc=0, column="basis", value=names)

    table_df["ratio1"] = table_df["CCSD1"]/table_df["QCCSD1"]
    table_df["ratio2"] = table_df["CCSD2"]/table_df["QCCSD2"]

    for col in ["CCSD1", "CCSD2", "QCCSD1", "QCCSD2"]:
        table_df[col] = table_df[col].map(lambda x: format_tex(x))        
    for col in ["ratio1", "ratio2"]:
        table_df[col] = table_df[col].map(lambda x: f"{x:.2f}")      

    print(table_df.to_latex(index=False))


def compare_with_fci(run=False):
    names = ["LiH"]
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
    

    name = "LiH"
    table = {"CCSD1": [], "QCCSD1": [], "CCSD2": [], "QCCSD2": []}
    for basis in basis_sets:
        methods = ["CCSD", "QCCSD"]

        for method in methods:
            data = load_json(f"{name}_{basis}_{method.lower()}")
            table[f"{method}1"].append(data["delta_rho_ob"])
            table[f"{method}2"].append(data["delta_rho_tb"])


    table_df = pd.DataFrame(table)
    table_df.insert(loc=0, column="basis", value=basis_sets)

    table_df["ratio1"] = table_df["CCSD1"]/table_df["QCCSD1"]
    table_df["ratio2"] = table_df["CCSD2"]/table_df["QCCSD2"]

    for col in ["CCSD1", "CCSD2", "QCCSD1", "QCCSD2"]:
        table_df[col] = table_df[col].map(lambda x: format_tex(x))        
    for col in ["ratio1", "ratio2"]:
        table_df[col] = table_df[col].map(lambda x: f"{x:.2f}")      

    print(table_df.to_latex(index=False))


    # for name in names:
    #     methods = ["CCSD", "QCCSD", "FCI"]
    #     table = {"met": methods}


    #     for basis in basis_sets:
    #         mu_z = []
    #         for method in methods:
    #             data = load_json(f"{name}_{basis}_{method.lower()}")
    #             mu_z.append(data["dipole"][2])

    #         mu_z = np.array(mu_z)
    #         mu_z[0:2] = mu_z[0:2] - mu_z[-1]
    #         # mu_z[0:2] = 100 * mu_z[0:2]/mu_z[-1]
    #         table[basis] = mu_z
                    
    #     table_df = pd.DataFrame(table)
        
    #     print(table_df)


def partial_expvals(rho, A, o, v, name, fci_reslts=None):
    oo = np.einsum("...pq,pq->...", A[:,o, o], rho[o, o])[2]
    vv = np.einsum("...pq,pq->...", A[:,v, v], rho[v, v])[2]
    ov = np.einsum("...pq,pq->...", A[:,o, v], rho[o, v])[2]
    vo = np.einsum("...pq,pq->...", A[:,v, o], rho[v, o])[2]

    if fci_reslts == None:
        print(f"""
            {name}
            {oo = }
            {vv = }
            {ov = }
            {vo = }
            sum = {oo+vv+ov+vo} 
            """)
    else:
        oo_fci, vv_fci, ov_fci, vo_fci = fci_reslts
        total_fci = oo_fci+vv_fci+ov_fci+vo_fci
        total_cc = oo+vv+ov+vo
        total_diff = total_cc - total_fci
        print(f"""
            {name}
            {oo = } \t deviation = {oo-oo_fci:.4e} \t {100 * (oo-oo_fci)/total_diff}
            {vv = } \t deviation = {vv-vv_fci:.4e} \t {100 * (vv-vv_fci)/total_diff}
            {ov = } \t deviation = {ov-ov_fci:.4e} \t {100 * (ov-ov_fci)/total_diff}
            {vo = } \t deviation = {vo-vo_fci:.4e} \t {100 * (vo-vo_fci)/total_diff}
            sum = {total_cc} \t deviation {total_diff}
            """)

    return oo, vv, ov, vo

def run_density_test():
    basis = "6-31g"
    geom = geometries["LiH"]

    fci = run_fci_density_matrix(geom, basis)

    b = cf.PyscfBasis(geom, basis, restricted=False).pyscf_hartree_fock(tol=1e-10)

    cc = cf.CCSD(b).run(tol=1e-10, include_l=True)
    qcc = cf.QCCSD(b).run(tol=1e-10)


    ccsd = cc.one_body_density()
    qccsd = qcc.one_body_density()

    # qccsd[b.o,b.o] = ccsd[b.o,b.o]
    # qccsd[b.v,b.v] = ccsd[b.v,b.v]
    # qccsd[b.o,b.v] = ccsd[b.o,b.v]
    # qccsd[b.v,b.o] = ccsd[b.v,b.o]


    diff_cc = ccsd - fci
    diff_qcc = qccsd - fci

    # b = cf.PyscfBasis(geom, basis, restricted=False)

    def print_diff(diff, rho, name):
        print(f"""
        {name}
        diff:
            norm = {np.linalg.norm(diff)}
            max = {np.max(diff)}
            mean = {np.mean(diff)}
            mean abs = {np.mean(np.abs(diff))}
        rho:
            asym norm = {np.linalg.norm(rho - rho.T)}
            asym max = {np.max(rho - rho.T)}
            asym mean = {np.mean(rho - rho.T)}
            """)

    fci_occ = fci[:b.N,:b.N]
    ccsd_occ = ccsd[:b.N,:b.N]
    qccsd_occ = qccsd[:b.N,:b.N]

    print_diff(diff_cc, ccsd, "CCSD")
    print_diff(diff_qcc, qccsd, "QCCSD")

    r_qcc = np.einsum("...pq,pq->...", b.r, qccsd)
    r_cc = np.einsum("...pq,pq->...", b.r, ccsd)
    r_fci = np.einsum("...pq,pq->...", b.r, fci)

    print(f"""
        mu(z) FCI: {r_fci[2]}
        mu(z) CCSD: {r_cc[2]} \t diff {r_cc[2] - r_fci[2]}
        mu(z) QCCSD: {r_qcc[2]} \t diff {r_qcc[2] - r_fci[2]}
    """)

    print("Partitioned expval into blocks")
    fci_partial = partial_expvals(fci, b.r, b.o, b.v, "FCI")
    partial_expvals(ccsd, b.r, b.o, b.v, "CCSD", fci_partial)
    partial_expvals(qccsd, b.r, b.o, b.v, "QCCSD", fci_partial)

    # fci_diag = np.diag( b.r[2,...] @ fci)
    # cc_diag = np.diag( b.r[2,...] @ ccsd)
    # qcc_diag = np.diag( b.r[2,...] @ qccsd)

    fig, ax = plt.subplots(nrows=1, ncols=2)

    vmin, vmax = min([diff_cc.min(),diff_qcc.min()]), max([diff_cc.max(),diff_qcc.max()])
    vabs = max(abs(vmin), abs(vmax))

    im1 = ax[0].imshow(diff_cc, vmin=-vabs, vmax=vabs, cmap="seismic")
    im2 = ax[1].imshow(diff_qcc, vmin=-vabs, vmax=vabs, cmap="seismic")

    ax[0].hlines(b.N-0.5, -0.5, b.L-0.5, color="k")
    ax[0].vlines(b.N-0.5, -0.5, b.L-0.5, color="k")

    ax[1].hlines(b.N-0.5, -0.5, b.L-0.5, color="k")
    ax[1].vlines(b.N-0.5, -0.5, b.L-0.5, color="k")

    ax[0].set_title(r"$\rho^{CCSD} - \rho^{FCI}$")
    ax[1].set_title(r"$\rho^{QCCSD} - \rho^{FCI}$")

    cbar = fig.colorbar(im1, ax=ax.ravel().tolist())
    
    plt.show()
def run_expval_test():
    basis = "3-21g*"
    name = "HF"

    fci = run_expvals_fci(name, basis)
    ccsd = run_expvals(name, basis, cf.CCSD)
    qccsd = run_expvals(name, basis, cf.QCCSD)

    from IPython import embed
    embed()

if __name__ == "__main__":
    # density_matrix_asymmetry(run=False)
    # compare_with_fci(run=False)
    run_density_test()
    # run_expval_test()