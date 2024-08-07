import quadratictheory as qt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json

import plotting.plot_utils as pl
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

    b = qt.PyscfBasis(geom, basis, charge=charge, restricted=False).pyscf_hartree_fock()

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

    b = qt.PyscfBasis(geom, basis, restricted=False, center=True).pyscf_hartree_fock()
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
            data_ccsd = run_expvals(name, basis, qt.CCSD)
            save_json(data_ccsd, f"{name}_{basis}_ccsd")
            print(f"DONE CCSD {name}")

            data_qccsd = run_expvals(name, basis, qt.QCCSD)
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
    table_df.insert(loc=0, column="sys", value=names)

    print(table_df)
    ### Plot
    fig, ax = plt.subplots()

    markers = ["." , "s" , "p" , "P" , "*" , "x", "d"]

    table_df.at[table_df['sys'].tolist().index('CH+'),'sys'] = "CH$^+$"
    table_df.at[table_df['sys'].tolist().index('H2O'),'sys'] = "H$_2$O"
    table_df.at[table_df['sys'].tolist().index('BeH2'),'sys'] = "BeH$_2$"
    table_df.at[table_df['sys'].tolist().index('N2'),'sys'] = "N$_2$"
    print(table_df)

    ax.scatter(None, None, label="One-body", color=pl.colors[1])
    for i, row in enumerate(table_df.itertuples()):
        ax.scatter(row.QCCSD1, row.CCSD1, color=pl.colors[1], marker=markers[i])
        ax.scatter(row.QCCSD2, row.CCSD2, color=pl.colors[0], marker=markers[i])

        ax.scatter(None, None, label=row.sys, marker=markers[i], color="k")
        if i == 3:
            ax.scatter(None, None, label="Two-body", color=pl.colors[0])


    ax.plot((0,1.2),(0,1.2), color="gray", ls="--", alpha=0.5)
    ax.plot((0,0.12),(0,1.2), color="gray", ls="--", alpha=0.5)
    ax.plot((0,0.012),(0,1.2), color="gray", ls="--", alpha=0.5)
    ax.set(xlim=(1e-5,1.2), ylim=(1e-5,1.2), ylabel="$N_i$ (CCSD)", xlabel="$N_i$ (QCCSD)")
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(ncol=2)
    fig.savefig("test.pdf", transparent=True)
    plt.show()

    ###


    table_df["ratio1"] = table_df["CCSD1"]/table_df["QCCSD1"]
    table_df["ratio2"] = table_df["CCSD2"]/table_df["QCCSD2"]

    for col in ["CCSD1", "CCSD2", "QCCSD1", "QCCSD2"]:
        table_df[col] = table_df[col].map(lambda x: format_tex(x))        
    for col in ["ratio1", "ratio2"]:
        table_df[col] = table_df[col].map(lambda x: f"{x:.2f}")      

    # print(table_df.to_latex(index=False))



def compare_with_fci(run=False):
    names = ["LiH", "HF"]
    basis_sets = ["6-31g", "6-31g*", "cc-pVDZ"] #["6-31g", "6-31g*"]

    if run:
        for basis in basis_sets:
            for name in names:
                data_ccsd = run_expvals(name, basis, qt.CCSD)
                save_json(data_ccsd, f"{name}_{basis}_ccsd")

                data_qccsd = run_expvals(name, basis, qt.QCCSD)
                save_json(data_qccsd, f"{name}_{basis}_qccsd")

                data_fci = run_expvals_fci(name, basis)
                save_json(data_fci, f"{name}_{basis}_fci")
    
    # Dipole and Quadropole combined
    # table = {"name": [], "basis": [], "CCSDm": [], "QCCSDm": [], "FCIm": [], "CCSDq": [], "QCCSDq": [], "FCIq": []}
    # for name in names:
    #     for basis in basis_sets:
    #         for method in ["CCSD", "QCCSD", "FCI"]:
    #             data = load_json(f"{name}_{basis}_{method.lower()}")
    #             table[f"{method}m"].append(data["dipole"][2])
    #             table[f"{method}q"].append(data["quadropole_e"][2][2])
                        
    #         table["name"].append(name)
    #         table["basis"].append(basis)

    # table_df = pd.DataFrame(table)

    # for method in ["CCSD", "QCCSD"]:
    #     table_df[f"{method}m"] = 1000*(table_df[f"{method}m"] - table_df[f"FCIm"])
    #     table_df[f"{method}q"] = 1000*(table_df[f"{method}q"] - table_df[f"FCIq"])

    #     table_df[f"{method}m"] = table_df[f"{method}m"].map(lambda x: f"{x:.3f}")
    #     table_df[f"{method}q"] = table_df[f"{method}q"].map(lambda x: f"{x:.3f}")

    # print(table_df.to_latex(index=False))

    # Just one
    expval = "dipole"
    table = {"name": [], "basis": [], "CCSD": [], "QCCSD": [], "FCI": []}
    for name in names:
        for basis in basis_sets:
            for method in ["CCSD", "QCCSD", "FCI"]:
                data = load_json(f"{name}_{basis}_{method.lower()}")
                if expval == "dipole":
                    table[f"{method}"].append(data["dipole"][2])
                if expval == "quadropole_e":
                    table[f"{method}"].append(data["quadropole_e"][2][2])
                        
            table["name"].append(name)
            table["basis"].append(basis)

    table_df = pd.DataFrame(table)
    print(table_df)
    for method in ["CCSD", "QCCSD"]:
        table_df[f"{method}d"] = 1000*(table_df[f"{method}"] - table_df[f"FCI"])

        table_df[f"{method}d"] = table_df[f"{method}d"].map(lambda x: f"{x:.3f}")

    print(table_df.to_latex(index=False))


def partial_expvals(rho, A, o, v, name, fci_reslts=None, vocal=True):
    oo = np.einsum("...pq,pq->...", A[:,o, o], rho[o, o])[2]
    vv = np.einsum("...pq,pq->...", A[:,v, v], rho[v, v])[2]
    ov = np.einsum("...pq,pq->...", A[:,o, v], rho[o, v])[2]
    vo = np.einsum("...pq,pq->...", A[:,v, o], rho[v, o])[2]

    if fci_reslts == None and vocal:
        print(f"""
            {name}
            {oo = }
            {vv = }
            {ov = }
            {vo = }
            sum = {oo+vv+ov+vo} 
            """)
    elif vocal:
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

def run_density_diff(run=False, show=False):
    # names = ["LiH", "HF"]
    basis_sets = ["6-31g", "6-31g*", "cc-pVDZ"]
    names = ["LiH"]
    # basis_sets = ["6-31g*", "cc-pVDZ"]

    folder = "dat/density"
    if run:
        for name in names:
            geom = geometries[name]

            for basis in basis_sets:
                b = qt.PyscfBasis(geom, basis, restricted=False).pyscf_hartree_fock(tol=1e-10)
                fci = run_fci_density_matrix(geom, basis)
                np.savez(f"{folder}/density_FCI_{name}_{basis}_test", fci)
                print(f"Done FCI {name} {basis}")

                cc = qt.CCSD(b).run(tol=1e-10, include_l=True)
                ccsd = cc.one_body_density()
                np.savez(f"{folder}/density_CCSD_{name}_{basis}_test", ccsd)
                print(f"Done CCSD {name} {basis}")


                qcc = qt.QCCSD(b).run(tol=1e-10)
                qccsd = qcc.one_body_density()
                np.savez(f"{folder}/density_QCCSD_{name}_{basis}_test", qccsd)
                print(f"Done QCCSD {name} {basis}")

    if show:
        for name in names:
            geom = geometries[name]
            for basis in basis_sets:
                b = qt.PyscfBasis(geom, basis, restricted=False).pyscf_hartree_fock(tol=1e-10)
                fci = np.load(f"{folder}/density_FCI_{name}_{basis}_test.npz", allow_pickle=True)["arr_0"]
                ccsd = np.load(f"{folder}/density_CCSD_{name}_{basis}_test.npz", allow_pickle=True)["arr_0"]
                qccsd = np.load(f"{folder}/density_QCCSD_{name}_{basis}_test.npz", allow_pickle=True)["arr_0"]

                diff_cc = ccsd - fci
                diff_qcc = qccsd - fci

                fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8,4), layout="constrained")

                vmin, vmax = min([diff_cc.min(),diff_qcc.min()]), max([diff_cc.max(),diff_qcc.max()])
                vabs = max(abs(vmin), abs(vmax))

                im1 = ax[0].imshow(diff_cc, vmin=-vabs, vmax=vabs, cmap="seismic")
                im2 = ax[1].imshow(diff_qcc, vmin=-vabs, vmax=vabs, cmap="seismic")

                ax[0].hlines(b.N-0.5, -0.5, b.L-0.5, color="k")
                ax[0].vlines(b.N-0.5, -0.5, b.L-0.5, color="k")

                ax[1].hlines(b.N-0.5, -0.5, b.L-0.5, color="k")
                ax[1].vlines(b.N-0.5, -0.5, b.L-0.5, color="k")

                ax[0].set_title(r"$\gamma^{CCSD} - \gamma^{FCI}$")
                ax[1].set_title(r"$\gamma^{QCCSD} - \gamma^{FCI}$")

                cbar = fig.colorbar(im1, ax=ax.ravel().tolist(), pad=0.025)
                
                ytickslocs0 = ax[0].get_yticks()[1:-1]
                ytickslocs1 = ax[1].get_yticks()[1:-1]
                ax[0].set_xticks(ytickslocs0)
                ax[1].set_xticks(ytickslocs1)

                pl.save(f"density_diff_{name}_{basis}")
                plt.show()

    prefix = 1000
    table = {"name": [], "basis": [], "CCSDoo": [], "CCSDvv": [], "CCSDov": [], "CCSDvo": [],"QCCSDoo": [], "QCCSDvv": [], "QCCSDov": [], "QCCSDvo": []}
    for name in names:
        geom = geometries[name]
        for basis in basis_sets:
            b = qt.PyscfBasis(geom, basis, restricted=False).pyscf_hartree_fock(tol=1e-10)
            fci = np.load(f"{folder}/density_FCI_{name}_{basis}.npz", allow_pickle=True)["arr_0"]
            ccsd = np.load(f"{folder}/density_CCSD_{name}_{basis}.npz", allow_pickle=True)["arr_0"]
            qccsd = np.load(f"{folder}/density_QCCSD_{name}_{basis}.npz", allow_pickle=True)["arr_0"]
    
            oo_fci, vv_fci, ov_fci, vo_fci = partial_expvals(fci, b.r, b.o, b.v, name, vocal=False)
            
            oo, vv, ov, vo = partial_expvals(ccsd, b.r, b.o, b.v, name, vocal=False)
            diff_oo, diff_vv, diff_ov, diff_vo = oo-oo_fci, vv-vv_fci, ov-ov_fci, vo-vo_fci

            table["CCSDoo"].append(prefix*diff_oo)
            table["CCSDvv"].append(prefix*diff_vv)
            table["CCSDov"].append(prefix*diff_ov)
            table["CCSDvo"].append(prefix*diff_vo)

            oo, vv, ov, vo = partial_expvals(qccsd, b.r, b.o, b.v, name, vocal=False)
            diff_oo, diff_vv, diff_ov, diff_vo = oo-oo_fci, vv-vv_fci, ov-ov_fci, vo-vo_fci

            table["QCCSDoo"].append(prefix*diff_oo)
            table["QCCSDvv"].append(prefix*diff_vv)
            table["QCCSDov"].append(prefix*diff_ov)
            table["QCCSDvo"].append(prefix*diff_vo)
        
            table["basis"].append(basis)
            table["name"].append(name)

    df_table = pd.DataFrame(table)
    print(df_table)
    for method in ["CCSD", "QCCSD"]:
        for block in ["oo", "vv", "ov", "vo"]:
            df_table[f"{method}{block}"] = df_table[f"{method}{block}"].map(lambda x: f"{x:.3f}")

    print(
        df_table.to_latex(index=False)
    )

    table = {"name": [], "basis": [], "CCSD": [], "QCCSD": []}
    for name in names:
        geom = geometries[name]
        for basis in basis_sets:
            b = qt.PyscfBasis(geom, basis, restricted=False).pyscf_hartree_fock(tol=1e-10)
            fci = np.load(f"{folder}/density_FCI_{name}_{basis}.npz", allow_pickle=True)["arr_0"]
            ccsd = np.load(f"{folder}/density_CCSD_{name}_{basis}.npz", allow_pickle=True)["arr_0"]
            qccsd = np.load(f"{folder}/density_QCCSD_{name}_{basis}.npz", allow_pickle=True)["arr_0"]

            ccsd_diff = np.sum(np.abs(ccsd - fci))
            qccsd_diff = np.sum(np.abs(qccsd - fci))

            table["CCSD"].append(ccsd_diff)
            table["QCCSD"].append(qccsd_diff)
            table["name"].append(name)
            table["basis"].append(basis)

    df_table = pd.DataFrame(table)

    for col in ["CCSD", "QCCSD"]:
        df_table[col] = df_table[col].map(lambda x: f"{x:.4f}")

    print(
        df_table.to_latex(index=False)
    )
def run_density_test():
    basis = "6-31g"
    geom = geometries["HF"]

    fci = run_fci_density_matrix(geom, basis)

    b = qt.PyscfBasis(geom, basis, restricted=False).pyscf_hartree_fock(tol=1e-10)

    cc = qt.CCSD(b).run(tol=1e-10, include_l=True)
    qcc = qt.QCCSD(b).run(tol=1e-10)

    ccsd = cc.one_body_density()
    qccsd = qcc.one_body_density()

    # qccsd[b.o,b.o] = ccsd[b.o,b.o]
    # qccsd[b.v,b.v] = ccsd[b.v,b.v]
    # qccsd[b.o,b.v] = ccsd[b.o,b.v]
    # qccsd[b.v,b.o] = ccsd[b.v,b.o]


    diff_cc = ccsd - fci
    diff_qcc = qccsd - fci

    # b = qt.PyscfBasis(geom, basis, restricted=False)

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
    ccsd = run_expvals(name, basis, qt.CCSD)
    qccsd = run_expvals(name, basis, qt.QCCSD)

    from IPython import embed
    embed()

def plot_expvals_differences():
    names = ["LiH", "HF"]
    basis_sets = ["6-31g", "6-31g*", "cc-pVDZ"] #["6-31g", "6-31g*"]

    expval = "dipole"
    table = {"name": [], "basis": [], "CCSD": [], "QCCSD": [], "FCI": []}
    for name in names:
        for basis in basis_sets:
            for method in ["CCSD", "QCCSD", "FCI"]:
                data = load_json(f"{name}_{basis}_{method.lower()}")
                if expval == "dipole":
                    table[f"{method}"].append(data["dipole"][2])
                if expval == "quadropole_e":
                    table[f"{method}"].append(data["quadropole_e"][2][2])
                        
            table["name"].append(name)
            table["basis"].append(basis)

    table_df = pd.DataFrame(table)
    
    system = "HF"

    df = table_df[table_df["name"] == system]
    
    species = df["basis"].to_list()

    ccsd = df["CCSD"] - df["FCI"]
    qccsd = df["QCCSD"] - df["FCI"]
    fci = df["FCI"]

    print(fci)
    exit()
    data = {
        "CCSD": ccsd/fci * 100,
        "QCCSD": qccsd/fci * 100,
    }

    colors = {
        "CCSD": "red",
        "QCCSD": "green",
    }

    x = np.arange(3)  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')

    for attribute, measurement in data.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute, color=colors[attribute],edgecolor="black")
        ax.bar_label(rects, padding=3, fmt=lambda x: rf"{x:.3f} \%")
        multiplier += 1
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(r'Dipole moment relative error to FCI [\%]')
    ax.set_xticks(x + width/2, species)
    ax.set_xlabel("Basis set")
    ax.legend(loc='upper left', ncols=3)
    # plt.gca().invert_yaxis()
    # ax.set_ylim(0, 250)

    fig.savefig("test.pdf", transparent=True)
    plt.show()
    # print(table_df)
    # for method in ["CCSD", "QCCSD"]:
    #     table_df[f"{method}d"] = 1000*(table_df[f"{method}"] - table_df[f"FCI"])

    #     table_df[f"{method}d"] = table_df[f"{method}d"].map(lambda x: f"{x:.3f}")


def plot_expvals_differences_partitioned():
    fci = 0.715342
    data = {
        "CCSD": np.array([-0.295, 2.562, -3.426, 2.827])*1e-3/fci * 100,
        "QCCSD": np.array([-0.287, 2.662, 1.521, 1.017])*1e-3/fci * 100,
    }

    species = ["$\gamma_{ij}$", "$\gamma_{ab}$", "$\gamma_{ia}$", "$\gamma_{ai}$"]

    print(
        data["CCSD"].sum(),
        data["QCCSD"].sum(),
    )

    x = np.arange(4)  # the label locations
    width = 0.4  # the width of the bars
    multiplier = 0

    colors = {
        "CCSD": "red",
        "QCCSD": "green",
    }
    fig, ax = plt.subplots(layout='constrained')

    for attribute, measurement in data.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute, color=colors[attribute],edgecolor="black")
        ax.bar_label(rects, padding=3, fmt=lambda x: rf"{x:.3f}\%")
        multiplier += 1
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(r'Dipole moment relative error to FCI [\%]')
    ax.set_xticks(x + width/2, species)
    ax.set_xlabel("Block contribution")
    ax.legend(loc='lower left', ncols=3)
    fig.savefig("test.pdf", transparent=True)
    plt.show()

def plot_expvals_differences_partitioned2():
    fci = 0.715342

    data = {
        "oo": np.array([-0.295,-0.287])*1e-3/fci * 100, 
        "vv": np.array([2.562,2.662])*1e-3/fci * 100,
        "ov": np.array([-3.426, 1.521])*1e-3/fci * 100,
        "vo": np.array([2.827, 1.017])*1e-3/fci * 100,
    }

    species = ["CCSD", "QCCSD", ]


    x = np.arange(2)  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0

    colors = {
        "oo": "red",
        "vv": "green",
        "vo": "blue",
        "ov": "purple",
    }
    fig, ax = plt.subplots(layout='constrained')

    for attribute, measurement in data.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute, color=colors[attribute],edgecolor="black")
        ax.bar_label(rects, padding=3, fmt=lambda x: rf"{x:.3f} \%")
        multiplier += 1
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(r'Dipole moment relative error to FCI [\%]', fontsize=12)
    ax.set_xticks(x + width/2, species)
    ax.set_xlabel("Method")
    ax.legend(loc='upper left', ncols=3)

    plt.show()

def wtf():
    name = "HF"
    geom = geometries[name]
    basis = "cc-pVDZ"

    b = qt.PyscfBasis(geom, basis, restricted=False).pyscf_hartree_fock()

    ccsd = qt.CCSD(b).run(include_l=True)
    qccsd = qt.QCCSD(b).run()

    rho_ccsd = ccsd.one_body_density()
    rho_qccsd = qccsd.one_body_density()
    rho_fci = run_fci_density_matrix(geom, basis)

    oo_fci, vv_fci, ov_fci, vo_fci = partial_expvals(rho_fci, b.r, b.o, b.v, name, vocal=False)
            
    oo, vv, ov, vo = partial_expvals(rho_ccsd, b.r, b.o, b.v, name, vocal=False)
    

    diff_oo = (oo - oo_fci)*1000
    diff_vv = (vv - vv_fci)*1000
    diff_ov = (ov - ov_fci)*1000
    diff_vo = (vo - vo_fci)*1000
    
    print(f"CCSD {name} {basis}, sum = {diff_oo+diff_vv+diff_ov+diff_vo:.3}")
    print(f"{diff_oo:.3f} & {diff_vv:.3f} & {diff_ov:.3f} & {diff_vo:.3f}")

    oo, vv, ov, vo = partial_expvals(rho_qccsd, b.r, b.o, b.v, name, vocal=False)

    diff_oo = (oo - oo_fci)*1000
    diff_vv = (vv - vv_fci)*1000
    diff_ov = (ov - ov_fci)*1000
    diff_vo = (vo - vo_fci)*1000
    
    print(f"QCCSD {name} {basis}, sum = {diff_oo+diff_vv+diff_ov+diff_vo:.3}")
    print(f"{diff_oo:.3f} & {diff_vv:.3f} & {diff_ov:.3f} & {diff_vo:.3f}")
    

if __name__ == "__main__":
    # density_matrix_asymmetry(run=False)
    # compare_with_fci(run=False)

    # run_density_test()
    # run_expval_test()

    # run_density_diff(run=False, show=False)
    # plot_expvals_differences()
    plot_expvals_differences_partitioned()

    # wtf()
    # run_density_diff(run=False)