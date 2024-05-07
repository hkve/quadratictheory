import clusterfock as cf
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plot_utils as pl

def run_hf(geometry, basis, restricted=False, **kwargs):
    default = {
        "charge": 0,
        "tol": 1e-4,
    }

    default.update(kwargs)

    charge = default["charge"]
    tol = default["tol"]

    b = cf.PyscfBasis(geometry, basis, charge=charge)
    b.pyscf_hartree_fock(tol=tol, max_cycle=500)

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
    
    return W

def append_to_file(filename, W):
    fields = [key for key in W.keys()]
    row = [value for value in W.values()]

    with open(filename,'a') as file:
        writer = csv.writer(file)
        writer.writerow(row)

def calculate_dissociation(filename, distances, geometries, basis, CC, **kwargs):
    for r, geom in zip(distances, geometries):
        b = run_hf(geom, basis, restricted=False)
        W = run_cc(b, CC)

        W["r"] = r

        print(f"Done {geom}, {CC.__name__}")
        append_to_file(filename, W)


def N2_ccPVTZ(run=False):
    distances = np.arange(2.00,7.00+0.1,0.25)
    geometries = [f"N 0 0 0; N 0 0 {r}" for r in distances]
    
    if run:
        calculate_dissociation("dat/N2_CCSD_cc-pVTZ.csv", distances, geometries, "cc-pVTZ", cf.CCSD)
        calculate_dissociation("dat/N2_QCCSD_cc-pVTZ.csv", distances, geometries, "cc-pVTZ", cf.QCCSD)

    df_ccsd = pd.read_csv("dat/N2_CCSD_cc-pVTZ.csv")
    df_qccsd = pd.read_csv("dat/N2_QCCSD_cc-pVTZ.csv")

    plot_weights(df_ccsd, df_qccsd, weights=["0", "D"], y_max=3.0, filename="N2_weights_CCSD_cc-pVTZ.pdf")

def get_color(name):
    if "CC" in name:
        if "Q" in name:
            return pl.colors[2]
        else:
            return pl.colors[1]
    else:
        return pl.colors[0]
   

def plot_weights(df1, df2, weights=["0", "S", "D"], **kwargs):
    default = {
        "names": ["CCSD", "QCCSD"],
        "legend_cols": 2,
        "y_max": None,
        "filename": None,
    }

    default.update(kwargs)

    names = default["names"]
    legend_cols = default["legend_cols"]
    y_max = default["y_max"]
    filename = default["filename"]

    ls = {
        "0": "solid",
        "S": "dotted",
        "D": "dashed",
        "T": "dashdot",
        "Q": (0, (3, 1, 1, 1, 1, 1)),
    }

    fig, ax = plt.subplots()
    for i, (name, df) in enumerate(zip(names, [df1, df2])):
        r = df["r"]

        for w in weights:
            label = "$W_{" + w + "}^{" + name + "}$"
            ax.plot(r, df[w], label=label, ls=ls[w], color=get_color(name))

    if y_max is not None:
        ylims = ax.get_ylim()
        ylims = (ylims[0], y_max)
        ax.set_ylim(ylims)

    ax.set(xlabel=r"$R$ [au]", ylabel=r"$W_\mu$")
    ax.legend(ncols=legend_cols, loc="upper left")

    if filename:
        pl.save(filename)

    plt.show()

if __name__ == "__main__":
    N2_ccPVTZ()