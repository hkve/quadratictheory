import clusterfock as cf
from geometries import LiH_ccpVDZ, disassociate_2dof, disassociate_h2o
from fci_pyscf import fci_pyscf

import matplotlib.pyplot as plt
import numpy as np
import plot_utils as pl
import pandas as pd

pack_as_list = lambda x: x if type(x) in (list, tuple) else [x]

def run_hf(geometry, basis, restricted=False, **kwargs):
    default = {
        "charge": 0,
        "tol": 1e-10,
    }

    default.update(kwargs)

    charge = default["charge"]
    tol = default["tol"]

    b = cf.PyscfBasis(geometry, basis, charge=charge)
    b.pyscf_hartree_fock(tol=tol)

    if not restricted:
        b.from_restricted()

    return b

def run_cc(basis, CC, **kwargs):
    default = {
        "tol": 1e-6,
        "vocal": False,
        "lowest_allowed_tol": 1e-4
    }

    default.update(kwargs)

    tol = default["tol"]
    vocal = default["vocal"]
    lowest_allowed_tol = default["lowest_allowed_tol"]

    is_quadratic = False
    is_SD = False
    run_kwargs = {"tol": tol, "vocal": vocal}
    if not CC.__name__.startswith("Q"):
        run_kwargs = {"include_l": True}
    else:
        is_quadratic = True
    
    if "SD" in CC.__name__:
        is_SD = True

    cc = CC(basis).run(**run_kwargs)

    W = {}
    if cc.t_info["converged"] and cc.l_info["converged"]:
        W["0"], W["D"] = cc.reference_weights(), 0.25*cc.doubles_weights().sum()

        if is_SD:
            W["S"] = cc.singles_weights().sum()
        
        if is_quadratic:
            W["Q"] = cc.quadruple_weight()
            if is_SD: 
                W["T"] = cc.triples_weight()
    else:
        lowest_tol = cc.get_lowest_norm()
        if lowest_tol > lowest_allowed_tol:
            raise ValueError("UPSID")
        else:
            default[tol] = lowest_tol*1.1
            print(f"Did not converge, rerun with tol = {lowest_tol}")
            run_cc(basis, CC, **default)

    return W

def run_weights_CC(geomtries, basis, CC, **kwargs):
    default = {
        "vocal": False,
        "hf_tol": 1e-6,
        "cc_tol": 1e-6,
    }

    default.update(kwargs)

    vocal = default["vocal"]
    hf_tol = default["hf_tol"]
    cc_tol = default["cc_tol"]

    geomtries = pack_as_list(geomtries)
    weights = []
    for geometry in geomtries:
        b = run_hf(geometry, basis, restricted=False, tol=hf_tol)
        W = run_cc(b, CC, vocal=vocal, tol=cc_tol)
        print(f"Done {geometry}, {CC.__name__}")
        weights.append(W)

    return weights

def run_weights_FCI(geomtries, basis, **kwargs):
    geomtries = pack_as_list(geomtries)
    weights = []
    for geometry in geomtries:
        E, W = fci_pyscf(geometry, basis)
        print(f"Done {geometry} FCI")
        weights.append(W)

    return weights

def _rename_weigths(weights):
    weight_cols = list(weights[0].keys())
    formatted_weigths = {col: [] for col in weight_cols}
    for W in weights:
        for k, v in W.items():
            formatted_weigths[k].append(v)

    renamed_weigths = {f"$W_{k}$": v for k, v in formatted_weigths.items()}

    return renamed_weigths

def format_weigths(weights, R):
    weights = pack_as_list(weights)
    keys = list(weights[0].keys())

    new_weights = {k: [] for k in keys}

    for weight in weights:
        for k in keys:
            new_weights[k].append(weight[k])

    new_weights["R"]  = R
    return new_weights

def save(names, weigths):

    names = pack_as_list(names)
    weigths = pack_as_list(weigths)

    for name, weigth in zip(names, weigths):
        np.savez(f"dat/{name}",  **weigth)


def load(names):
    names = pack_as_list(names)
    weigths = []

    for name in names:
        weigth = np.load(f"dat/{name}.npz", allow_pickle=True)
        weigths.append(weigth)

    return weigths

def make_diatom_table(weights_CC, R, weights_FCI=None):
    df_r = pd.DataFrame({"R": R})
    df_cc = pd.DataFrame(weights_CC)
    df_fci = pd.DataFrame(weights_FCI)

    df1 = pd.concat([df_r, df_cc], axis=1)
    df2 = pd.concat([df_r, df_fci], axis=1)

    print(
        df1.to_latex()
    )

    print(
        df2.to_latex()
    )
    # if weights_FCI is not None:
    #     fci_cols = ["0", "S", "D"]
    #     cols = list(set(df_cc.columns).intersection(set(fci_cols)))        
    #     df_fci = pd.DataFrame(weights_FCI)
    #     df_fci = df_fci[cols]
    #     df_cc = pd.concat([df_cc, df_fci], keys=['CC', 'FCI'], axis=1)
    #     df_cc.swaplevel(0,1,1)
    
        # col_order = []
        # for v in weights_CC.keys():
        #     for m in ["CC", "FCI"]:
        #         col_order.append((v, m))

        # df_cc = df_cc.reindex(columns=col_order)

    # df = pd.concat([df_r, df_cc], axis=1)

    # print(
    #     df.to_latex()
    # )
    # from IPython import embed
    # embed()

def plot_weights(weights, drop_weights=["S", "Q"], **kwargs):
    default = {
        "names": ["!"]*len(weights),
        "legend_cols": 3,
        "y_max": None,
        "filename": None,
    }

    default.update(kwargs)

    names = default["names"]
    legend_cols = default["legend_cols"]
    y_max = default["y_max"]
    filename = default["filename"]

    ls = ["solid", "dashed", "dotted", "dasheddot"]

    fig, ax = plt.subplots()
    for i, name in enumerate(names):
        W = dict(weights[i])
        R = W.pop("R")
        new_keys = list(set(W.keys()).difference(drop_weights))
        new_keys = [tuple for x in ["0", "S", "D", "T", "Q"] for tuple in new_keys if tuple[0] == x]
        W = {k: W[k] for k in new_keys}

        for j, (k, v) in enumerate( W.items()):
            label = "$W_{" + k + "}^{" + name + "}$"
            ax.plot(R, v, label=label, ls=ls[j], color=pl.colors[i])


    if y_max is not None:
        ylims = ax.get_ylim()
        ylims = (ylims[0], y_max)
        ax.set_ylim(ylims)

    ax.set(xlabel=r"$R$ [au]", ylabel=r"$W_\mu$")
    ax.legend(ncols=legend_cols, loc="upper left")

    if filename:
        pl.save(filename)

    plt.show()

def plot_h2o():
    run = False

    names = [f"H20_{met}" for met in ["CCD", "QCCD", "FCI"]]
    if run:
        distances  =  [1.0, 1.3, 1.5, 1.7, 1.9, 2.1, 2.3, 2.5, 2.7, 2.8, 3.1, 3.3, 3.5, 3.7, 3.9, 4.1, 4.3, 4.5]
        geoms = disassociate_h2o(distances)

        weights_CC = run_weights_CC(geoms, "sto-3g", cf.CCD) 
        weights_QCC = run_weights_CC(geoms, "sto-3g", cf.QCCD)
        weights_FCI = run_weights_FCI(geoms, "sto-3g")

        weights = [format_weigths(W, distances) for W in [weights_CC, weights_QCC, weights_FCI]]

        save(names, weights)
    else:
        weights = load(names)

    plot_weights(weights, names=["CCD", "QCCD", "FCI"], y_max=1.4, filename="H2O_weights")

def plot_n2():
    run = False

    names = [f"N2_{met}" for met in ["CCD", "QCCD", "FCI"]]
    if run:
        distances1 = np.array([1.2, 1.4, 1.6, 1.7, 1.8, 1.9])
        distances2 = np.arange(2.0, 3.8+0.1, 0.1)

        distances = np.r_[distances1, distances2]
        geoms = disassociate_2dof("N", "N", distances)

        weights_CC = run_weights_CC(geoms, "sto-3g", cf.CCD) 
        weights_QCC = run_weights_CC(geoms, "sto-3g", cf.QCCD)
        weights_FCI = run_weights_FCI(geoms, "sto-3g")

        weights = [format_weigths(W, distances) for W in [weights_CC, weights_QCC, weights_FCI]]

        save(names, weights)
    else:
        weights = load(names)

    plot_weights(weights, names=["CCD", "QCCD", "FCI"], filename="N2_weights")

if __name__ == "__main__":
    # geom = LiH_ccpVDZ["geometry"]
    # R = LiH_ccpVDZ["R"]
    # weights_CC = run_weights_CC(geom, "cc-pVDZ", cf.QCCSD, vocal=True)
    # weights_FCI = run_weights_FCI(geom, "cc-pVDZ")

    # make_diatom_table(weights_CC, R, weights_FCI)
    plot_h2o()
    plot_n2()