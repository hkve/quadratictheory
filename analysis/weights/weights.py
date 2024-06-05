import quadratictheory as qt
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

    b = qt.PyscfBasis(geometry, basis, charge=charge)
    b.pyscf_hartree_fock(tol=tol)

    if not restricted:
        b.from_restricted()

    return b

def run_cc(basis, CC, tol, maxiters, mixer, vocal):
    is_quadratic = False
    is_SD = False
    run_kwargs = {"tol": tol, "vocal": vocal, "maxiters": maxiters}
    if not CC.__name__.startswith("Q"):
        run_kwargs = {"include_l": True}
    else:
        is_quadratic = True
    
    if "SD" in CC.__name__:
        is_SD = True

    cc = CC(basis)
    cc.mixer = mixer
    cc.run(**run_kwargs)

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
        print("New attempt")
        new_mixer = qt.mix.SoftStartDIISMixer(alpha=mixer.alpha + 0.1, start_DIIS_after=mixer.start_DIIS_after+10, n_vectors=mixer.n_vectors+4)
        return run_cc(basis, CC, tol, maxiters, mixer, vocal)

    return W

def run_weights_CC(geomtries, basis, CC, **kwargs):
    default = {
        "vocal": False,
        "hf_tol": 1e-6,
        "cc_tol": 1e-4,
        "mixer": qt.mix.RelaxedMixer(alpha=0.5),
        "maxiters": 300
    }

    default.update(kwargs)

    vocal = default["vocal"]
    hf_tol = default["hf_tol"]
    cc_tol = default["cc_tol"]
    mixer = default["mixer"]
    maxiters = default["maxiters"]

    geomtries = pack_as_list(geomtries)
    weights = []
    for geometry in geomtries:
        b = run_hf(geometry, basis, restricted=False, tol=hf_tol)
        W = run_cc(b, CC, vocal=vocal, tol=cc_tol, maxiters=maxiters, mixer=mixer)
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

def perform_save(names, weigths):

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

def get_color(name):
    if "CC" in name:
        if "Q" in name:
            return pl.colors[2]
        else:
            return pl.colors[1]
    else:
        return pl.colors[0]
    
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

    ls = {
        "0": "solid",
        "S": "dotted",
        "D": "dashed",
        "T": "dashdot",
        "Q": (0, (3, 1, 1, 1, 1, 1)),
    }

    fig, ax = plt.subplots()
    for i, name in enumerate(names):
        W = dict(weights[i])
        R = W.pop("R")

        new_keys = list(set(W.keys()).difference(drop_weights))
        new_keys = [tuple for x in ["0", "S", "D", "T", "Q"] for tuple in new_keys if tuple[0] == x]
        W = {k: W[k] for k in new_keys}

        for j, (k, v) in enumerate( W.items()):
            label = "$W_{" + k + "}^{" + name + "}$"
            ax.plot(R, v, label=label, ls=ls[k], color=get_color(name))

    if y_max is not None:
        ylims = ax.get_ylim()
        ylims = (ylims[0], y_max)
        ax.set_ylim(ylims)

    ax.set(xlabel=r"$R$ [$a_0$]", ylabel=r"$W_\mu$")
    ax.legend(ncols=legend_cols, loc="upper left")

    if filename:
        pl.save(filename)

    plt.show()

def plot_h2o(run, standard="CCD", quad="QCCD", drop_weights=["T", "Q"], save=False):
    mets_map = {"CCD": qt.CCD, "CCSD": qt.CCSD} 
    qmets_map = {"QCCD": qt.QCCD, "QCCSD": qt.QCCSD} 
    all_mets = [standard, quad] + ["FCI"]

    names = [f"H20_{met}" for met in all_mets]
    if run:
        distances  =  [1.0, 1.3, 1.5, 1.7, 1.9, 2.1, 2.3, 2.5, 2.7, 2.8, 3.1, 3.3, 3.5, 3.7, 3.9, 4.1, 4.3, 4.5]
        geoms = disassociate_h2o(distances)

        weights_CC = run_weights_CC(geoms, "sto-3g", mets_map[standard]) 
        weights_QCC = run_weights_CC(geoms, "sto-3g", qmets_map[quad])
        weights_FCI = run_weights_FCI(geoms, "sto-3g")

        weights = [format_weigths(W, distances) for W in [weights_CC, weights_QCC, weights_FCI]]

        perform_save(names, weights)
    else:
        weights = load(names)

    filename = f"H2O_weights_{standard}" if save else None
    plot_weights(weights, names=all_mets, y_max=1.5, filename=filename, drop_weights=drop_weights)

def plot_n2(run, standard="CCD", quad="QCCD", drop_weights=["T", "Q"], save=False):
    mets_map = {"CCD": qt.CCD, "CCSD": qt.CCSD} 
    qmets_map = {"QCCD": qt.QCCD, "QCCSD": qt.QCCSD} 
    all_mets = [standard, quad] + ["FCI"]

    names = [f"N2_{met}" for met in all_mets]

    if run:
        distances = np.arange(2.00,7.00+0.1,0.25)

        distances_ccd_1 = np.arange(2.00, 5.00+0.1, 0.25)
        distances_ccd_2 = np.arange(5.25, 7.00+0.1, 0.25)

        distances_qccd_1 = np.arange(2.00, 5.5+0.1, 0.25)
        distances_qccd_2 = np.arange(5.75, 7.0+0.1, 0.25)
        

        geoms = disassociate_2dof("N", "N", distances)
        geoms_ccd_1 = disassociate_2dof("N", "N", distances_ccd_1)
        geoms_ccd_2 = disassociate_2dof("N", "N", distances_ccd_2)

        geoms_qccd_1 = disassociate_2dof("N", "N", distances_qccd_1)
        geoms_qccd_2 = disassociate_2dof("N", "N", distances_qccd_2)
        
        cc_mixer_1 = qt.mix.RelaxedMixer(alpha=0.5)
        cc_mixer_2 = qt.mix.SoftStartDIISMixer(alpha=0.90, start_DIIS_after=40, n_vectors=10)
        
        qcc_mixer_1 = qt.mix.SoftStartDIISMixer(alpha=0.75, start_DIIS_after=10, n_vectors=5)
        qcc_mixer_2 = qt.mix.SoftStartDIISMixer(alpha=0.9, start_DIIS_after=30, n_vectors=20)

        # weights_CC_1 = run_weights_CC(geoms_ccd_1, "sto-3g", mets_map[standard], mixer=cc_mixer_1)
        # weights_CC_2 = run_weights_CC(geoms_ccd_2, "sto-3g", mets_map[standard], mixer=cc_mixer_2)
        # weights_CC = weights_CC_1 + weights_CC_2

        weights_QCC_1 = run_weights_CC(geoms_qccd_1, "sto-3g", qmets_map[quad],maxiters=500, mixer=qcc_mixer_1)
        weights_QCC_2 = run_weights_CC(geoms_qccd_2, "sto-3g", qmets_map[quad],maxiters=500, mixer=qcc_mixer_2)
        weights_QCC =  weights_QCC_1 + weights_QCC_2

        # weights_FCI = run_weights_FCI(geoms, "sto-3g")

        # weights = [format_weigths(W, distances) for W in [weights_CC, weights_QCC, weights_FCI]]
        weights = [format_weigths(W, distances) for W in [weights_QCC]]

        perform_save(names[1], weights)
    else:
        weights = load(names)

    filename = f"N2_weights_{standard}" if save else None
    plot_weights(weights, names=all_mets, filename=filename, drop_weights=drop_weights, y_max=2.8)

if __name__ == "__main__":
    # geom = LiH_ccpVDZ["geometry"]
    # R = LiH_ccpVDZ["R"]
    # weights_CC = run_weights_CC(geom, "cc-pVDZ", qt.QCCSD, vocal=True)
    # weights_FCI = run_weights_FCI(geom, "cc-pVDZ")

    # make_diatom_table(weights_CC, R, weights_FCI)
    # plot_h2o(False, standard="CCD", quad="QCCD", save=False)
    # plot_h2o(False, standard="CCSD", quad="QCCSD", drop_weights=["T", "Q"], save=False)
    
    plot_n2(False, standard="CCD", quad="QCCD", drop_weights=["S", "T", "Q"], save=True)
    # plot_n2(True, standard="CCSD", quad="QCCSD", drop_weights=["T", "Q"], save=False)