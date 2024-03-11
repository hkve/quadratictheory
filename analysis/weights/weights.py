import clusterfock as cf
from geometries import LiH_ccpVDZ
from fci_pyscf import fci_pyscf

import matplotlib.pyplot as plt
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
        "vocal": False
    }

    default.update(kwargs)

    vocal = default["vocal"]

    geomtries = pack_as_list(geomtries)
    weights = []
    for geometry in geomtries:
        b = run_hf(geometry, basis, restricted=False)
        W = run_cc(b, CC, vocal=vocal)
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

if __name__ == "__main__":

    geom = LiH_ccpVDZ["geometry"]
    R = LiH_ccpVDZ["R"]
    weights_CC = run_weights_CC(geom, "cc-pVDZ", cf.QCCSD, vocal=True)
    weights_FCI = run_weights_FCI(geom, "cc-pVDZ")

    make_diatom_table(weights_CC, R, weights_FCI)