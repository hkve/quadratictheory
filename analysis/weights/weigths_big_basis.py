import clusterfock as cf
import csv
import numpy as np

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


def N2_ccPVTZ():
    distances = np.arange(2.00,7.00+0.1,0.25)
    geometries = [f"N 0 0 0; N 0 0 {r}" for r in distances]
    calculate_dissociation("dat/N2_cc-pVTZ.csv", distances, geometries, "cc-pVTZ", cf.CCSD)
    calculate_dissociation("dat/N2_cc-pVTZ.csv", distances, geometries, "cc-pVTZ", cf.QCCSD)

if __name__ == "__main__":
    N2_ccPVTZ()